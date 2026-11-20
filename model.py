"""Next-day (n+1) close prediction with rigorous out-of-sample validation.

Features (all known at close of day t - no look-ahead):
  ret_1, ret_5      lagged close returns
  volatility_10     rolling 10-day std of returns
  volume_change     day volume vs 10-day mean
  social_sentiment  engagement-weighted daily social sentiment (0 if none)
  mention_count     log1p social mention volume
  news_sentiment    daily news sentiment (0 if none)
  has_social        1 if any social data existed that day (lets the model
                    distinguish "neutral" from "no data")

Target: the next-day RETURN (close t+1 / close t - 1). The predicted close
is recovered as close_t * (1 + predicted return). Predicting returns rather
than price levels matters: tree models cannot extrapolate price levels and
otherwise collapse to predicting the historical mean price, which produces
absurd double-digit "predicted moves" after any trend (verified empirically
in this project - see git history / run logs).

Validation: expanding-window walk-forward. For each ticker, train on the
first MIN_TRAIN_DAYS, predict the next day, expand, repeat. Metrics are
computed ONLY on out-of-sample predictions and compared against a naive
baseline (tomorrow = today). Per-ticker models throughout.

Model: GradientBoostingRegressor (scikit-learn) - strong on small tabular
data, no GPU / no extra services needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from config import DATA_DIR, MIN_TRAIN_DAYS, RANDOM_STATE

FEATURES = ["ret_1", "ret_5", "volatility_10", "volume_change",
            "social_sentiment", "mention_count", "news_sentiment", "has_social"]


def build_features(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret_1"] = g["close"].pct_change(1)
        g["ret_5"] = g["close"].pct_change(5)
        g["volatility_10"] = g["ret_1"].rolling(10).std()
        g["volume_change"] = g["volume"] / g["volume"].rolling(10).mean() - 1.0
        g["target_close"] = g["close"].shift(-1)
        g["target_return"] = g["target_close"] / g["close"] - 1.0
        g["target_date"] = g["date"].shift(-1)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    if sentiment is not None and not sentiment.empty:
        s = sentiment.copy()
        s["date"] = pd.to_datetime(s["date"])
        df = df.merge(s[["date", "ticker", "social_sentiment", "mention_count",
                         "news_sentiment"]],
                      on=["date", "ticker"], how="left")
    else:
        df["social_sentiment"] = np.nan
        df["mention_count"] = 0
        df["news_sentiment"] = np.nan

    df["has_social"] = (df["mention_count"].fillna(0) > 0).astype(int)
    df["mention_count"] = np.log1p(df["mention_count"].fillna(0))
    df["social_sentiment"] = df["social_sentiment"].fillna(0.0)
    df["news_sentiment"] = df["news_sentiment"].fillna(0.0)
    return df


def _new_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                     learning_rate=0.05, subsample=0.9,
                                     random_state=RANDOM_STATE)


def walk_forward(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window walk-forward for ONE ticker. Returns OOS predictions."""
    d = df_ticker.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)
    preds = []
    if len(d) <= MIN_TRAIN_DAYS:
        return pd.DataFrame(preds)
    for i in range(MIN_TRAIN_DAYS, len(d)):
        train = d.iloc[:i]
        model = _new_model()
        model.fit(train[FEATURES], train["target_return"])
        row = d.iloc[i]
        pred_ret = float(model.predict(row[FEATURES].to_frame().T)[0])
        preds.append({
            "ticker": row["ticker"], "date": row["target_date"],
            "actual_close": float(row["target_close"]),
            "predicted_close": float(row["close"]) * (1.0 + pred_ret),
            "naive_close": float(row["close"]),
            "prev_close": float(row["close"]),
        })
    return pd.DataFrame(preds)


def evaluate(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"n_oos": 0, "rmse": None, "naive_rmse": None, "mape": None,
                "direction_accuracy": None, "naive_direction_accuracy": None}
    rmse = float(np.sqrt(mean_squared_error(preds["actual_close"], preds["predicted_close"])))
    naive_rmse = float(np.sqrt(mean_squared_error(preds["actual_close"], preds["naive_close"])))
    # MAPE is scale-free, so it stays meaningful when pooled across tickers
    # whose prices differ by 4 orders of magnitude (HKD vs KRW listings).
    mape = float((np.abs(preds["actual_close"] - preds["predicted_close"])
                  / preds["actual_close"]).mean() * 100)
    actual_dir = np.sign(preds["actual_close"] - preds["prev_close"])
    pred_dir = np.sign(preds["predicted_close"] - preds["prev_close"])
    # Ties (no move) count as correct only if both are ties.
    dir_acc = float((actual_dir == pred_dir).mean())
    return {"n_oos": int(len(preds)), "rmse": rmse, "naive_rmse": naive_rmse,
            "mape": mape, "direction_accuracy": dir_acc,
            "naive_direction_accuracy": 0.0}  # naive predicts no move by construction


def predict_next(df: pd.DataFrame) -> pd.DataFrame:
    """Train on all available history per ticker; predict the next close (n+1)."""
    rows = []
    for ticker, g in df.groupby("ticker"):
        d = g.dropna(subset=FEATURES).copy()
        train = d.dropna(subset=["target_return"])
        if len(train) < MIN_TRAIN_DAYS or d.empty:
            # Not enough history: fall back to naive (last close) and say so.
            last = g.sort_values("date").iloc[-1]
            rows.append({"ticker": ticker, "exchange": last["exchange"],
                         "last_close": float(last["close"]),
                         "last_date": last["date"],
                         "predicted_close_next": float(last["close"]),
                         "predicted_return_pct": 0.0,
                         "direction": "FLAT (insufficient history - naive)",
                         "model_used": "naive-fallback"})
            continue
        model = _new_model()
        model.fit(train[FEATURES], train["target_return"])
        latest = d.iloc[-1]  # most recent day with full features
        pred_ret = float(model.predict(latest[FEATURES].to_frame().T)[0])
        last_close = float(latest["close"])
        pred = last_close * (1.0 + pred_ret)
        ret = pred_ret * 100.0
        direction = "UP" if ret > 0.1 else ("DOWN" if ret < -0.1 else "FLAT")
        rows.append({"ticker": ticker, "exchange": latest["exchange"],
                     "last_close": last_close, "last_date": latest["date"],
                     "predicted_close_next": pred,
                     "predicted_return_pct": ret,
                     "direction": direction, "model_used": "gradient-boosting"})
    return pd.DataFrame(rows)


def position_signal(predicted_return_pct: float, sentiment_score: float | None) -> str:
    """Simple rule-based paper/research signal. NOT financial advice.

    LONG  : model predicts > +0.5% AND sentiment is not negative
    SHORT : model predicts < -0.5% AND sentiment is not positive
    NONE  : otherwise (model and sentiment disagree, or move too small)
    """
    s = sentiment_score if sentiment_score is not None else 0.0
    if predicted_return_pct > 0.5 and s >= -0.1:
        return "LONG (paper)"
    if predicted_return_pct < -0.5 and s <= 0.1:
        return "SHORT (paper)"
    return "NONE"


def main() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    sent_path = DATA_DIR / "daily_sentiment.csv"
    sentiment = pd.read_csv(sent_path, parse_dates=["date"]) if sent_path.exists() else pd.DataFrame()
    df = build_features(prices, sentiment)

    all_preds, metrics_by_ticker = [], {}
    for ticker, g in df.groupby("ticker"):
        p = walk_forward(g)
        if not p.empty:
            all_preds.append(p)
        metrics_by_ticker[ticker] = evaluate(p)
    oos = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    overall = evaluate(oos)
    next_preds = predict_next(df)

    oos.to_csv(DATA_DIR / "oos_predictions.csv", index=False)
    next_preds.to_csv(DATA_DIR / "next_day_predictions.csv", index=False)

    print("[model] ===== Out-of-sample (walk-forward) results =====")
    print(f"[model] total OOS predictions: {overall['n_oos']}")
    if overall["rmse"] is not None:
        print(f"[model] pooled RMSE: {overall['rmse']:,.2f}  | naive RMSE: {overall['naive_rmse']:,.2f}")
        print(f"[model] pooled MAPE: {overall['mape']:.2f}%  | direction accuracy: {overall['direction_accuracy']:.1%}")
        for tk, m in sorted(metrics_by_ticker.items()):
            if m["rmse"] is not None:
                print(f"[model]   {tk}: n={m['n_oos']} RMSE {m['rmse']:,.2f} "
                      f"(naive {m['naive_rmse']:,.2f}), MAPE {m['mape']:.2f}%, "
                      f"dir-acc {m['direction_accuracy']:.1%}")
    print("[model] ===== n+1 predictions =====")
    for _, r in next_preds.iterrows():
        print(f"[model] {r['ticker']}: last {r['last_close']:,.2f} -> "
              f"predicted {r['predicted_close_next']:,.2f} "
              f"({r['predicted_return_pct']:+.2f}%, {r['direction']}, {r['model_used']})")
    return next_preds, {"overall": overall, "by_ticker": metrics_by_ticker}, oos


if __name__ == "__main__":
    main()
