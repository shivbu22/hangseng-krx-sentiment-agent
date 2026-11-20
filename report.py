"""Visual HTML report.

Per exchange: a table of tickers sorted by social mentions, then sentiment
score, showing current price, sentiment + direction, paper position signal,
support/resistance and the model's n+1 prediction. Plus matplotlib charts
(sentiment-score bar chart, mentions bar chart, price history per exchange)
and the walk-forward validation metrics. Direction arrows are plain text
(UP / DOWN / FLAT) so the report renders anywhere.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CHARTS_DIR, DATA_DIR, EXCHANGES, REPORTS_DIR, TICKERS
from levels import compute_levels
from model import position_signal

NAME = {t.symbol: t.name for t in TICKERS}


def _latest_sentiment(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Most recent daily sentiment row per ticker + total mentions (10d window)."""
    if sentiment is None or sentiment.empty:
        return pd.DataFrame(columns=["ticker", "social_sentiment", "news_sentiment",
                                     "mention_count_total", "mention_count_latest"])
    s = sentiment.copy()
    s["date"] = pd.to_datetime(s["date"])
    latest = s.sort_values("date").groupby("ticker").tail(1)[
        ["ticker", "social_sentiment", "news_sentiment", "mention_count"]]
    totals = s.groupby("ticker")["mention_count"].sum().rename("mention_count_total")
    out = latest.rename(columns={"mention_count": "mention_count_latest"}).merge(
        totals, on="ticker", how="left")
    return out


def _charts(summary: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    if not summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        s = summary.sort_values("social_sentiment")
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in s["social_sentiment"].fillna(0)]
        axes[0].barh(s["ticker"], s["social_sentiment"].fillna(0), color=colors)
        axes[0].axvline(0, color="#333", lw=0.8)
        axes[0].set_title("Latest social sentiment score by ticker")
        axes[0].set_xlabel("sentiment (-1 bearish .. +1 bullish)")
        m = summary.sort_values("mention_count_total")
        axes[1].barh(m["ticker"], m["mention_count_total"].fillna(0), color="#2e6da4")
        axes[1].set_title("Social mentions (last 10 days, all sources)")
        axes[1].set_xlabel("posts")
        fig.tight_layout()
        p = CHARTS_DIR / "sentiment_overview.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)

    for exchange in EXCHANGES:
        g = prices[prices["exchange"] == exchange]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for ticker, tg in g.groupby("ticker"):
            tg = tg.sort_values("date")
            ax.plot(tg["date"], tg["close"] / tg["close"].iloc[0] * 100,
                    label=f"{ticker} {NAME.get(ticker, '')}", lw=1.4)
        ax.set_title(f"{exchange} - normalised close (first day = 100)")
        ax.set_ylabel("normalised close")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = CHARTS_DIR / f"prices_{exchange}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)
    return made


def _direction_word(sent: float | None) -> str:
    if sent is None or pd.isna(sent):
        return "NO DATA"
    if sent > 0.1:
        return "BULLISH"
    if sent < -0.1:
        return "BEARISH"
    return "NEUTRAL"


def build_report(prices: pd.DataFrame, sentiment: pd.DataFrame,
                 next_preds: pd.DataFrame, metrics: dict,
                 news: pd.DataFrame, charts: list[str]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    levels = compute_levels(prices)
    latest_sent = _latest_sentiment(sentiment)

    summary = next_preds.merge(levels[["ticker", "support", "resistance"]],
                               on="ticker", how="left")
    summary = summary.merge(latest_sent, on="ticker", how="left")
    for col in ["social_sentiment", "news_sentiment", "mention_count_total"]:
        if col not in summary.columns:
            summary[col] = None
    summary["sentiment_direction"] = summary["social_sentiment"].map(_direction_word)
    summary["position"] = [
        position_signal(r["predicted_return_pct"], r["social_sentiment"])
        for _, r in summary.iterrows()
    ]
    summary.to_csv(DATA_DIR / "ticker_summary.csv", index=False)

    chart_files = _charts(summary, prices)
    chart_imgs = "".join(
        f'<div class="chart"><img src="../charts/{html.escape(c)}" alt="{html.escape(c)}"></div>'
        for c in chart_files)

    overall = metrics.get("overall", {}) if metrics else {}
    if overall.get("rmse") is not None:
        metrics_html = (
            f"<p>Out-of-sample predictions: <b>{overall['n_oos']}</b> &nbsp;|&nbsp; "
            f"Model RMSE: <b>{overall['rmse']:,.2f}</b> &nbsp;|&nbsp; "
            f"Naive RMSE (tomorrow = today): <b>{overall['naive_rmse']:,.2f}</b> &nbsp;|&nbsp; "
            f"MAPE: <b>{overall.get('mape', 0):.2f}%</b> &nbsp;|&nbsp; "
            f"Direction accuracy: <b>{overall['direction_accuracy']:.1%}</b></p>"
            "<p class='note'>Pooled RMSE mixes HKD and KRW price scales; MAPE is the "
            "scale-free comparison. The model predicts next-day returns, converted back to prices.</p>"
            "<p class='note'>Walk-forward expanding-window validation: every prediction was made "
            "by a model trained only on data available before that day. No look-ahead.</p>")
    else:
        metrics_html = "<p class='note'>Not enough price history yet for walk-forward metrics.</p>"

    sections = []
    for exchange in EXCHANGES:
        sub = summary[summary["exchange"] == exchange].copy()
        sub = sub.sort_values(["mention_count_total", "social_sentiment"],
                              ascending=False, na_position="last")
        rows_html = []
        for _, r in sub.iterrows():
            sent = "n/a" if pd.isna(r["social_sentiment"]) else f"{r['social_sentiment']:+.2f}"
            news_s = "n/a" if pd.isna(r.get("news_sentiment")) else f"{r['news_sentiment']:+.2f}"
            mentions = 0 if pd.isna(r["mention_count_total"]) else int(r["mention_count_total"])
            rows_html.append(
                "<tr>"
                f"<td><b>{html.escape(r['ticker'])}</b><br><span class='sub'>{html.escape(NAME.get(r['ticker'], ''))}</span></td>"
                f"<td>{r['last_close']:,.2f}</td>"
                f"<td>{mentions}</td>"
                f"<td>{sent} ({html.escape(str(r['sentiment_direction']))})</td>"
                f"<td>{news_s}</td>"
                f"<td>{html.escape(str(r['position']))}</td>"
                f"<td>{r['support']:,.2f}</td>"
                f"<td>{r['resistance']:,.2f}</td>"
                f"<td>{r['predicted_close_next']:,.2f} ({r['predicted_return_pct']:+.2f}%, {html.escape(str(r['direction']))})</td>"
                "</tr>")
        sections.append(f"""
        <h2>{exchange}</h2>
        <table>
          <tr><th>Ticker</th><th>Current price</th><th>Mentions (10d)</th>
              <th>Social sentiment</th><th>News sentiment</th><th>Position (paper)</th>
              <th>Support</th><th>Resistance</th><th>Predicted n+1</th></tr>
          {''.join(rows_html)}
        </table>""")

    news_html = ""
    if news is not None and not news.empty:
        items = []
        for _, n in news.head(20).iterrows():
            items.append(f"<li><b>{html.escape(str(n['ticker']))}</b> - "
                         f"<a href='{html.escape(str(n['url']))}'>{html.escape(str(n['title']))}</a> "
                         f"<span class='sub'>{html.escape(str(n['published_date']))[:10]}</span></li>")
        news_html = "<h2>Latest news (Exa)</h2><ul>" + "".join(items) + "</ul>"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Hang Seng / KRX Sentiment Report</title>
<style>
 body {{ font-family: Georgia, 'Times New Roman', serif; margin: 32px auto; max-width: 1080px; color: #1a1a1a; background: #fdfcf9; padding: 0 16px; }}
 h1 {{ font-size: 26px; border-bottom: 3px solid #1a1a1a; padding-bottom: 8px; }}
 h2 {{ font-size: 19px; margin-top: 28px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; background: #fff; }}
 th, td {{ border: 1px solid #d8d2c4; padding: 7px 8px; text-align: left; vertical-align: top; }}
 th {{ background: #efe9dc; }}
 .sub {{ color: #666; font-size: 12px; }}
 .note {{ color: #555; font-size: 13px; }}
 .chart img {{ max-width: 100%; border: 1px solid #d8d2c4; background:#fff; }}
 .disclaimer {{ margin-top: 32px; border-top: 1px solid #ccc; padding-top: 10px; color: #555; font-size: 12.5px; }}
</style></head><body>
<h1>Hang Seng / KRX - Sentiment Price Prediction Report</h1>
<p class="note">Generated {generated}. Social window: last 10 days (Apify). News: Exa.
Prices: Yahoo Finance daily bars. Tickers sorted by social mentions, then sentiment score.</p>
<h2>Model validation (out-of-sample)</h2>
{metrics_html}
{''.join(sections)}
<h2>Charts</h2>
{chart_imgs}
{news_html}
<p class="disclaimer"><b>Disclaimer:</b> This report is a research / coursework artefact.
Position signals are rule-based paper signals derived from a statistical model and public
sentiment; they are NOT financial advice and must not be used for live trading decisions.
Sentiment is lexicon-based and can misread sarcasm, non-English posts and context.</p>
</body></html>"""
    out = REPORTS_DIR / "report.html"
    out.write_text(doc, encoding="utf-8")
    print(f"[report] wrote {out} (charts: {', '.join(chart_files)})")
    return out


def main() -> Path:
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    sent_path = DATA_DIR / "daily_sentiment.csv"
    sentiment = pd.read_csv(sent_path) if sent_path.exists() else pd.DataFrame()
    news_path = DATA_DIR / "news.csv"
    news = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()
    pred_path = DATA_DIR / "next_day_predictions.csv"
    next_preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    # Metrics are recomputed by model.main() in the full pipeline; standalone
    # report runs show whatever OOS file exists.
    metrics = {"overall": {"n_oos": 0, "rmse": None, "naive_rmse": None,
                           "direction_accuracy": None}}
    oos_path = DATA_DIR / "oos_predictions.csv"
    if oos_path.exists() and not next_preds.empty:
        from model import evaluate
        oos = pd.read_csv(oos_path)
        if not oos.empty:
            metrics = {"overall": evaluate(oos)}
    return build_report(prices, sentiment, next_preds, metrics, news, [])


if __name__ == "__main__":
    main()
