"""Price ingestion via yfinance (no API key required).

Downloads daily OHLCV bars for every configured ticker and writes a single
long-format CSV: date, ticker, exchange, open, high, low, close, volume.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from config import DATA_DIR, PRICE_LOOKBACK_DAYS, TICKERS, utc_now
from datetime import timedelta


def fetch_prices(lookback_days: int = PRICE_LOOKBACK_DAYS) -> pd.DataFrame:
    end = utc_now().date()
    start = end - timedelta(days=lookback_days)
    frames: list[pd.DataFrame] = []
    for t in TICKERS:
        print(f"[prices] downloading {t.symbol} ({t.name}) ...")
        hist = yf.Ticker(t.symbol).history(start=start.isoformat(),
                                           end=end.isoformat(),
                                           auto_adjust=False)
        if hist.empty:
            print(f"[prices] WARNING: no data returned for {t.symbol}")
            continue
        hist = hist.reset_index()
        hist.columns = [c.lower() for c in hist.columns]
        out = pd.DataFrame({
            "date": pd.to_datetime(hist["date"]).dt.tz_localize(None).dt.normalize(),
            "ticker": t.symbol,
            "exchange": t.exchange,
            "open": hist["open"],
            "high": hist["high"],
            "low": hist["low"],
            "close": hist["close"],
            "volume": hist.get("volume", 0),
        })
        frames.append(out)
        print(f"[prices]   {t.symbol}: {len(out)} bars, "
              f"{out['date'].min().date()} -> {out['date'].max().date()}, "
              f"last close {out['close'].iloc[-1]:,.2f}")
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "exchange", "open",
                                     "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    return df


def main() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_prices()
    path = DATA_DIR / "prices.csv"
    df.to_csv(path, index=False)
    print(f"[prices] saved {len(df)} rows -> {path}")
    return df


if __name__ == "__main__":
    main()
