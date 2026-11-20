"""Support / resistance levels from recent daily bars.

Method: classic pivot points computed on the last completed bar, blended with
rolling 20-day extremes. Reported levels:
  resistance = max(pivot R1, 20-day high)
  support    = min(pivot S1, 20-day low)
This is a transparent, reproducible rule - not a prediction.
"""
from __future__ import annotations

import pandas as pd


def compute_levels(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        pivot = (last["high"] + last["low"] + last["close"]) / 3.0
        r1 = 2 * pivot - last["low"]
        s1 = 2 * pivot - last["high"]
        window = g.tail(20)
        rows.append({
            "ticker": ticker,
            "exchange": g["exchange"].iloc[-1],
            "last_close": float(last["close"]),
            "last_date": last["date"],
            "pivot": float(pivot),
            "resistance": float(max(r1, window["high"].max())),
            "support": float(min(s1, window["low"].min())),
            "high_20d": float(window["high"].max()),
            "low_20d": float(window["low"].min()),
        })
    return pd.DataFrame(rows)
