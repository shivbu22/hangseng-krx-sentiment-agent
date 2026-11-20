"""End-to-end orchestrator.

    python run_all.py

Stages: prices -> social (Apify) -> news (Exa) -> sentiment -> model -> report.

Graceful degradation: if APIFY_TOKEN / EXA_API_KEY are missing, those stages
are skipped with a clear message and the pipeline continues on cached CSVs
in data/ (or with price-only features). This lets the whole pipeline be
verified before any keys exist.
"""
from __future__ import annotations

import sys

import pandas as pd

import ingest_news
import ingest_social
import model as model_mod
import prices as prices_mod
import report as report_mod
import sentiment as sentiment_mod
from config import APIFY_TOKEN, DATA_DIR, EXA_API_KEY


def _load(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main() -> int:
    print("=" * 72)
    print("Hang Seng / KRX Sentiment Price Prediction Agent - full run")
    print("=" * 72)
    skipped = []

    # 1. Prices (no key needed - always runs)
    prices = prices_mod.main()

    # 2. Social (Apify)
    if APIFY_TOKEN:
        ingest_social.main()
    else:
        skipped.append("social ingestion (APIFY_TOKEN missing)")
        print("[run_all] SKIPPED: social ingestion - APIFY_TOKEN not set in .env")

    # 3. News (Exa)
    if EXA_API_KEY:
        ingest_news.main()
    else:
        skipped.append("news ingestion (EXA_API_KEY missing)")
        print("[run_all] SKIPPED: news ingestion - EXA_API_KEY not set in .env")

    # 4. Sentiment aggregation (works on cached CSVs; empty inputs are fine)
    sentiment_mod.main()

    # 5. Model: walk-forward validation + n+1 predictions
    next_preds, metrics, _oos = model_mod.main()

    # 6. Visual report
    sentiment = _load("daily_sentiment.csv")
    news = _load("news.csv")
    report_mod.build_report(prices, sentiment, next_preds, metrics, news, [])

    print("=" * 72)
    if skipped:
        print("[run_all] Completed WITH SKIPPED STAGES:")
        for s in skipped:
            print(f"  - {s}")
        print("[run_all] Add the keys to .env and re-run for the full pipeline.")
    else:
        print("[run_all] Completed - all stages ran.")
    print("[run_all] Report: reports/report.html")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
