"""News ingestion via Exa (exa.com).

For each ticker, fetch the most recent + relevant news and write
data/news.csv: ticker, exchange, title, url, published_date, snippet, source.
News text is scored by the same sentiment module as social posts.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from config import (DATA_DIR, EXA_API_KEY, NEWS_LOOKBACK_DAYS,
                    NEWS_RESULTS_PER_TICKER, TICKERS, utc_now)

SCHEMA = ["ticker", "exchange", "title", "url", "published_date", "snippet", "source"]


def fetch_news() -> pd.DataFrame:
    if not EXA_API_KEY:
        print("[news] EXA_API_KEY not set - skipping news ingestion. "
              "Cached data/news.csv (if any) will be used downstream.")
        return pd.DataFrame(columns=SCHEMA)

    from exa_py import Exa
    exa = Exa(api_key=EXA_API_KEY)
    start = (utc_now() - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat()

    rows: list[dict] = []
    for t in TICKERS:
        query = f"{t.name} ({t.symbol}) stock news"
        print(f"[news] Exa search: {query}")
        try:
            res = exa.search_and_contents(
                query,
                type="news",
                num_results=NEWS_RESULTS_PER_TICKER,
                start_published_date=start,
                text={"max_characters": 600},
            )
        except Exception as exc:
            print(f"[news] WARNING: Exa failed for {t.symbol}: {exc}")
            continue
        for r in res.results:
            rows.append({
                "ticker": t.symbol, "exchange": t.exchange,
                "title": r.title or "",
                "url": r.url or "",
                "published_date": r.published_date or "",
                "snippet": (r.text or "")[:600],
                "source": getattr(r, "author", "") or "",
            })
        print(f"[news]   {t.symbol}: {len(res.results)} articles")
    return pd.DataFrame(rows, columns=SCHEMA)


def main() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_news()
    path = DATA_DIR / "news.csv"
    if df.empty and path.exists():
        print(f"[news] no new articles fetched; keeping existing {path}")
        return pd.read_csv(path)
    df.to_csv(path, index=False)
    print(f"[news] saved {len(df)} articles -> {path}")
    return df


if __name__ == "__main__":
    main()
