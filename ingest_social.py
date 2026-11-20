"""Social ingestion via Apify.

Runs 2 sources per exchange (see config.SOURCES_PER_EXCHANGE / README mapping
table), normalises every post to a common schema and writes data/social_posts.csv:

    source, exchange, ticker, timestamp_utc, text,
    likes, reposts, views, author_followers, url

Design notes
------------
* Actor inputs differ per actor; each runner builds the input dict for its
  actor and maps the actor's output items defensively (missing fields -> 0 /
  empty string) so an actor schema change degrades one source, not the run.
* Ticker assignment: a post is matched to a ticker when any of the ticker's
  keywords / cashtag appears in the text (case-insensitive). Posts matching
  no configured ticker are dropped.
* All timestamps are converted to UTC - the brief explicitly asks for UTC
  alignment with price bars.
* If APIFY_TOKEN is missing the module exits cleanly and run_all.py continues
  with whatever cached CSV exists.
"""
from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

from config import (APIFY_ACTORS, APIFY_TOKEN, DATA_DIR, SOCIAL_LOOKBACK_DAYS,
                    SOURCES_PER_EXCHANGE, TICKERS, utc_now)

SCHEMA = ["source", "exchange", "ticker", "timestamp_utc", "text",
          "likes", "reposts", "views", "author_followers", "url"]


def _match_ticker(text: str, exchange: str) -> str | None:
    low = text.lower()
    for t in TICKERS:
        if t.exchange != exchange:
            continue
        for kw in t.keywords:
            if kw.lower() in low:
                return t.symbol
    return None


def _search_terms(exchange: str) -> list[str]:
    terms: list[str] = []
    for t in TICKERS:
        if t.exchange == exchange:
            terms.extend(t.keywords[:3])
    return terms


# ---------------------------------------------------------------------------
# Actor runners - each returns a list of raw dicts already in SCHEMA shape
# (ticker still to be matched by caller where the actor returns free text).
# ---------------------------------------------------------------------------

def _dataset_id(run) -> str:
    """apify-client v2 returns a Run object; older versions returned a dict."""
    if isinstance(run, dict):
        return run["defaultDatasetId"]
    return getattr(run, "default_dataset_id") or run.model_dump()["defaultDatasetId"]


def _run_twitter(client, exchange: str) -> list[dict]:
    """X/Twitter via apidojo/tweet-scraper (search terms = cashtags/keywords)."""
    terms = _search_terms(exchange)
    since = (utc_now() - timedelta(days=SOCIAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    queries = [f"{term} since:{since}" for term in terms]
    run_input = {
        "searchTerms": queries,
        "maxTweets": 500,
        "sort": "Latest",
        "tweetLanguage": "en",
    }
    run = client.actor(APIFY_ACTORS["twitter"]).call(run_input=run_input)
    items = client.dataset(_dataset_id(run)).iterate_items()
    rows = []
    for it in items:
        text = it.get("text") or it.get("full_text") or ""
        author = it.get("author") or {}
        rows.append({
            "source": "twitter",
            "text": text,
            "timestamp_utc": it.get("createdAt") or it.get("created_at"),
            "likes": it.get("likeCount") or it.get("favorite_count") or 0,
            "reposts": it.get("retweetCount") or it.get("retweet_count") or 0,
            "views": it.get("viewCount") or it.get("views") or 0,
            "author_followers": author.get("followers") or author.get("followers_count") or 0,
            "url": it.get("url") or it.get("twitterUrl") or "",
        })
    return rows


def _run_reddit(client, exchange: str) -> list[dict]:
    """Reddit via harshmaur/reddit-scraper (keyword search, no login needed).

    Input field names verified against the actor's live input schema
    (Apify store API, 2026-11-20).
    """
    terms = _search_terms(exchange)
    since = (utc_now() - timedelta(days=SOCIAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    run_input = {
        "searchTerms": terms,
        "searchPosts": True,
        "searchComments": False,
        "searchCommunities": False,
        "searchSort": "new",
        "searchTime": "month",
        "maxPostsCount": 30,
        "postedAfter": since,
    }
    run = client.actor(APIFY_ACTORS["reddit"]).call(run_input=run_input)
    items = client.dataset(_dataset_id(run)).iterate_items()
    rows = []
    for it in items:
        text = " ".join(filter(None, [it.get("title", ""),
                                      it.get("body", "") or it.get("selftext", "") or it.get("text", "")]))
        rows.append({
            "source": "reddit",
            "text": text,
            "timestamp_utc": it.get("createdAt") or it.get("created_utc") or it.get("date"),
            "likes": it.get("upVotes") or it.get("score") or it.get("upvotes") or 0,
            "reposts": 0,
            "views": 0,
            "author_followers": 0,  # Reddit actors do not expose follower counts
            "url": it.get("url") or it.get("link") or it.get("postUrl") or it.get("permalink") or "",
        })
    return rows


RUNNERS = {"twitter": _run_twitter, "reddit": _run_reddit}


def fetch_social() -> pd.DataFrame:
    if not APIFY_TOKEN:
        print("[social] APIFY_TOKEN not set - skipping social ingestion. "
              "Cached data/social_posts.csv (if any) will be used downstream.")
        return pd.DataFrame(columns=SCHEMA)

    from apify_client import ApifyClient  # imported lazily: optional at import time
    client = ApifyClient(APIFY_TOKEN)

    all_rows: list[dict] = []
    for exchange, sources in SOURCES_PER_EXCHANGE.items():
        for source in sources:
            runner = RUNNERS.get(source)
            if runner is None:
                print(f"[social] no runner for source '{source}', skipping")
                continue
            print(f"[social] running {source} actor ({APIFY_ACTORS[source]}) for {exchange} ...")
            try:
                raw = runner(client, exchange)
            except Exception as exc:  # one source failing must not kill the run
                print(f"[social] WARNING: {source}/{exchange} failed: {exc}")
                continue
            kept = 0
            for r in raw:
                ticker = _match_ticker(r["text"], exchange)
                if ticker is None:
                    continue
                ts = pd.to_datetime(r["timestamp_utc"], utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                # Enforce the 10-day window client-side as well.
                if ts < utc_now() - timedelta(days=SOCIAL_LOOKBACK_DAYS + 1):
                    continue
                all_rows.append({
                    "source": source, "exchange": exchange, "ticker": ticker,
                    "timestamp_utc": ts.isoformat(),
                    "text": re.sub(r"\s+", " ", r["text"]).strip()[:2000],
                    "likes": int(r["likes"] or 0), "reposts": int(r["reposts"] or 0),
                    "views": int(r["views"] or 0),
                    "author_followers": int(r["author_followers"] or 0),
                    "url": r["url"],
                })
                kept += 1
            print(f"[social]   {source}/{exchange}: {len(raw)} raw, {kept} matched to tickers")

    df = pd.DataFrame(all_rows, columns=SCHEMA)
    if not df.empty:
        df = df.drop_duplicates(subset=["source", "timestamp_utc", "text"])
    return df


def main() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_social()
    path = DATA_DIR / "social_posts.csv"
    if df.empty and path.exists():
        print(f"[social] no new posts fetched; keeping existing {path}")
        return pd.read_csv(path)
    df.to_csv(path, index=False)
    print(f"[social] saved {len(df)} posts -> {path}")
    return df


if __name__ == "__main__":
    main()
