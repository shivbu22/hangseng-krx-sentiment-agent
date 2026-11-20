"""Central configuration for the Hang Seng / KRX Sentiment Price Prediction Agent.

All tickers, sources, windows and paths live here so the rest of the pipeline
stays declarative. Secrets are NEVER stored here - they come from .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = BASE_DIR / "charts"

load_dotenv(BASE_DIR / ".env")

APIFY_TOKEN: str | None = os.getenv("APIFY_TOKEN") or None
EXA_API_KEY: str | None = os.getenv("EXA_API_KEY") or None


@dataclass(frozen=True)
class Ticker:
    symbol: str          # Yahoo Finance symbol
    name: str
    exchange: str        # "HKEX" or "KRX"
    cashtag: str         # e.g. "$0700.HK" style search term base
    keywords: list[str] = field(default_factory=list)


TICKERS: list[Ticker] = [
    # ---- Hang Seng / HKEX (.HK) ----
    Ticker("0700.HK", "Tencent Holdings", "HKEX", "0700.HK",
           ["Tencent", "0700.HK", "$0700", "Tencent stock"]),
    Ticker("9988.HK", "Alibaba Group", "HKEX", "9988.HK",
           ["Alibaba", "9988.HK", "$9988", "BABA HK", "Alibaba stock"]),
    Ticker("1810.HK", "Xiaomi Corporation", "HKEX", "1810.HK",
           ["Xiaomi", "1810.HK", "$1810", "Xiaomi stock"]),
    Ticker("3690.HK", "Meituan", "HKEX", "3690.HK",
           ["Meituan", "3690.HK", "$3690", "Meituan stock"]),
    Ticker("2318.HK", "Ping An Insurance", "HKEX", "2318.HK",
           ["Ping An", "2318.HK", "$2318", "Ping An Insurance stock"]),
    # ---- KRX / Korea (.KS) ----
    Ticker("005930.KS", "Samsung Electronics", "KRX", "005930.KS",
           ["Samsung Electronics", "005930.KS", "$005930", "Samsung stock"]),
    Ticker("000660.KS", "SK Hynix", "KRX", "000660.KS",
           ["SK Hynix", "000660.KS", "$000660", "SK Hynix stock"]),
    Ticker("035420.KS", "Naver Corporation", "KRX", "035420.KS",
           ["Naver", "035420.KS", "$035420", "Naver stock"]),
    Ticker("035720.KS", "Kakao Corporation", "KRX", "035720.KS",
           ["Kakao", "035720.KS", "$035720", "Kakao stock"]),
    Ticker("005380.KS", "Hyundai Motor", "KRX", "005380.KS",
           ["Hyundai Motor", "005380.KS", "$005380", "Hyundai Motor stock"]),
]

EXCHANGES = ["HKEX", "KRX"]

# ---------------------------------------------------------------------------
# Apify actors
# ---------------------------------------------------------------------------
# The brief asks for 2 different social sources per exchange, chosen from the
# employer's list. Several listed sources (LIHKG, Naver Finance boards, Blind,
# KakaoTalk) have NO public Apify actor - they require login / are closed
# communities. We therefore use the closest PUBLIC alternatives and document
# the mapping honestly in README.md:
#
#   Source A (both exchanges): X / Twitter search - cashtag + keyword search.
#       Actor: apidojo/tweet-scraper  (widely used public Twitter scraper)
#       Fallback actor: quacker/twitter-scraper
#   Source B (both exchanges): Reddit search - r/stocks, r/investing and
#       HK/KR-relevant subreddits via keyword search.
#       Actor: harshmaur/reddit-scraper (verified in the Apify store +
#        input schema checked live 2026-11-20; keyword search, no login)
#       (trudax/reddit-scraper-lite was tried first and rejected: verified
#        live 2026-11-20 that it refuses all searches without a logged-in
#        Reddit sessionToken, so it cannot run on a token alone.
#        apidojo/reddit-scraper does not exist in the store.)
#   Source B alt: Telegram public channels via telegram-scraper where a
#       public channel exists. Actor: lexis-solutions/telegram-scraper
#
# Actor IDs are configurable via .env so the reviewer can swap them without
# touching code.
APIFY_ACTORS = {
    "twitter": os.getenv("APIFY_ACTOR_TWITTER", "apidojo/tweet-scraper"),
    "reddit": os.getenv("APIFY_ACTOR_REDDIT", "harshmaur/reddit-scraper"),
    "telegram": os.getenv("APIFY_ACTOR_TELEGRAM", "lexis-solutions/telegram-scraper"),
}

# Which sources to run per exchange (2 per exchange, per the brief).
SOURCES_PER_EXCHANGE = {
    "HKEX": ["twitter", "reddit"],
    "KRX": ["twitter", "reddit"],
}

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


SOCIAL_LOOKBACK_DAYS = 10          # brief: last 10 days of social posts
PRICE_LOOKBACK_DAYS = 120          # ~90 trading days of daily bars for the model
NEWS_RESULTS_PER_TICKER = 5
NEWS_LOOKBACK_DAYS = 10

# Model
MIN_TRAIN_DAYS = 30                # walk-forward: minimum training window
RANDOM_STATE = 42
