"""Finance sentiment scoring - dependency-light, no model download required.

Approach: a curated finance lexicon (bullish / bearish terms incl. cashtag-era
slang) with simple negation handling ("not good" flips polarity within a
3-token window) and intensifier weighting. Each post's score is in [-1, 1].

Aggregation is engagement-weighted: a post's influence weight is
    w = 1 + log1p(likes + 2*reposts) + 0.25 * log1p(author_followers)
so high-engagement / high-reach posts count more, with diminishing returns.
News headlines+snippets are scored with the identical function so social and
news sentiment are directly comparable.

Outputs (data/daily_sentiment.csv): one row per (date, ticker) with
social_sentiment, mention_count, engagement_sum, news_sentiment, news_count.
Days with no posts get NaN sentiment (the model forward-fills / zeroes them
explicitly - no silent fabrication).
"""
from __future__ import annotations

import math
import re

import pandas as pd

from config import DATA_DIR

POSITIVE = {
    "bullish", "buy", "buying", "long", "moon", "rally", "surge", "surged",
    "soar", "soared", "jump", "jumped", "gain", "gains", "gained", "profit",
    "profits", "growth", "grew", "beat", "beats", "upgrade", "upgraded",
    "outperform", "strong", "record", "high", "breakout", "upside",
    "momentum", "accumulate", "overweight", "optimistic", "recovery",
    "rebound", "boom", "winner", "earnings beat", "raised guidance",
    "dividend", "buyback", "expansion", "partnership", "approval",
    "approved", "launch", "launched", "milestone", "exceed", "exceeded",
}
NEGATIVE = {
    "bearish", "sell", "selling", "short", "crash", "crashed", "plunge",
    "plunged", "drop", "dropped", "fall", "fell", "loss", "losses", "miss",
    "missed", "downgrade", "downgraded", "underperform", "weak", "slump",
    "downside", "risk", "risks", "warning", "warned", "fraud", "lawsuit",
    "probe", "investigation", "ban", "banned", "delist", "bankruptcy",
    "debt", "layoff", "layoffs", "cut", "cuts", "slowdown", "decline",
    "declined", "disappointing", "concern", "concerns", "volatile",
    "selloff", "sell-off", "overvalued", "bubble", "sanction", "tariff",
}
INTENSIFIERS = {"very": 1.5, "strongly": 1.5, "huge": 1.6, "massive": 1.6,
                "sharply": 1.5, "significantly": 1.4, "slightly": 0.6}
NEGATIONS = {"not", "no", "never", "isn't", "isnt", "don't", "dont",
             "won't", "wont", "cannot", "can't", "cant"}

_TOKEN_RE = re.compile(r"[a-z']+")


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1] for one text."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    low = text.lower()
    # Multi-word phrases first (counted once, then removed).
    score = 0.0
    hits = 0
    for phrase in ("earnings beat", "raised guidance", "sell-off"):
        if phrase in low:
            score += 1.0 if phrase != "sell-off" else -1.0
            hits += 1
            low = low.replace(phrase, " ")
    tokens = _TOKEN_RE.findall(low)
    for i, tok in enumerate(tokens):
        polarity = 0.0
        if tok in POSITIVE:
            polarity = 1.0
        elif tok in NEGATIVE:
            polarity = -1.0
        if polarity == 0.0:
            continue
        weight = 1.0
        window = tokens[max(0, i - 3):i]
        if any(w in NEGATIONS for w in window):
            polarity *= -1.0
        for w in window:
            if w in INTENSIFIERS:
                weight *= INTENSIFIERS[w]
        score += polarity * weight
        hits += 1
    if hits == 0:
        return 0.0
    # Squash to [-1, 1]; average hit strength, not raw sum, so long posts
    # don't dominate purely by length.
    return max(-1.0, min(1.0, score / math.sqrt(hits + 1)))


def _engagement_weight(row) -> float:
    likes = float(row.get("likes", 0) or 0)
    reposts = float(row.get("reposts", 0) or 0)
    followers = float(row.get("author_followers", 0) or 0)
    return 1.0 + math.log1p(likes + 2.0 * reposts) + 0.25 * math.log1p(followers)


def daily_sentiment(social: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-level and news-level sentiment to (date, ticker) rows."""
    rows: list[dict] = []

    social_by: dict[tuple, dict] = {}
    if social is not None and not social.empty:
        s = social.copy()
        s["date"] = pd.to_datetime(s["timestamp_utc"], utc=True).dt.tz_localize(None).dt.normalize()
        s["score"] = s["text"].map(score_text)
        s["weight"] = s.apply(_engagement_weight, axis=1)
        for (date, ticker), g in s.groupby(["date", "ticker"]):
            wsum = g["weight"].sum()
            social_by[(date, ticker)] = {
                "social_sentiment": float((g["score"] * g["weight"]).sum() / wsum) if wsum else 0.0,
                "mention_count": int(len(g)),
                "engagement_sum": float(g["likes"].sum() + g["reposts"].sum()),
            }

    news_by: dict[tuple, dict] = {}
    if news is not None and not news.empty:
        n = news.copy()
        n["date"] = pd.to_datetime(n["published_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        n = n.dropna(subset=["date"])
        n["score"] = (n["title"].fillna("") + " " + n["snippet"].fillna("")).map(score_text)
        for (date, ticker), g in n.groupby(["date", "ticker"]):
            news_by[(date, ticker)] = {
                "news_sentiment": float(g["score"].mean()),
                "news_count": int(len(g)),
            }

    keys = set(social_by) | set(news_by)
    for date, ticker in sorted(keys):
        row = {"date": date, "ticker": ticker}
        row.update({"social_sentiment": None, "mention_count": 0, "engagement_sum": 0.0})
        row.update(social_by.get((date, ticker), {}))
        row.update({"news_sentiment": None, "news_count": 0})
        row.update(news_by.get((date, ticker), {}))
        rows.append(row)

    cols = ["date", "ticker", "social_sentiment", "mention_count",
            "engagement_sum", "news_sentiment", "news_count"]
    return pd.DataFrame(rows, columns=cols)


def main() -> pd.DataFrame:
    social_path, news_path = DATA_DIR / "social_posts.csv", DATA_DIR / "news.csv"
    social = pd.read_csv(social_path) if social_path.exists() else pd.DataFrame()
    news = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()
    out = daily_sentiment(social, news)
    path = DATA_DIR / "daily_sentiment.csv"
    out.to_csv(path, index=False)
    print(f"[sentiment] scored social={len(social)} posts, news={len(news)} articles "
          f"-> {len(out)} daily rows saved to {path}")
    return out


if __name__ == "__main__":
    main()
