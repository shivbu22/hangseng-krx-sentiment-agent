# Hang Seng Sentiment Price Prediction Agent

**TZURONI LTD / CrowdWisdomTrading - Intern Assessment**
Author: Shivam Yadav · Deadline: 23 Nov 2026

A Python pipeline that collects social-media sentiment and news for 5 Hang
Seng (HKEX) and 5 KRX (Korea) stocks, combines them with daily price data,
trains a walk-forward-validated model to predict the next day's (n+1) close,
and produces a visual HTML report sorted by social mentions and sentiment.

> **Disclaimer:** Everything here is a research / coursework artefact. The
> position signals are simple rule-based *paper* signals. They are **not
> financial advice** and must not be used for live trading.

---

## Tickers

| Exchange | Ticker | Asset |
|---|---|---|
| HKEX | 0700.HK | Tencent Holdings |
| HKEX | 9988.HK | Alibaba Group |
| HKEX | 1810.HK | Xiaomi Corporation |
| HKEX | 3690.HK | Meituan |
| HKEX | 2318.HK | Ping An Insurance |
| KRX | 005930.KS | Samsung Electronics |
| KRX | 000660.KS | SK Hynix |
| KRX | 035420.KS | Naver Corporation |
| KRX | 035720.KS | Kakao Corporation |
| KRX | 005380.KS | Hyundai Motor |

Prices come from Yahoo Finance via `yfinance` (no key needed).

## Architecture

```
config.py          tickers, sources, windows, actor IDs, .env loading
prices.py          yfinance daily bars            -> data/prices.csv
ingest_social.py   Apify actors (2 sources/exchange) -> data/social_posts.csv
ingest_news.py     Exa news search per ticker     -> data/news.csv
sentiment.py       finance-lexicon scoring + engagement-weighted
                   daily aggregation              -> data/daily_sentiment.csv
levels.py          support / resistance (pivot + 20-day extremes)
model.py           features + walk-forward validation + n+1 prediction
                   -> data/oos_predictions.csv, data/next_day_predictions.csv
report.py          visual HTML report + charts    -> reports/report.html
run_all.py         orchestrates all stages (degrades gracefully without keys)
```

## Social sources and Apify actor mapping (honest note)

The brief lists per-exchange sources (HK: LIHKG, Telegram, Futu/Moomoo,
AAStocks/ETNet; KR: Naver Finance boards, Daum, KakaoTalk/Telegram, Blind).
Several of those have **no public Apify actor**: LIHKG, Naver Finance boards,
Blind and KakaoTalk are login-gated / closed communities, and Futu/Moomoo
community posts sit behind account walls. Scraping them would require
account credentials and would breach those platforms' terms, so this project
uses the closest **public** alternatives - 2 sources per exchange - and says
so plainly:

| Slot | Source used | Apify actor ID | Maps to (brief list) |
|---|---|---|---|
| A (HKEX + KRX) | X / Twitter cashtag + keyword search | `apidojo/tweet-scraper` | Public equivalent of Telegram / KakaoTalk open-channel chatter - same real-time retail-sentiment role |
| B (HKEX + KRX) | Reddit keyword search | `harshmaur/reddit-scraper` | Public equivalent of LIHKG / Naver board discussion - forum-style retail discussion |

Notes on slot B, verified live on 2026-11-20: `trudax/reddit-scraper-lite`
was tried first and rejected - it refuses every search without a logged-in
Reddit `sessionToken`, so it cannot run on an Apify token alone.
`apidojo/reddit-scraper` does not exist in the Apify store. The chosen
actor's input schema was checked against the live store API before use.
| B-alt (optional) | Telegram public channels | `lexis-solutions/telegram-scraper` | Direct match to the brief's Telegram source, where a public channel exists |

Actor IDs are overridable in `.env` (`APIFY_ACTOR_TWITTER`, etc.) so a
reviewer with access to a private actor for a listed source can swap it in
without code changes. Search uses ticker, cashtag and keyword terms; every
post is timestamp-normalised to **UTC** and aligned to daily price bars.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in APIFY_TOKEN and EXA_API_KEY
```

Keys:
- `APIFY_TOKEN` - https://console.apify.com/account/integrations
- `EXA_API_KEY` - https://dashboard.exa.ai/api-keys

## Run

```bash
python run_all.py
```

Outputs: `reports/report.html` (open in a browser), CSVs in `data/`, charts
in `charts/`. Without keys the run still completes: price, model and report
stages run on real price data and any cached CSVs, and the skipped stages are
printed clearly.

Tests (offline, uses a clearly-labelled synthetic fixture - never real data):

```bash
pip install pytest
python -m pytest tests/ -q
```

## Data schema

`social_posts.csv`: `source, exchange, ticker, timestamp_utc, text, likes,
reposts, views, author_followers, url`

`news.csv`: `ticker, exchange, title, url, published_date, snippet, source`

`prices.csv`: `date, ticker, exchange, open, high, low, close, volume`

`daily_sentiment.csv`: `date, ticker, social_sentiment (-1..1),
mention_count, engagement_sum, news_sentiment, news_count`

## Method

- **Sentiment:** curated finance lexicon with negation handling and
  intensifiers (`sentiment.py`). Posts are engagement-weighted:
  `w = 1 + log1p(likes + 2*reposts) + 0.25*log1p(author_followers)`.
  News headlines/snippets are scored with the identical function.
- **Features (known at close of day t, no look-ahead):** 1-day and 5-day
  returns, 10-day volatility, volume vs 10-day mean, daily social sentiment,
  log mention volume, news sentiment, and a has-social-data flag.
- **Model:** scikit-learn `GradientBoostingRegressor`, per ticker. The model
  predicts the next-day **return** (converted back to a price), not the price
  level - tree models cannot extrapolate price levels and otherwise collapse
  to the historical mean price.
- **Validation:** expanding-window **walk-forward** - train on all data up
  to day t, predict t+1, expand, repeat. Metrics are computed only on
  out-of-sample predictions: **RMSE vs a naive baseline** (tomorrow = today)
  and **direction accuracy**.
- **Support / resistance:** classic pivot points on the last bar, blended
  with 20-day high/low.
- **Position signal (paper only):** LONG if predicted return > +0.5% and
  sentiment >= -0.1; SHORT if < -0.5% and sentiment <= +0.1; else NONE.

## Honest limitations

- Lexicon sentiment misreads sarcasm, memes and non-English posts (LIHKG /
  Naver content is largely Cantonese / Korean - a multilingual transformer
  such as FinBERT-multilingual would be the upgrade path).
- Social coverage for HK/KR small-discussion sources is thinner on public
  platforms than on the login-gated originals; mention counts reflect the
  public sources actually scraped.
- Apify plan limits apply: on a Free Plan account the tweet-scraper actor
  returns at most 10 items per run (verified live 2026-11-20), so mention
  volume is a sample, not a census. The code requests up to 500 tweets per
  exchange; a paid Apify plan lifts the cap with no code change.
- ~90 trading days is a small training set; walk-forward metrics have wide
  confidence intervals and the model should be re-validated as data grows.
- Next-day close prediction is intrinsically noisy; the naive baseline is
  genuinely hard to beat, and the report shows both numbers so the reader
  can judge.

## Submission

Per the brief, submission is this GitHub repo link plus the Apify/Exa tokens
and actor IDs by email. All secrets are read from `.env` (git-ignored); only
`.env.example` is committed.
