"""Unit tests - run with: python -m pytest tests/ -q

The social fixture in this folder is SYNTHETIC sample data, clearly labelled,
used only to exercise the sentiment/aggregation code path offline. It is never
presented as real scraped data anywhere in the report.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiment import daily_sentiment, score_text  # noqa: E402
from model import position_signal  # noqa: E402
from levels import compute_levels  # noqa: E402

FIXTURE = Path(__file__).parent / "sample_social_posts.csv"


def test_score_text_polarity():
    assert score_text("bullish breakout, strong earnings beat") > 0.2
    assert score_text("crash, massive losses and downgrade") < -0.2
    assert score_text("the company held a meeting on tuesday") == 0.0


def test_negation_flips():
    # "not" directly negates the lexicon word inside the 3-token window.
    assert score_text("this is not bullish") < 0
    assert score_text("this is bullish") > 0
    assert score_text("strong growth and record profit") > 0


def test_daily_sentiment_on_synthetic_fixture():
    social = pd.read_csv(FIXTURE)
    out = daily_sentiment(social, pd.DataFrame())
    assert not out.empty
    tencent = out[out["ticker"] == "0700.HK"]
    assert int(tencent["mention_count"].sum()) == 2
    # One strongly positive + one negated post -> aggregate stays in [-1, 1]
    assert tencent["social_sentiment"].between(-1, 1).all()


def test_position_signal_rules():
    assert position_signal(1.2, 0.3) == "LONG (paper)"
    assert position_signal(-1.2, -0.3) == "SHORT (paper)"
    assert position_signal(1.2, -0.8) == "NONE"   # sentiment disagrees
    assert position_signal(0.2, 0.9) == "NONE"    # move too small


def test_levels_ordering():
    prices = pd.DataFrame({
        "date": pd.date_range("2026-10-01", periods=25),
        "ticker": ["0700.HK"] * 25, "exchange": ["HKEX"] * 25,
        "open": [100.0] * 25, "high": [105.0] * 25, "low": [95.0] * 25,
        "close": [100.0] * 25, "volume": [1000] * 25,
    })
    lv = compute_levels(prices)
    assert lv.iloc[0]["support"] < lv.iloc[0]["last_close"] < lv.iloc[0]["resistance"]
