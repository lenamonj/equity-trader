import numpy as np
import pandas as pd
import pytest

from equity_trader.data import pattern_tools as pt
from tests.data.test_technical_tools import frame


def use(monkeypatch, df):
    monkeypatch.setattr(pt.yft, "get_price_history", lambda ticker, period="max": df)


def test_seasonality_finds_a_month_that_always_rises(monkeypatch):
    index = pd.bdate_range("2015-01-01", "2024-12-31")
    daily = np.where(index.month == 1, 1.005, 1.0)
    df = frame(100 * np.cumprod(daily))
    df.index = index
    use(monkeypatch, df)

    out = pt.seasonality("ACME")

    january = out["monthly_return_by_month"]["Jan"]
    assert january["n"] == 9  # the first January has no prior month-end to measure from
    assert january["mean"] > 0.05 and january["hit_rate"] == 1.0
    assert out["monthly_return_by_month"]["Feb"]["mean"] == 0.0
    assert out["history_start"] == "2015-01-01" and out["current_month"] == "Dec"
    assert set(out["daily_return_bps_by_weekday"]) == {"Mon", "Tue", "Wed", "Thu", "Fri"}


def test_earnings_study_puts_the_reaction_on_the_right_session(monkeypatch):
    close = np.full(40, 100.0)
    close[10:] = 105.0   # report before the open on session 10: that session reacts
    close[21:] = 115.5   # report after the close on session 20: session 21 reacts
    df = frame(close)
    use(monkeypatch, df)
    reports = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.0], "Reported EPS": [1.2, 0.9], "Surprise(%)": [20.0, -10.0]},
        index=pd.DatetimeIndex([df.index[20] + pd.Timedelta(hours=16, minutes=30),
                                df.index[10] + pd.Timedelta(hours=8)]).tz_localize(
            "America/New_York"))
    monkeypatch.setattr(pt.yft, "get_earnings_dates", lambda t: reports)

    out = pt.earnings_event_study("ACME")

    assert [e["reaction"] for e in out["events"]] == [0.1, 0.05]
    assert out["events"][0]["post_5d"] == 0.0 and out["events"][0]["post_60d"] is None
    assert out["reaction_day"]["avg_absolute_move"] == pytest.approx(0.075)
    assert out["after_beats"]["reaction"]["n"] == 1
    assert out["after_misses"]["reaction"]["mean"] == 0.05


def test_earnings_study_with_no_reports_says_so(monkeypatch):
    use(monkeypatch, frame(np.full(40, 100.0)))
    empty = pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
    monkeypatch.setattr(pt.yft, "get_earnings_dates", lambda t: empty)
    out = pt.earnings_event_study("ACME")
    assert out["reports"] == 0 and out["events"] == []


def test_episodes_count_separate_stretches():
    mask = pd.Series([False, True, True, False, True])
    assert pt._episodes(mask) == 2


def test_forward_study_describes_today_and_what_followed(monkeypatch):
    use(monkeypatch, frame(100 * 1.001 ** np.arange(600)))
    out = pt.forward_return_study("ACME")
    conditions = [s["condition"] for s in out["days_like_today"]]
    assert conditions == ["price above its 200-day average", "within 10% of the 52-week high",
                          "14-day RSI above 70"]
    assert out["baseline_all_days"]["next_63d"]["mean"] == pytest.approx(1.001 ** 63 - 1, abs=1e-4)
    assert out["baseline_all_days"]["next_63d"]["hit_rate"] == 1.0
    above = out["days_like_today"][0]
    assert above["episodes"] == 1 and above["next_126d"]["hit_rate"] == 1.0


def test_forward_study_buckets_a_deep_drawdown(monkeypatch):
    use(monkeypatch, frame(np.concatenate([np.linspace(100, 200, 300),
                                           np.linspace(200, 120, 60)])))
    conditions = [s["condition"] for s in pt.forward_return_study("ACME")["days_like_today"]]
    assert "more than 35% below the 52-week high" in conditions
    assert "price below its 200-day average" in conditions


def test_gaps_count_fills_within_five_sessions(monkeypatch):
    close = np.full(160, 100.0)
    close[100:] = 96.0                  # gaps down 4% and never trades back to 100
    open_, low = close.copy(), close.copy()
    open_[50], low[50] = 103.0, 99.5    # gaps up 3% and fills the same day
    use(monkeypatch, frame(close, open_=open_, low=low, high=np.maximum(close, open_)))

    gaps = pt.volatility_and_gaps("ACME")["gaps_of_2pct_or_more_5y"]

    assert gaps["up"] == {"count": 1, "filled_within_5_sessions": 1.0,
                          "avg_open_to_close": pytest.approx(100 / 103 - 1, abs=1e-4)}
    assert gaps["down"]["count"] == 1 and gaps["down"]["filled_within_5_sessions"] == 0.0


def test_volatility_regime_is_named_and_measured(monkeypatch):
    rng = np.random.default_rng(1)
    calm, wild = rng.normal(0, 0.005, 400), rng.normal(0, 0.04, 100)
    use(monkeypatch, frame(100 * np.cumprod(1 + np.concatenate([calm, wild]))))
    out = pt.volatility_and_gaps("ACME")
    assert out["volatility_regime_now"] == "high"
    assert out["realized_volatility"]["20d_pctile_5y"] > 80
    assert out["return_autocorrelation"]["daily_lag_1"]["n"] == 499
