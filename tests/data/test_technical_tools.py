import numpy as np
import pandas as pd
import pytest

from equity_trader.data import technical_tools as tt


def frame(close, volume=None, open_=None, high=None, low=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"Open": close if open_ is None else open_,
         "High": close if high is None else high,
         "Low": close if low is None else low,
         "Close": close,
         "Volume": np.full(len(close), 1e6) if volume is None else volume},
        index=pd.bdate_range(end="2026-11-17", periods=len(close)))


def use(monkeypatch, df):
    monkeypatch.setattr(tt.yft, "get_price_history", lambda ticker, period="max": df)


def test_technicals_read_an_uptrend_on_every_timeframe(monkeypatch):
    use(monkeypatch, frame(np.linspace(50, 150, 600)))
    out = tt.compute_technicals("ACME")
    assert out["as_of"] == "2026-11-17" and out["last_close"] == 150.0
    assert set(out["averages"]) == {"sma_20", "sma_50", "sma_100", "sma_200"}
    assert out["averages"]["sma_200"]["price_vs"] > 0
    assert out["averages"]["sma_200"]["slope_20d"] > 0
    assert out["drawdown_from_52w_high"] == 0.0
    assert out["weekly"]["return_52w"] > 0 and out["monthly"]["return_12m"] > 0
    assert out["bollinger_20"]["position"] > 0.5


def test_technicals_date_the_last_moving_average_cross(monkeypatch):
    use(monkeypatch, frame(np.concatenate([np.linspace(200, 100, 400),
                                           np.linspace(100, 220, 400)])))
    cross = tt.compute_technicals("ACME")["last_50_200_cross"]
    assert cross["type"] == "golden cross"
    assert cross["date"] > "2025-06-01"


def test_key_levels_find_swing_points_on_each_side_of_price(monkeypatch):
    close = np.concatenate([np.linspace(100, 150, 101), np.linspace(150, 80, 81)[1:],
                            np.linspace(80, 100, 72)[1:]])
    use(monkeypatch, frame(close))
    out = tt.key_levels("ACME")
    assert out["resistance"][0]["price"] == 150.0
    assert out["resistance"][0]["kind"] == "swing high"
    assert out["resistance"][0]["pct_from_price"] == 0.5
    assert out["support"][0]["price"] == 80.0 and out["support"][0]["kind"] == "swing low"
    assert out["high_52w"]["price"] == 150.0 and out["low_52w"]["price"] == 80.0
    assert sum(z["share_of_volume"] for z in out["heaviest_volume_zones"]) < 1.0


def test_key_levels_list_only_gaps_that_have_not_been_filled(monkeypatch):
    close = np.full(60, 100.0)
    close[40:] = 110.0
    open_, low = close.copy(), close.copy()
    open_[10], low[10] = 103.0, 99.5   # gap up that traded back through 100 the same day
    low[40:] = 108.0                   # gap up from 100 to 110 that never came back
    use(monkeypatch, frame(close, open_=open_, low=low, high=np.maximum(close, open_)))
    gaps = tt.key_levels("ACME")["unfilled_gaps"]
    assert gaps == [{"date": gaps[0]["date"], "direction": "up", "from": 100.0, "to": 110.0,
                     "pct": 0.1}]


def test_volume_analysis_separates_heavy_up_days_from_light_down_days(monkeypatch):
    steps = np.where(np.arange(120) % 2 == 0, 1.015, 0.99)
    volume = np.where(np.arange(120) % 2 == 0, 3e6, 1e6)
    use(monkeypatch, frame(100 * np.cumprod(steps), volume=volume))
    out = tt.volume_analysis("ACME")
    assert out["up_down_volume_ratio_20d"] == 3.0
    assert out["obv_trend_20d"] == "rising"
    assert out["accumulation_days_25"] > 0 and out["distribution_days_25"] == 0
    assert len(out["heaviest_days_60"]) == 3


def test_relative_strength_measures_against_market_and_sector(monkeypatch):
    n = 300
    wobble = 100 + np.sin(np.arange(n))
    closes = pd.DataFrame({"ACME": np.linspace(100, 200, n), "SPY": wobble, "XLK": wobble},
                          index=pd.bdate_range(end="2026-11-17", periods=n))
    monkeypatch.setattr(tt.yft, "get_info", lambda t: {"sector": "Technology"})
    monkeypatch.setattr(tt.yft, "get_closes", lambda symbols, period: closes[symbols])
    out = tt.relative_strength("acme")
    assert out["sector_etf"] == "XLK" and set(out["versus"]) == {"SPY", "XLK"}
    spy = out["versus"]["SPY"]
    assert spy["excess_return_3m"] > 0 and spy["excess_return_12m"] > 0
    assert spy["ratio_position_in_1y_range"] == pytest.approx(1.0, abs=0.02)


def test_relative_strength_without_a_sector_etf_uses_the_market_only(monkeypatch):
    closes = pd.DataFrame({"ACME": np.linspace(100, 90, 300), "SPY": np.linspace(100, 120, 300)},
                          index=pd.bdate_range(end="2026-11-17", periods=300))
    monkeypatch.setattr(tt.yft, "get_info", lambda t: {})
    monkeypatch.setattr(tt.yft, "get_closes", lambda symbols, period: closes[symbols])
    out = tt.relative_strength("ACME")
    assert out["sector_etf"] is None and list(out["versus"]) == ["SPY"]
    assert out["versus"]["SPY"]["excess_return_6m"] < 0
