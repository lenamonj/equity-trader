from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from equity_trader.data import options_tools as ot
from tests.data.test_technical_tools import frame

TODAY = date.today()


def _in(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat()


def _chain():
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    calls = pd.DataFrame({
        "strike": strikes, "lastPrice": [21.0, 12.0, 5.0, 2.0, 1.0],
        "bid": [20.5, 11.5, 4.9, 1.9, 0.9], "ask": [21.5, 12.5, 5.1, 2.1, 1.1],
        "impliedVolatility": [0.50, 0.45, 0.40, 0.38, 0.37],
        "openInterest": [10, 50, 500, 300, 0], "volume": [1, 5, 4000, 100, np.nan]})
    puts = pd.DataFrame({
        "strike": strikes, "lastPrice": [1.0, 2.0, 5.0, 12.0, 21.0],
        "bid": [0.9, 1.9, 4.9, 11.5, 20.5], "ask": [1.1, 2.1, 5.1, 12.5, 21.5],
        "impliedVolatility": [0.60, 0.50, 0.40, 0.39, 0.38],
        "openInterest": [400, 200, 100, 20, 5], "volume": [50, 100, 200, 10, 1]})
    return calls, puts


@pytest.fixture
def options(monkeypatch):
    monkeypatch.setattr(ot, "_expiries", lambda t: [_in(10), _in(35), _in(95), _in(200)])
    monkeypatch.setattr(ot, "_chain", lambda t, expiry: _chain())
    monkeypatch.setattr(ot.yft, "get_quote", lambda t: {"last_price": 100.0})
    monkeypatch.setattr(ot.calendar_tools, "next_earnings_date", lambda t, today: _in(20))


def test_pick_expiries_takes_one_per_target_maturity():
    listed = [_in(3), _in(8), _in(29), _in(33), _in(91)]
    assert ot.pick_expiries(listed, TODAY) == [_in(8), _in(29), _in(33), _in(91)]


def test_horizon_expiry_skips_weeklies():
    listed = [_in(1), _in(8), _in(31), _in(122), _in(213)]
    assert ot.horizon_expiry(listed, TODAY) == _in(122)
    assert ot.horizon_expiry(listed[:3], TODAY) == _in(31)  # nothing far enough: take the last


def test_surface_reads_volatility_skew_and_positioning_per_expiry(options):
    out = ot.options_surface("ACME")
    first, second = out["expiries"][0], out["expiries"][1]
    assert [r["days"] for r in out["expiries"]] == [10, 35, 95, 200]
    assert first["atm_strike"] == 100.0 and first["atm_iv"] == 0.4
    assert first["implied_move"] == 0.1                 # 5 + 5 straddle on a 100 stock
    assert first["skew_90_110"] == pytest.approx(0.12)  # 0.50 put IV minus 0.38 call IV
    assert first["put_call_volume_ratio"] == 0.09       # 361 puts against 4,106 calls
    assert first["largest_put_oi"][0] == {"strike": 80.0, "open_interest": 400}
    assert first["spans_next_earnings"] is False and second["spans_next_earnings"] is True
    assert out["unusual_volume"][0] == {"expiry": _in(10), "type": "call", "strike": 100.0,
                                        "volume": 4000, "open_interest": 500}


def test_surface_without_listed_options_says_so(monkeypatch):
    monkeypatch.setattr(ot, "_expiries", lambda t: [])
    assert ot.options_surface("ACME")["expiries"] == []


def test_volatility_premium_compares_implied_with_realized(options, monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.setattr(ot, "history",
                        lambda t: frame(100 * np.cumprod(1 + rng.normal(0, 0.01, 400))))
    monkeypatch.setattr(ot.pattern_tools, "earnings_event_study",
                        lambda t: {"events": [{"reaction": 0.05}, {"reaction": -0.03}]})
    out = ot.volatility_premium("ACME")
    assert out["implied_1m"] == {"expiry": _in(35), "atm_iv": 0.4}
    assert out["implied_minus_realized_1m"] == pytest.approx(
        0.4 - out["realized_volatility"]["20d"], abs=1e-4)
    assert out["implied_minus_realized_1m"] > 0.2       # 40% implied against about 16% realized
    assert out["earnings"]["avg_absolute_move_last_reports"] == 0.04
    assert out["earnings"]["first_expiry_after_next_report"]["expiry"] == _in(35)


def test_chain_near_money_drops_strikes_with_no_open_interest(options):
    out = ot.options_chain_near_money("ACME")
    assert out["selected_expiry"] == _in(95)
    assert [c["strike"] for c in out["calls"]] == [80.0, 90.0, 100.0, 110.0]
    assert out["calls"][2] == {"strike": 100.0, "price": 5.0, "iv": 0.4,
                               "open_interest": 500, "volume": 4000}
    assert len(out["puts"]) == 5


def test_market_volatility_names_an_inverted_term_structure(monkeypatch):
    index = pd.bdate_range(end="2026-11-17", periods=60)
    closes = pd.DataFrame({"^VIX": np.linspace(15, 27, 60), "^VIX9D": np.linspace(14, 29, 60),
                           "^VIX3M": np.linspace(18, 25, 60), "^SKEW": 140.0, "^MOVE": 100.0},
                          index=index)
    monkeypatch.setattr(ot.yft, "get_closes", lambda symbols, period: closes[symbols])
    out = ot.market_volatility()
    assert out["vix_term_structure"]["vix3m_minus_vix"] == -2.0
    assert out["vix_term_structure"]["state"].startswith("inverted")
    vix = out["indexes"]["VIX (30-day S&P 500 implied volatility)"]
    assert vix["value"] == 27.0 and vix["pctile_1y"] == 100
