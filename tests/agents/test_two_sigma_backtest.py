from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from equity_trader.agents import two_sigma_backtest as mod
from equity_trader.schemas import AgentVerdict
from tests.agents import gathered_for


def _frame(prices):
    prices = np.asarray(prices, dtype=float)
    return pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices,
                         "Volume": np.full(len(prices), 1e6)},
                        index=pd.date_range("2018-01-01", periods=len(prices), freq="B"))


def test_backtest_covers_long_and_short_rules_at_horizon_matched_holds(monkeypatch):
    rng = np.random.default_rng(0)
    df = _frame(100 * np.cumprod(1 + rng.normal(0.0003, 0.02, 2000)))
    monkeypatch.setattr(mod, "history", lambda ticker: df)
    monkeypatch.setattr(mod.yft, "get_info", lambda ticker: {"sector": "Technology"})

    out = mod.backtest_rules("AAPL")

    assert len(out["rules"]) == 11 and out["tests_run"] == 33
    assert {r["direction"] for r in out["rules"].values()} == {"long", "short"}
    for rule in out["rules"].values():
        assert set(rule["holds"]) == {"20d", "63d", "126d"}
        for result in rule["holds"].values():
            assert {"trades", "hit_rate", "baseline_avg_return", "edge_after_cost", "t_stat",
                    "edge_first_half", "edge_second_half"} <= set(result)
    assert set(out["same_rules_63d_elsewhere"]) == {"SPY", "XLK"}
    assert set(out["same_rules_63d_elsewhere"]["SPY"]["death_cross"]) == {
        "trades", "edge", "t_stat"}
    assert out["as_of"] == "2025-08-29" and out["cost_assumption"] == mod.COST


def test_backtest_rule_counts_a_run_of_signal_days_as_one_trade():
    close = pd.Series([100.0, 90.0, 80.0, 88.0, 96.0, 100.0, 104.0, 108.0],
                      index=pd.date_range("2026-01-01", periods=8, freq="B"))
    signals = pd.Series([False, True, True, True, False, False, False, False],
                        index=close.index)
    out = mod._backtest_rule(close, signals, hold=3)
    # One entry at 90, exit three sessions later at 96; the next two signal
    # days fall inside that holding period.
    assert out["trades"] == 1
    assert out["avg_return"] == pytest.approx(96.0 / 90.0 - 1.0, abs=1e-4)
    assert out["worst_trade"] == out["avg_return"]
    assert out["edge"] == pytest.approx(out["avg_return"] - out["baseline_avg_return"], abs=1e-4)
    assert out["edge_after_cost"] == pytest.approx(out["edge"] - mod.COST, abs=1e-4)


def test_short_rule_is_scored_in_its_own_direction():
    close = pd.Series(np.linspace(100, 60, 41),
                      index=pd.date_range("2026-01-01", periods=41, freq="B"))
    signals = pd.Series(False, index=close.index)
    signals.iloc[[0, 10, 20]] = True
    out = mod._backtest_rule(close, signals, hold=10, direction=-1)
    assert out["trades"] == 3
    assert out["hit_rate"] == 1.0       # the stock fell after every short signal
    assert out["worst_trade"] > 0       # so even the worst short made money
    assert out["avg_return"] < 0


def test_rule_with_no_signals_reports_no_trades():
    close = pd.Series(np.linspace(100, 110, 30))
    out = mod._backtest_rule(close, pd.Series(False, index=close.index), hold=5)
    assert out["trades"] == 0 and out["hit_rate"] is None and out["edge"] is None


def test_breakout_rule_fires_on_the_first_new_high():
    df = _frame(np.concatenate([np.full(300, 100.0), [101.0, 102.0]]))
    direction, signals = mod._signals(df)["new_52_week_high"]
    assert direction == 1
    assert signals.iloc[-2] and not signals.iloc[-3]


@pytest.mark.asyncio
async def test_two_sigma_backtest_run_returns_agent_verdict(monkeypatch):
    fake = AgentVerdict(
        agent="two_sigma_backtest", ticker="NVDA", recommendation="HOLD",
        conviction=4, thesis=["weak signal"], risks=["noisy"], data_cited=["bt"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA", gathered_for(mod))
        assert out.agent == "two_sigma_backtest"
