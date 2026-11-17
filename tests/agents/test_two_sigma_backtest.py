from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from equity_trader.agents import two_sigma_backtest as mod
from equity_trader.schemas import AgentVerdict


def test_backtest_rsi_mean_reversion_returns_summary(monkeypatch):
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 500))
    df = pd.DataFrame({"Close": prices,
                        "Open": prices, "High": prices + 1, "Low": prices - 1,
                        "Volume": rng.integers(1e6, 5e6, 500)},
                       index=pd.date_range("2023-01-01", periods=500, freq="B"))
    monkeypatch.setattr(mod.yft, "get_price_history", lambda t, period="2y": df)
    summary = mod.backtest_simple_rules("AAPL")
    assert "rsi_mean_reversion" in summary
    assert "sma_crossover" in summary
    for v in summary.values():
        if isinstance(v, dict):
            assert {"trades", "hit_rate", "avg_return"}.issubset(v.keys())
    assert summary["as_of"] == "2024-11-29"


def test_backtest_rule_counts_a_run_of_signal_days_as_one_trade():
    close = pd.Series([100.0, 90.0, 80.0, 88.0, 96.0, 100.0, 104.0, 108.0],
                      index=pd.date_range("2026-01-01", periods=8, freq="B"))
    signals = pd.Series([False, True, True, True, False, False, False, False],
                        index=close.index)
    out = mod._backtest_rule(close, signals, hold=3)
    # One entry at 90, exit three sessions later at 96; the next two signal
    # days fall inside that holding period.
    assert out["trades"] == 1
    assert out["avg_return"] == pytest.approx(96.0 / 90.0 - 1.0)
    assert out["worst_trade"] == out["avg_return"]
    assert out["last_signal_date"] == "2026-01-06"
    assert out["signal_active_now"] is False


@pytest.mark.asyncio
async def test_two_sigma_backtest_run_returns_agent_verdict(monkeypatch):
    fake = AgentVerdict(
        agent="two_sigma_backtest", ticker="NVDA", recommendation="HOLD",
        conviction=4, thesis=["weak signal"], risks=["noisy"], data_cited=["bt"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert out.agent == "two_sigma_backtest"
