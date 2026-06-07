import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import two_sigma_backtest as mod


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
    for k, v in summary.items():
        assert {"trades", "hit_rate", "avg_return"}.issubset(v.keys())


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
