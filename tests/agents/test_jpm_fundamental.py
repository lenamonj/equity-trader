import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import jpm_fundamental as mod


@pytest.mark.asyncio
async def test_jpm_fundamental_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="jpm_fundamental", ticker="NVDA", recommendation="BUY",
        conviction=8, price_target_6mo=215.0,
        thesis=["FCF expanding"], risks=["China"], data_cited=["10-Q"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "jpm_fundamental"
