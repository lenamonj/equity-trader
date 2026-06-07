import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import citadel_quant as mod


@pytest.mark.asyncio
async def test_citadel_quant_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="citadel_quant", ticker="NVDA", recommendation="BUY",
        conviction=6, price_target_6mo=205.0,
        thesis=["Strong momentum 12-1"], risks=["High vol"], data_cited=["mom=0.32"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "citadel_quant"
