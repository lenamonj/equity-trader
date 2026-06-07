import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import bridgewater_macro as mod


@pytest.mark.asyncio
async def test_bridgewater_macro_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="bridgewater_macro", ticker="NVDA", recommendation="HOLD",
        conviction=6, thesis=["Curve flat"], risks=["Recession"], data_cited=["FRED"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "bridgewater_macro"
