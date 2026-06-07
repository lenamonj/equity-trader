import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import gs_technical as mod


@pytest.mark.asyncio
async def test_gs_technical_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="gs_technical", ticker="NVDA", recommendation="BUY",
        conviction=7, price_target_6mo=208.0,
        thesis=["Above 50dma"], risks=["RSI hot"], data_cited=["RSI 62"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "gs_technical"
