from unittest.mock import AsyncMock, patch

import pytest

from equity_trader.agents import renaissance_pattern as mod
from equity_trader.schemas import AgentVerdict
from tests.agents import gathered_for


@pytest.mark.asyncio
async def test_renaissance_pattern_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="renaissance_pattern", ticker="NVDA", recommendation="HOLD",
        conviction=5, thesis=["Mixed regime"], risks=["Small sample"],
        data_cited=["ac1=0.02"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA", gathered_for(mod))
        assert isinstance(out, AgentVerdict)
        assert out.agent == "renaissance_pattern"
