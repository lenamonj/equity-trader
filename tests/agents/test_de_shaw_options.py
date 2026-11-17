from unittest.mock import AsyncMock, patch

import pytest

from equity_trader.agents import de_shaw_options as mod
from equity_trader.schemas import AgentVerdict
from tests.agents import gathered_for


@pytest.mark.asyncio
async def test_de_shaw_options_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="de_shaw_options", ticker="NVDA", recommendation="BUY",
        conviction=6, price_target_6mo=210.0,
        thesis=["IV cheap vs realized"], risks=["Skew elevated"],
        data_cited=["ATM IV 28%"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA", gathered_for(mod))
        assert isinstance(out, AgentVerdict)
        assert out.agent == "de_shaw_options"
