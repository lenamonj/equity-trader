import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import jane_street_etf as mod


@pytest.mark.asyncio
async def test_jane_street_etf_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="jane_street_etf", ticker="NVDA", recommendation="BUY",
        conviction=7, price_target_6mo=200.0,
        thesis=["Sector ETF leading SPY by 4% over 3m",
                "Ticker outperforming sector by 2% with rising volume"],
        risks=["XLK weighting concentration in NVDA = 6%"],
        data_cited=["XLK 3m ret 12%, ticker 3m ret 14%"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "jane_street_etf"
