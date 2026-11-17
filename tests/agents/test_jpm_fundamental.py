from unittest.mock import AsyncMock, patch

import pytest

from equity_trader.agents import jpm_fundamental as mod
from equity_trader.exceptions import DataFetchError
from equity_trader.schemas import AgentVerdict


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


def test_fundamentals_come_from_the_sec_with_market_valuation(monkeypatch):
    monkeypatch.setattr(mod.edg, "get_financial_statements",
                        lambda t: {"source": "SEC EDGAR XBRL company facts", "annual": []})
    monkeypatch.setattr(mod.yft, "get_valuation", lambda t: {"trailing_pe": 30.0})
    out = mod.get_fundamentals("NVDA")
    assert out["source"] == "SEC EDGAR XBRL company facts"
    assert out["valuation"] == {"trailing_pe": 30.0}


def test_fundamentals_fall_back_to_yfinance_and_say_so(monkeypatch):
    def no_filings(ticker):
        raise DataFetchError("No US-GAAP annual statements on EDGAR for TSM")

    monkeypatch.setattr(mod.edg, "get_financial_statements", no_filings)
    monkeypatch.setattr(mod.yft, "get_fundamentals", lambda t: {"income_stmt": {}})
    out = mod.get_fundamentals("TSM")
    assert out["source"].startswith("yfinance")
    assert "income_stmt" in out
