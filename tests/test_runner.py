import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader import runner as run_mod


def _v(name, rec="BUY", conv=6, err=None):
    return AgentVerdict(agent=name, ticker="NVDA", recommendation=rec,
                        conviction=conv, thesis=["x"], risks=["y"],
                        data_cited=["z"], error_note=err)


@pytest.mark.asyncio
async def test_analyze_ticker_calls_all_agents_then_orchestrator(monkeypatch, tmp_path):
    from datetime import datetime
    from equity_trader.agents import AGENT_NAMES

    async def fake_agent_run(t):
        return _v(fake_agent_run._name)

    fake_funcs = []
    for n in AGENT_NAMES:
        f = AsyncMock(side_effect=lambda t, _n=n: _v(_n))
        fake_funcs.append((n, f))
    monkeypatch.setattr(run_mod, "ALL_AGENTS", fake_funcs)

    fake_quote = lambda t: {"last_price": 190.0, "market_cap": 0, "day_high": 0, "day_low": 0}
    monkeypatch.setattr(run_mod, "get_quote", fake_quote)

    fake_orch = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
        weighted_score=3.5, weights_used={}, weight_overrides_rationale=None,
        synthesis="x" * 60,
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[],
        agent_verdicts=[_v(n) for n in AGENT_NAMES],
    )
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value=fake_orch))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")

    out = await run_mod.analyze_ticker("NVDA")
    assert isinstance(out, OrchestratorVerdict)
    for _, fn in fake_funcs:
        fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_ticker_survives_one_agent_failure(monkeypatch):
    from datetime import datetime
    from equity_trader.agents import AGENT_NAMES

    async def good(t):
        return _v("g")

    async def bad(t):
        raise RuntimeError("boom")

    fake_funcs = [(n, good if i > 0 else bad) for i, n in enumerate(AGENT_NAMES)]
    monkeypatch.setattr(run_mod, "ALL_AGENTS", fake_funcs)
    monkeypatch.setattr(run_mod, "get_quote",
                         lambda t: {"last_price": 190.0, "market_cap": 0,
                                    "day_high": 0, "day_low": 0})

    fake_orch = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="HOLD", conviction=5, price_target_6mo=None,
        weighted_score=1.0, weights_used={}, weight_overrides_rationale=None,
        synthesis="x" * 60,
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="not sizeable",
        stop_loss_level=None, catalyst_calendar=[],
        agent_verdicts=[],
    )
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value=fake_orch))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")

    out = await run_mod.analyze_ticker("NVDA")
    assert isinstance(out, OrchestratorVerdict)
