from unittest.mock import AsyncMock

import pytest

from equity_trader import runner as run_mod
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict


@pytest.fixture(autouse=True)
def _no_gathering(monkeypatch):
    monkeypatch.setattr(run_mod, "gather", lambda ticker: [])


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

    monkeypatch.setattr(run_mod, "get_quote", lambda t: {"last_price": 190.0})

    fake_orch = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
        weighted_score=3.5, weights_used={}, score_departure_rationale=None,
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
        weighted_score=1.0, weights_used={}, score_departure_rationale=None,
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


@pytest.mark.asyncio
async def test_model_written_error_note_does_not_mark_the_desk_failed():
    # Models fill error_note with data caveats. Only the runner may set it.
    async def caveated(t):
        return _v("wrong_name", err="Filing contents were unavailable.")

    out = await run_mod._run_one("jpm_fundamental", caveated, "NVDA")
    assert out.error_note is None
    assert out.agent == "jpm_fundamental"
    assert out.risks == ["y", "Filing contents were unavailable."]


@pytest.mark.asyncio
async def test_analyze_ticker_reports_each_desk_as_it_finishes(monkeypatch):
    from equity_trader.agents import AGENT_NAMES

    async def good(t):
        return _v("x")

    monkeypatch.setattr(run_mod, "ALL_AGENTS", [(n, good) for n in AGENT_NAMES])
    monkeypatch.setattr(run_mod, "get_quote", lambda t: {"last_price": 190.0})
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value="final"))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")

    seen = []
    await run_mod.analyze_ticker("NVDA", on_verdict=seen.append)
    assert sorted(v.agent for v in seen) == sorted(AGENT_NAMES)


@pytest.mark.asyncio
async def test_the_gatherer_runs_before_any_desk(monkeypatch):
    order = []

    async def desk(ticker):
        order.append("desk")
        return _v("jpm_fundamental")

    def gather(ticker):
        order.append("gather")
        return [{"tool": "tool_x", "seconds": 0.1, "error": None}]

    monkeypatch.setattr(run_mod, "ALL_AGENTS", [("jpm_fundamental", desk)])
    monkeypatch.setattr(run_mod, "get_quote", lambda t: {"last_price": 190.0})
    monkeypatch.setattr(run_mod, "gather", gather)
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value="final"))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")
    reports = []
    await run_mod.analyze_ticker("NVDA", on_gathered=reports.append)
    assert order == ["gather", "desk"]
    assert reports == [[{"tool": "tool_x", "seconds": 0.1, "error": None}]]
