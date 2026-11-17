from unittest.mock import AsyncMock

import pytest

from equity_trader import runner as run_mod
from equity_trader.schemas import AgentVerdict, DeskReport, Evidence, OrchestratorVerdict


@pytest.fixture(autouse=True)
def _no_gathering(monkeypatch):
    monkeypatch.setattr(run_mod, "gather", lambda ticker, sections: GATHERED)


def _reading(metric, value):
    return Evidence(metric=metric, value=value, unit="index", as_of="2026-11-17",
                    source="technicals", signal="bearish")


def _report(rec="BUY", conv=6, evidence=()):
    return DeskReport(evidence=list(evidence) or [_reading("RSI, 14-day", 16.0)],
                      thesis=["x"], risks=["y"], recommendation=rec, conviction=conv,
                      price_target_6mo=None)


# What the gatherer fetched, as far as these desks cite it.
GATHERED = {"technicals": {"data": {"rsi_14": 15.97, "note": "ATR is 11.44"}, "error": None}}


def _v(name, rec="BUY", conv=6, err=None):
    return AgentVerdict(agent=name, ticker="NVDA", recommendation=rec,
                        conviction=conv, thesis=["x"], risks=["y"],
                        error_note=err)


@pytest.mark.asyncio
async def test_analyze_ticker_calls_all_agents_then_orchestrator(monkeypatch, tmp_path):
    from datetime import datetime

    from equity_trader.agents import AGENT_NAMES

    async def fake_agent_run(t, gathered):
        return _v(fake_agent_run._name)

    fake_funcs = []
    for n in AGENT_NAMES:
        f = AsyncMock(side_effect=lambda t, gathered: _report())
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

    async def good(t, gathered):
        return _report()

    async def bad(t, gathered):
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
async def test_the_runner_adds_what_code_knows_to_a_desks_report():
    report = _report("SELL", 8, [_reading("RSI, 14-day", 16.0), _reading("ATR", 11.44),
                                 _reading("A number from nowhere", 987.65)])

    async def desk(t, gathered):
        return report

    out = await run_mod._run_one("gs_technical", desk, "NVDA", GATHERED)

    assert isinstance(out, AgentVerdict)
    assert (out.agent, out.ticker, out.error_note) == ("gs_technical", "NVDA", None)
    assert (out.recommendation, out.conviction, out.thesis) == ("SELL", 8, ["x"])
    # Each cited number is looked for in the section it names.
    assert [(e.metric, e.in_data) for e in out.evidence] == [
        ("RSI, 14-day", True), ("ATR", True), ("A number from nowhere", False)]


@pytest.mark.asyncio
async def test_analyze_ticker_reports_each_desk_as_it_finishes(monkeypatch):
    from equity_trader.agents import AGENT_NAMES

    async def good(t, gathered):
        return _report()

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

    async def desk(ticker, gathered):
        assert gathered == fetched
        order.append("desk")
        return _report()

    fetched = {**GATHERED, "company_snapshot": {"data": {"price": 190.0}, "error": None}}

    def gather(ticker, sections):
        order.append("gather")
        assert sections is run_mod.ALL_DATA
        return fetched

    monkeypatch.setattr(run_mod, "ALL_AGENTS", [("jpm_fundamental", desk)])
    monkeypatch.setattr(run_mod, "get_quote", lambda t: {"last_price": 190.0})
    monkeypatch.setattr(run_mod, "gather", gather)
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value="final"))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")
    reports = []
    await run_mod.analyze_ticker("NVDA", on_gathered=reports.append)
    assert order == ["gather", "desk"]
    assert reports == [fetched]
