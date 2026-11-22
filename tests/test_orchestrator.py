import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from agents import AgentOutputSchema
from pydantic import ValidationError

from equity_trader import orchestrator as orch
from equity_trader.config import AGENT_WEIGHTS
from equity_trader.schemas import (
    AgentVerdict,
    ChairMemo,
    CheckedEvidence,
    OrchestratorVerdict,
    chair_memo_for,
)

# The gatherer's `recent_events` entry, as the runner hands it to the chair.
EVENTS = {"error": None, "data": {
    "as_of": "2026-11-20", "sector_etf": "XLK", "notes": "How to read this.",
    "events": [{
        "rank": 1, "session": "2026-11-12",
        "filings": [{"filing_date": "2026-11-12",
                     "items": ["5.02 Departure or appointment of directors or officers"],
                     "excerpt": "Item 5.02. The Chief Executive Officer resigned."}],
        "heavy_volume": True, "unexplained": False, "opening_gap": -0.1308,
        "close_to_close_return": -0.0434, "volume_vs_50d_avg": 5.21,
        "sector_etf_return": -0.0034, "return_vs_sector_etf": -0.04, "move": 0.1308,
        "material": True,
        "headlines": [{"title": "Acme chief executive resigns", "source": "Reuters",
                       "published": "2026-11-12"}]}]}}
NO_EVENTS = {"data": None, "error": "TickerNotFoundError: Ticker not found: NVDA"}

Memo = chair_memo_for(list(AGENT_WEIGHTS))


def _v(agent, rec, conv=7, tgt=200.0, err=None):
    return AgentVerdict(agent=agent, ticker="NVDA", recommendation=rec,
                        conviction=conv, price_target_6mo=tgt,
                        thesis=["t"], risks=["r"], error_note=err)


def _memo(rec, rationale=None, drivers=("jpm_fundamental",)):
    return Memo(
        key_agreements=["FCF and technicals align"],
        key_disagreements=["Macro against fundamental on the rate path"],
        red_flags=["existing flag"], dominant_drivers=list(drivers),
        synthesis="A synthesis sufficiently long to clear the schema minimum length.",
        final_recommendation=rec, conviction=7, score_departure_rationale=rationale,
        position_sizing_suggestion="starter (~1%)", stop_loss_level=None)


async def _run_with(memo, verdicts, ticker="NVDA", price=190.0):
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": memo})())
        return await orch.run(ticker, verdicts, current_price=price, recent_events=EVENTS)


@pytest.fixture(autouse=True)
def _no_real_calendar(monkeypatch):
    # Orchestrator tests do not hit yfinance or FRED for the calendar unless
    # a test sets orch.get_real_catalysts itself.
    monkeypatch.setattr(orch, "get_real_catalysts", lambda ticker, today: [])


@pytest.mark.asyncio
async def test_the_chair_writes_a_memo_and_code_supplies_everything_else(monkeypatch):
    real_date = date.today() + timedelta(days=90)
    monkeypatch.setattr(orch, "get_real_catalysts", lambda ticker, t: [
        {"event": "Earnings release", "date": real_date.isoformat(),
         "expected_impact": "HIGH", "notes": "Source: yfinance calendar"}])
    verdicts = [_v("jpm_fundamental", "BUY", 8, 215), _v("gs_technical", "BUY", 7, None)]

    out = await _run_with(_memo("BUY", drivers=["jpm_fundamental", "gs_technical"]), verdicts)

    assert isinstance(out, OrchestratorVerdict)
    # The judgment is the model's.
    assert out.final_recommendation == "BUY" and out.conviction == 7
    assert "synthesis sufficiently long" in out.synthesis
    assert out.dominant_drivers == ["jpm_fundamental", "gs_technical"]
    # Everything else is code's.
    assert (out.ticker, out.current_price) == ("NVDA", 190.0)
    assert out.run_timestamp.year >= 2026
    assert out.weighted_score == pytest.approx(orch.compute_weighted_score(verdicts,
                                                                           AGENT_WEIGHTS))
    assert out.price_target_6mo == 215.0
    assert out.weights_used == AGENT_WEIGHTS
    assert out.agent_verdicts == verdicts
    assert [(c.date, c.event) for c in out.catalyst_calendar] == [(real_date,
                                                                   "Earnings release")]
    top = out.recent_events.events[0]
    assert (out.recent_events.sector_etf, top.session, top.opening_gap, top.material) == (
        "XLK", date(2026, 11, 12), -0.1308, True)
    assert top.filings[0].items == ["5.02 Departure or appointment of directors or officers"]
    assert top.headlines[0].source == "Reuters"


def test_the_memo_has_no_field_that_code_owns():
    owned = {"ticker", "run_timestamp", "current_price", "weighted_score", "weights_used",
             "price_target_6mo", "catalyst_calendar", "agent_verdicts", "recent_events"}
    assert not owned & set(ChairMemo.model_fields)
    assert owned <= set(OrchestratorVerdict.model_fields)


def test_the_memo_is_written_reasoning_first_and_the_schema_is_enforceable():
    order = list(AgentOutputSchema(Memo, strict_json_schema=True).json_schema()["properties"])
    assert order.index("key_disagreements") < order.index("synthesis") < order.index(
        "final_recommendation")
    assert AgentOutputSchema(Memo, strict_json_schema=True).is_strict_json_schema()


def test_dominant_drivers_can_only_name_a_desk():
    with pytest.raises(ValidationError):
        _memo("HOLD", drivers=["jpm_fundamental", "Strong margins offset by debt."])


@pytest.mark.asyncio
async def test_orchestrator_flags_call_against_score_without_rationale():
    bearish = [_v("jpm_fundamental", "SELL", 8), _v("gs_technical", "SELL", 7)]
    out = await _run_with(_memo("BUY"), bearish)
    assert out.weighted_score < 0
    assert out.red_flags == ["existing flag", orch._UNEXPLAINED_DEPARTURE]


@pytest.mark.asyncio
async def test_orchestrator_accepts_call_against_score_with_rationale():
    bearish = [_v("jpm_fundamental", "SELL", 8)]
    out = await _run_with(_memo("BUY", rationale="Macro desk downweighted."), bearish)
    assert out.red_flags == ["existing flag"]


@pytest.mark.asyncio
async def test_orchestrator_reports_weights_the_score_was_computed_with():
    verdicts = [_v("jpm_fundamental", "BUY", 8),
                _v("gs_technical", "HOLD", 1, None, err="boom")]
    out = await _run_with(_memo("HOLD"), verdicts)
    assert "gs_technical" not in out.weights_used
    assert abs(sum(out.weights_used.values()) - 100.0) < 1e-9
    assert out.weights_used["jpm_fundamental"] > AGENT_WEIGHTS["jpm_fundamental"]


def test_the_chair_is_handed_one_json_document():
    reading = CheckedEvidence(metric="RSI, 14-day", value=16.0, unit="index",
                              as_of=date(2026, 11, 17), source="technicals",
                              signal="bearish", in_data=True)
    verdicts = [
        _v("gs_technical", "SELL", 8, 148.8).model_copy(update={"evidence": [reading]}),
        _v("jpm_fundamental", "HOLD", 1, None, err="RuntimeError: boom"),
    ]
    catalysts = [{"event": "CPI release", "date": "2026-12-10",
                  "expected_impact": "MEDIUM", "notes": "Source: FRED release calendar"}]

    doc = json.loads(orch._format_input("NVDA", verdicts, 153.81500244140625, -4.67,
                                        date(2026, 11, 17), catalysts, EVENTS))

    assert (doc["today"], doc["horizon_end"]) == ("2026-11-17", "2027-05-16")
    assert (doc["ticker"], doc["current_price"], doc["weighted_score"]) == ("NVDA", 153.82,
                                                                            -4.67)
    assert doc["catalysts"] == catalysts
    assert doc["recent_events"] == EVENTS["data"]
    goldman, morgan = doc["desks"]
    assert (goldman["agent"], goldman["house"], goldman["discipline"]) == (
        "gs_technical", "Goldman Sachs", "Technical")
    assert goldman["weight"] == 16.0  # 12 of the 75 left once the failed desk drops out
    assert goldman["report"]["evidence"] == [{
        "metric": "RSI, 14-day", "value": 16.0, "unit": "index", "as_of": "2026-11-17",
        "source": "technicals", "signal": "bearish", "in_data": True}]
    assert set(goldman["report"]) == {"evidence", "thesis", "risks", "recommendation",
                                      "conviction", "price_target_6mo"}
    assert morgan["failed"] == "RuntimeError: boom" and "report" not in morgan


def test_catalysts_include_scheduled_macro_releases(monkeypatch):
    from equity_trader.data import calendar_tools
    monkeypatch.setattr(calendar_tools, "_calendar",
                        lambda ticker: {"Earnings Date": [date(2026, 11, 20)]})
    monkeypatch.setattr(calendar_tools.fred_tools, "get_release_calendar",
                        lambda horizon_days: [{"date": "2026-12-10", "release": "CPI"}])
    out = calendar_tools.get_real_catalysts("NVDA", date(2026, 11, 17))
    assert [(c["date"], c["event"], c["expected_impact"]) for c in out] == [
        ("2026-11-20", "Earnings release", "HIGH"), ("2026-12-10", "CPI release", "MEDIUM")]


@pytest.mark.asyncio
async def test_orchestrator_falls_back_on_llm_failure():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await orch.run("NVDA", verdicts, current_price=190.0, recent_events=EVENTS)
        assert isinstance(out, OrchestratorVerdict)
        assert "mechanical aggregation" in out.synthesis.lower()
        # The events are on the report whether or not the chair wrote a memo.
        assert out.recent_events.events[0].session == date(2026, 11, 12)


@pytest.mark.asyncio
async def test_events_that_could_not_be_gathered_are_reported_as_unavailable():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    doc = json.loads(orch._format_input("NVDA", verdicts, 190.0, 5.0, date(2026, 11, 17), [],
                                        NO_EVENTS))
    assert doc["recent_events"] == {"unavailable": NO_EVENTS["error"]}
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": _memo("BUY")})())
        out = await orch.run("NVDA", verdicts, current_price=190.0, recent_events=NO_EVENTS)
    assert out.recent_events is None


def test_the_chair_must_address_the_largest_recent_event():
    text = " ".join(orch._INSTRUCTIONS.split())
    assert "`recent_events`" in text
    assert "the only future dated events you may cite" in text
    assert "ranked 1" in text and "`material`" in text
