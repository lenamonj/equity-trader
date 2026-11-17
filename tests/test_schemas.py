from datetime import date, datetime

import pytest
from pydantic import ValidationError

from equity_trader.schemas import (
    AgentVerdict,
    CatalystEvent,
    CheckedEvidence,
    DeskReport,
    Evidence,
    OrchestratorVerdict,
    desk_report_for,
)


def _reading(**overrides):
    return {"metric": "RSI, 14-day", "value": 16.0, "unit": "index",
            "as_of": "2026-11-17", "source": "technicals", "signal": "bearish", **overrides}


def _report(**overrides):
    return {"evidence": [_reading()], "thesis": ["Deeply oversold."], "risks": ["Bounce."],
            "recommendation": "SELL", "conviction": 7, "price_target_6mo": None, **overrides}


Report = desk_report_for(["technicals", "key_levels"])


def test_a_desk_report_is_written_evidence_first_and_the_call_last():
    assert list(DeskReport.model_fields) == [
        "evidence", "thesis", "risks", "recommendation", "conviction", "price_target_6mo"]
    assert list(Evidence.model_fields) == [
        "metric", "value", "unit", "as_of", "source", "signal"]
    report = Report.model_validate(_report())
    assert report.evidence[0].as_of == date(2026, 11, 17)


def test_a_desk_report_has_no_field_that_code_owns():
    owned = {"agent", "ticker", "error_note"}
    assert not owned & set(DeskReport.model_fields) and owned <= set(AgentVerdict.model_fields)
    assert "in_data" not in Evidence.model_fields and "in_data" in CheckedEvidence.model_fields


@pytest.mark.parametrize("change", [
    {"conviction": 11},
    {"recommendation": "STRONG_BUY"},
    {"evidence": []},
    {"thesis": []},
    {"risks": ["a", "b", "c", "d", "e"]},
    {"evidence": [_reading(source="fundamentals")]},   # a section this desk was not handed
    {"evidence": [_reading(signal="strongly bullish")]},
    {"evidence": [_reading(unit="basis points")]},
    {"evidence": [_reading(value="about sixteen")]},
    {"evidence": [_reading(as_of="last Tuesday")]},
])
def test_a_desk_report_outside_the_schema_is_rejected(change):
    with pytest.raises(ValidationError):
        Report.model_validate(_report(**change))


def test_a_failed_desks_verdict_needs_no_evidence():
    v = AgentVerdict(agent="a", ticker="T", recommendation="HOLD", conviction=1,
                     error_note="RuntimeError: boom")
    assert (v.evidence, v.thesis) == ([], [])


def test_catalyst_event():
    e = CatalystEvent(event="Q2 earnings", date=date(2026, 8, 27),
                      expected_impact="HIGH", notes="Guidance update")
    assert e.expected_impact == "HIGH"



def _sample_verdict(agent="a", rec="BUY"):
    return AgentVerdict(
        agent=agent, ticker="NVDA", recommendation=rec, conviction=7,
        price_target_6mo=200.0, thesis=["t"], risks=["r"], )


def test_orchestrator_verdict_minimal_ok():
    ov = OrchestratorVerdict(
        ticker="NVDA",
        run_timestamp=datetime.utcnow(),
        current_price=190.0,
        final_recommendation="BUY",
        conviction=7,
        price_target_6mo=210.0,
        weighted_score=4.2,
        weights_used={"jpm_fundamental": 25.0},
        score_departure_rationale=None,
        synthesis="Strong fundamentals; macro mixed but acceptable for horizon.",
        key_agreements=["FCF strong"],
        key_disagreements=["Macro vs technical"],
        dominant_drivers=["jpm_fundamental", "gs_technical"],
        red_flags=["Customer concentration"],
        position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0,
        catalyst_calendar=[],
        agent_verdicts=[_sample_verdict()],
    )
    assert ov.final_recommendation == "BUY"


def test_orchestrator_verdict_position_sizing_literal():
    with pytest.raises(ValidationError):
        OrchestratorVerdict(
            ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
            final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
            weighted_score=4.2, weights_used={"a": 100.0},
            score_departure_rationale=None, synthesis="x",
            key_agreements=[], key_disagreements=[], dominant_drivers=[],
            red_flags=[], position_sizing_suggestion="MAX OUT",
            stop_loss_level=None, catalyst_calendar=[], agent_verdicts=[],
        )
