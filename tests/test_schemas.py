import pytest
from datetime import date
from pydantic import ValidationError
from equity_trader.schemas import AgentVerdict, CatalystEvent, Recommendation


def test_agent_verdict_minimal_ok():
    v = AgentVerdict(
        agent="jpm_fundamental",
        ticker="NVDA",
        recommendation="BUY",
        conviction=8,
        price_target_6mo=215.0,
        thesis=["FCF expanding", "Capex visibility"],
        risks=["China exports"],
        data_cited=["10-Q Q1"],
    )
    assert v.recommendation == "BUY"
    assert v.error_note is None


def test_agent_verdict_conviction_range():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="BUY", conviction=11,
            thesis=["x"], risks=["y"], data_cited=["z"],
        )


def test_agent_verdict_recommendation_literal():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="STRONG_BUY", conviction=5,
            thesis=["x"], risks=["y"], data_cited=["z"],
        )


def test_agent_verdict_empty_lists_allowed():
    # Models occasionally return empty thesis/risks/data_cited; we accept them
    # rather than fail Pydantic validation and lose the rest of the verdict.
    v = AgentVerdict(
        agent="a", ticker="T", recommendation="HOLD", conviction=3,
        thesis=[], risks=[], data_cited=[],
    )
    assert v.thesis == []


def test_catalyst_event():
    e = CatalystEvent(event="Q2 earnings", date=date(2026, 8, 27),
                      expected_impact="HIGH", notes="Guidance update")
    assert e.expected_impact == "HIGH"


from datetime import datetime
from equity_trader.schemas import OrchestratorVerdict


def _sample_verdict(agent="a", rec="BUY"):
    return AgentVerdict(
        agent=agent, ticker="NVDA", recommendation=rec, conviction=7,
        price_target_6mo=200.0, thesis=["t"], risks=["r"], data_cited=["d"],
    )


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
        weight_overrides_rationale=None,
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
            weight_overrides_rationale=None, synthesis="x",
            key_agreements=[], key_disagreements=[], dominant_drivers=[],
            red_flags=[], position_sizing_suggestion="MAX OUT",
            stop_loss_level=None, catalyst_calendar=[], agent_verdicts=[],
        )
