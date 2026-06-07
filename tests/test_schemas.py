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


def test_agent_verdict_thesis_min_one():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="BUY", conviction=5,
            thesis=[], risks=["y"], data_cited=["z"],
        )


def test_catalyst_event():
    e = CatalystEvent(event="Q2 earnings", date=date(2026, 8, 27),
                      expected_impact="HIGH", notes="Guidance update")
    assert e.expected_impact == "HIGH"
