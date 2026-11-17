from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Recommendation = Literal["BUY", "HOLD", "SELL"]
Impact = Literal["LOW", "MEDIUM", "HIGH"]
PositionSize = Literal[
    "not sizeable", "starter (~1%)", "full (~3%)", "high conviction (~5%)"
]


class AgentVerdict(BaseModel):
    agent: str
    ticker: str
    recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    price_target_6mo: float | None = None
    thesis: list[str] = Field(default_factory=list, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=4)
    data_cited: list[str] = Field(default_factory=list)
    error_note: str | None = None


class CatalystEvent(BaseModel):
    event: str
    date: date
    expected_impact: Impact
    notes: str | None = None


class OrchestratorVerdict(BaseModel):
    ticker: str
    run_timestamp: datetime
    current_price: float
    final_recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    price_target_6mo: float | None = None
    weighted_score: float
    weights_used: dict[str, float]
    score_departure_rationale: str | None = None
    synthesis: str = Field(min_length=50, max_length=3000)
    key_agreements: list[str]
    key_disagreements: list[str]
    dominant_drivers: list[str]
    red_flags: list[str]
    position_sizing_suggestion: PositionSize
    stop_loss_level: float | None = None
    catalyst_calendar: list[CatalystEvent]
    agent_verdicts: list[AgentVerdict]
