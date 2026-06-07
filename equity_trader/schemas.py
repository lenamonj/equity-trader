from datetime import date, datetime
from typing import Literal, Optional
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
    price_target_6mo: Optional[float] = None
    thesis: list[str] = Field(min_length=1, max_length=6)
    risks: list[str] = Field(min_length=1, max_length=4)
    data_cited: list[str] = Field(min_length=1)
    error_note: Optional[str] = None


class CatalystEvent(BaseModel):
    event: str
    date: date
    expected_impact: Impact
    notes: Optional[str] = None
