"""The contracts between the stages of a run.

A model writes only what needs judgment: a desk writes a `DeskReport`, the
chair writes a `ChairMemo`. Everything code already knows (which desk, which
ticker, the score, the price, the catalysts) is added in code, as an
`AgentVerdict` and an `OrchestratorVerdict`.

In the two schemas a model writes, the order of the fields is the order the
model produces them in. Evidence and reasoning come first and the call comes
last, so the call follows from the evidence. Those schemas use only
constraints the model provider can enforce while the model writes.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, create_model

Recommendation = Literal["BUY", "HOLD", "SELL"]
Impact = Literal["LOW", "MEDIUM", "HIGH"]
PositionSize = Literal[
    "not sizeable", "starter (~1%)", "full (~3%)", "high conviction (~5%)"
]
Signal = Literal["bullish", "bearish", "neutral"]
Unit = Literal["percent", "fraction", "usd", "ratio", "count", "index", "other"]


class Evidence(BaseModel):
    """One reading a desk relies on, copied from the data it was handed."""
    metric: str = Field(description=(
        "What was measured, in a few plain words, with its period where one applies: "
        "'RSI, 14-day' or 'Revenue growth, Q2 FY2027, year over year'."))
    value: float = Field(description=(
        "The number exactly as it appears in the DATA. Copy it; do not compute or rescale "
        "it."))
    unit: Unit = Field(description=(
        "The unit the DATA gives the number in. fraction for a share or rate written as a "
        "decimal, such as 0.29 for 29%; percent when it is written as a percentage, such "
        "as 29.0; usd for a dollar amount or price; ratio for a multiple; count for a "
        "tally or a rank; index for an indicator or index level; else other."))
    as_of: date = Field(description="The date the reading is for, taken from the DATA.")
    source: str = Field(description="The DATA section the reading was copied from.")
    signal: Signal = Field(description=(
        "What this reading says about owning the stock over the next 3 to 6 months."))


class CheckedEvidence(Evidence):
    """Evidence after code has looked for its value in the section it names."""
    in_data: bool


class DeskReport(BaseModel):
    """What a desk's model writes. Evidence first, the call last."""
    evidence: list[Evidence] = Field(min_length=1, max_length=12, description=(
        "The readings the call rests on, most important first. One number each."))
    thesis: list[str] = Field(min_length=1, max_length=6, description=(
        "Findings that led to the call. One or two sentences each, built on the evidence."))
    risks: list[str] = Field(min_length=1, max_length=4, description=(
        "What could invalidate the findings or hurt the position, and any data that was "
        "unavailable. One or two sentences each."))
    recommendation: Recommendation
    conviction: int = Field(ge=1, le=10, description="10 is bet-the-book confidence.")
    price_target_6mo: float | None = Field(description=(
        "A price six months out, anchored to the current price, or null if the desk's "
        "discipline does not produce one."))


def desk_report_for(sections: list[str]) -> type[DeskReport]:
    """The report schema for one desk. `source` can only name a section that
    desk was handed, so the model cannot cite anything else."""
    evidence = create_model(
        "Evidence", __base__=Evidence,
        source=(Literal[tuple(sections)], Evidence.model_fields["source"]))
    return create_model(
        "DeskReport", __base__=DeskReport,
        evidence=(list[evidence], DeskReport.model_fields["evidence"]))


class AgentVerdict(BaseModel):
    """A desk's report as the chair and the UI see it. `agent`, `ticker`,
    `in_data` and `error_note` are set in code, never by a model."""
    agent: str
    ticker: str
    evidence: list[CheckedEvidence] = Field(default_factory=list)
    thesis: list[str] = Field(default_factory=list, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=4)
    recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    price_target_6mo: float | None = None
    error_note: str | None = None


class CatalystEvent(BaseModel):
    event: str
    date: date
    expected_impact: Impact
    notes: str | None = None


class ChairMemo(BaseModel):
    """What the chair's model writes. The reading of the desks first, the call last."""
    key_agreements: list[str] = Field(description="Where desks agree. One sentence each.")
    key_disagreements: list[str] = Field(description=(
        "Where desks split, stated plainly. One sentence each."))
    red_flags: list[str] = Field(description=(
        "What could make this call wrong, including evidence a desk cited that was not "
        "found in its data. One sentence each."))
    dominant_drivers: list[str] = Field(max_length=3, description=(
        "The `agent` codes of the one to three desks whose reports drove the call."))
    synthesis: str = Field(description=(
        "A 150 to 300 word memo to a portfolio manager: direct, built on the desks' "
        "evidence, no hedging filler."))
    final_recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    score_departure_rationale: str | None = Field(description=(
        "Why the call goes against the sign of the weighted score, or null if it does not."))
    position_sizing_suggestion: PositionSize
    stop_loss_level: float | None = Field(description=(
        "A technically anchored stop for a BUY, taken from desk evidence. Null otherwise."))


def chair_memo_for(agents: list[str]) -> type[ChairMemo]:
    """The memo schema. `dominant_drivers` can only name a desk on the committee."""
    return create_model(
        "ChairMemo", __base__=ChairMemo,
        dominant_drivers=(list[Literal[tuple(agents)]],
                          ChairMemo.model_fields["dominant_drivers"]))


class OrchestratorVerdict(BaseModel):
    """The finished verdict: the chair's memo plus everything code supplies."""
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
