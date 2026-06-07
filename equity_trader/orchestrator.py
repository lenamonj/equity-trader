import logging
from datetime import date, datetime, timedelta
from agents import Runner

from equity_trader.agents.base import build_agent
from equity_trader.config import AGENT_WEIGHTS, ORCHESTRATOR_MODEL
from equity_trader.data.calendar_tools import get_real_catalysts
from equity_trader.schemas import AgentVerdict, CatalystEvent, OrchestratorVerdict
from equity_trader.scoring import compute_weighted_score, redistribute_weights


_HORIZON_DAYS = 180


log = logging.getLogger(__name__)


_INSTRUCTIONS = """You are the head of a multi-strategy investment committee.
Eight specialist agents have each analyzed the ticker. Your job is to
synthesize their structured verdicts into a final BUY / HOLD / SELL for the
3-6 month horizon.

You will be given:
- TODAY's date (use it for every temporal claim — do NOT rely on training data
  for dates).
- Each agent's full AgentVerdict (recommendation, conviction, target, thesis,
  risks, data_cited, error_note).
- The default weights for each agent.
- The deterministic weighted_score already computed from those weights.
- The current price of the ticker.
- A REAL_CATALYSTS list sourced from yfinance (earnings, dividends). Treat
  these as ground truth.

Hard rules:
1. Horizon is strictly 3-6 months from TODAY. Reject theses pegged to longer
   windows or to past dates.
2. You may use the default weights or override them. If you override, populate
   `weight_overrides_rationale` explaining why (e.g., 'downweighting macro
   because Fed meeting is post-horizon').
3. If your `final_recommendation` sign disagrees with the sign of
   `weighted_score`, you MUST populate `weight_overrides_rationale`.
4. Surface disagreements explicitly in `key_disagreements`. Do not paper over splits.
5. `price_target_6mo` is conviction-weighted average of agent targets. Agents
   without a target are excluded from the average.
6. You may NOT introduce new data not cited by an agent. You are a synthesizer,
   not an analyst.
7. `position_sizing_suggestion` must be one of: 'not sizeable', 'starter (~1%)',
   'full (~3%)', 'high conviction (~5%)'.
8. `stop_loss_level` is technical-anchored; populate only on BUY recommendations,
   otherwise null.
9. `catalyst_calendar` must START with every entry from REAL_CATALYSTS verbatim
   (same date, event, expected_impact). You MAY add additional soft catalysts
   (e.g., "Capital Markets Day", "Q3 guidance update") ONLY if an agent's
   data_cited or thesis references them. Every date you write MUST be on or
   after TODAY and no more than 6 months after TODAY. NEVER write a date you
   cannot trace to REAL_CATALYSTS or an agent citation. Any catalyst whose
   date falls outside [TODAY, TODAY+6mo] will be dropped post-hoc.

Write a 150-300 word synthesis in `synthesis` that reads like a PM memo -
direct, evidence-cited, no hedging filler.

Return an OrchestratorVerdict.
"""


_agent = build_agent(
    name="orchestrator",
    instructions=_INSTRUCTIONS,
    tools=[],
    spec=ORCHESTRATOR_MODEL,
    output_type=OrchestratorVerdict,
)


def _format_input(ticker: str, verdicts: list[AgentVerdict],
                   current_price: float, weighted_score: float,
                   today: date, real_catalysts: list[dict]) -> str:
    horizon_end = today + timedelta(days=_HORIZON_DAYS)
    catalyst_lines = (
        [f"  - {c['date']} | {c['event']} | {c['expected_impact']} | {c['notes']}"
         for c in real_catalysts]
        if real_catalysts else ["  (none in horizon)"]
    )
    lines = [
        f"TODAY: {today.isoformat()}",
        f"HORIZON_END (3-6mo): {horizon_end.isoformat()}",
        f"Ticker: {ticker}",
        f"Current price: {current_price}",
        f"Default weights: {AGENT_WEIGHTS}",
        f"Deterministic weighted_score (using defaults): {weighted_score:.4f}",
        "",
        "REAL_CATALYSTS (from yfinance, treat as ground truth):",
        *catalyst_lines,
        "",
        "Agent verdicts:",
    ]
    for v in verdicts:
        lines.append(v.model_dump_json(indent=2))
    return "\n".join(lines)


def _mechanical_fallback(ticker: str, verdicts: list[AgentVerdict],
                          current_price: float,
                          weighted_score: float) -> OrchestratorVerdict:
    if weighted_score > 1.0:
        rec, sizing = "BUY", "starter (~1%)"
    elif weighted_score < -1.0:
        rec, sizing = "SELL", "not sizeable"
    else:
        rec, sizing = "HOLD", "not sizeable"
    targets = [(v.price_target_6mo, v.conviction) for v in verdicts
               if v.price_target_6mo and not v.error_note]
    tgt = None
    if targets:
        num = sum(t * c for t, c in targets)
        den = sum(c for _, c in targets)
        tgt = num / den if den else None
    return OrchestratorVerdict(
        ticker=ticker,
        run_timestamp=datetime.utcnow(),
        current_price=current_price,
        final_recommendation=rec,
        conviction=min(10, max(1, int(abs(weighted_score) + 1))),
        price_target_6mo=tgt,
        weighted_score=weighted_score,
        weights_used=AGENT_WEIGHTS,
        weight_overrides_rationale=None,
        synthesis=("LLM synthesis unavailable; mechanical aggregation below. "
                   f"Deterministic weighted score = {weighted_score:.2f} "
                   f"on default weights. Recommendation derived from score sign."),
        key_agreements=[],
        key_disagreements=[],
        dominant_drivers=[],
        red_flags=[],
        position_sizing_suggestion=sizing,
        stop_loss_level=None,
        catalyst_calendar=[],
        agent_verdicts=verdicts,
    )


def _filter_catalysts(events: list[CatalystEvent], today: date,
                       horizon_days: int = _HORIZON_DAYS) -> list[CatalystEvent]:
    """Drop catalysts whose date is in the past or beyond the horizon.

    Models will occasionally write plausible-looking but fabricated dates
    (e.g., from training-data earnings cycles). Out-of-window entries are
    discarded rather than corrected; if no real upcoming catalysts exist,
    `catalyst_calendar` should simply be empty rather than misleading.
    """
    cutoff = today + timedelta(days=horizon_days)
    return [e for e in events if today <= e.date <= cutoff]


def _stamp_deterministic(v: OrchestratorVerdict, ticker: str,
                          verdicts: list[AgentVerdict], current_price: float,
                          weighted: float, today: date,
                          real_catalysts: list[dict]) -> OrchestratorVerdict:
    # The LLM is asked to fill the entire schema and may hallucinate or
    # paraphrase pass-through values. Overwrite the deterministic fields with
    # ground truth from the runner.
    llm_catalysts = _filter_catalysts(v.catalyst_calendar, today)

    # Ensure every REAL_CATALYSTS entry is present even if the LLM dropped it.
    real_keys = {(c["date"], c["event"]) for c in real_catalysts}
    have_keys = {(e.date.isoformat(), e.event) for e in llm_catalysts}
    for c in real_catalysts:
        if (c["date"], c["event"]) not in have_keys:
            llm_catalysts.append(CatalystEvent(
                event=c["event"], date=date.fromisoformat(c["date"]),
                expected_impact=c["expected_impact"], notes=c["notes"],
            ))
    llm_catalysts.sort(key=lambda e: e.date)

    return v.model_copy(update={
        "ticker": ticker,
        "run_timestamp": datetime.utcnow(),
        "current_price": current_price,
        "weighted_score": weighted,
        "weights_used": v.weights_used or AGENT_WEIGHTS,
        "agent_verdicts": verdicts,
        "catalyst_calendar": llm_catalysts,
    })


async def run(ticker: str, verdicts: list[AgentVerdict],
              current_price: float) -> OrchestratorVerdict:
    weighted = compute_weighted_score(verdicts, AGENT_WEIGHTS)
    today = date.today()
    real_catalysts = get_real_catalysts(ticker, today)
    try:
        result = await Runner.run(
            _agent,
            input=_format_input(ticker, verdicts, current_price, weighted,
                                today, real_catalysts),
        )
        return _stamp_deterministic(result.final_output, ticker, verdicts,
                                     current_price, weighted, today,
                                     real_catalysts)
    except Exception as e:
        log.exception("Orchestrator LLM failed; falling back to mechanical: %s", e)
        return _mechanical_fallback(ticker, verdicts, current_price, weighted)
