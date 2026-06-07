import logging
from datetime import datetime
from agents import Runner

from equity_trader.agents.base import build_agent
from equity_trader.config import AGENT_WEIGHTS, ORCHESTRATOR_MODEL
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader.scoring import compute_weighted_score, redistribute_weights


log = logging.getLogger(__name__)


_INSTRUCTIONS = """You are the head of a multi-strategy investment committee.
Seven specialist agents have each analyzed the ticker. Your job is to
synthesize their structured verdicts into a final BUY / HOLD / SELL for the
3-6 month horizon.

You will be given:
- Each agent's full AgentVerdict (recommendation, conviction, target, thesis,
  risks, data_cited, error_note).
- The default weights for each agent.
- The deterministic weighted_score already computed from those weights.
- The current price of the ticker.

Hard rules:
1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.
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
9. `catalyst_calendar` lists events within the 3-6 month window with expected_impact.

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
                   current_price: float, weighted_score: float) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Current price: {current_price}",
        f"Default weights: {AGENT_WEIGHTS}",
        f"Deterministic weighted_score (using defaults): {weighted_score:.4f}",
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


def _stamp_deterministic(v: OrchestratorVerdict, ticker: str,
                          verdicts: list[AgentVerdict], current_price: float,
                          weighted: float) -> OrchestratorVerdict:
    # The LLM is asked to fill the entire schema and may hallucinate or
    # paraphrase pass-through values. Overwrite the deterministic fields with
    # ground truth from the runner.
    return v.model_copy(update={
        "ticker": ticker,
        "run_timestamp": datetime.utcnow(),
        "current_price": current_price,
        "weighted_score": weighted,
        "weights_used": v.weights_used or AGENT_WEIGHTS,
        "agent_verdicts": verdicts,
    })


async def run(ticker: str, verdicts: list[AgentVerdict],
              current_price: float) -> OrchestratorVerdict:
    weighted = compute_weighted_score(verdicts, AGENT_WEIGHTS)
    try:
        result = await Runner.run(_agent,
                                   input=_format_input(ticker, verdicts,
                                                       current_price, weighted))
        return _stamp_deterministic(result.final_output, ticker, verdicts,
                                     current_price, weighted)
    except Exception as e:
        log.exception("Orchestrator LLM failed; falling back to mechanical: %s", e)
        return _mechanical_fallback(ticker, verdicts, current_price, weighted)
