from equity_trader.schemas import AgentVerdict

_SIGN = {"BUY": 1, "HOLD": 0, "SELL": -1}


def redistribute_weights(weights: dict[str, float],
                         excluded: set[str]) -> dict[str, float]:
    kept = {k: v for k, v in weights.items() if k not in excluded}
    total = sum(kept.values())
    if total <= 0:
        return kept
    factor = 100.0 / total
    return {k: v * factor for k, v in kept.items()}


def compute_weighted_score(verdicts: list[AgentVerdict],
                            weights: dict[str, float]) -> float:
    excluded = {v.agent for v in verdicts if v.error_note}
    effective = redistribute_weights(weights, excluded)
    score = 0.0
    for v in verdicts:
        if v.error_note:
            continue
        w = effective.get(v.agent, 0.0)
        score += _SIGN[v.recommendation] * v.conviction * w
    return score / 100.0


def _targets(verdicts: list[AgentVerdict]) -> list[tuple[float, int]]:
    """The target and conviction of each desk that reported and gave a target."""
    return [(v.price_target_6mo, v.conviction) for v in verdicts
            if v.price_target_6mo and not v.error_note]


def target_desks(verdicts: list[AgentVerdict]) -> int:
    """How many desks the price target rests on."""
    return len(_targets(verdicts))


def conviction_weighted_target(verdicts: list[AgentVerdict]) -> float | None:
    """Average of agent price targets, weighted by conviction.

    Agents without a target, or that errored, are excluded.
    """
    targets = _targets(verdicts)
    if not targets:
        return None
    return sum(t * c for t, c in targets) / sum(c for _, c in targets)
