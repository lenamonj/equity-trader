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
