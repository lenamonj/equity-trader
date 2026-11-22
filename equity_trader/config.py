from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "groq", "deepseek", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


AGENT_MODELS: dict[str, ModelSpec] = {
    "jpm_fundamental":     ModelSpec("openai",   "gpt-6-luna"),
    "bridgewater_macro":   ModelSpec("openai",   "gpt-6-luna"),
    "gs_technical":        ModelSpec("openai",   "gpt-6-luna"),
    "citadel_quant":       ModelSpec("openai",   "gpt-6-luna"),
    "renaissance_pattern": ModelSpec("openai",   "gpt-6-luna"),
    "de_shaw_options":     ModelSpec("openai",   "gpt-6-luna"),
    "two_sigma_backtest":  ModelSpec("openai",   "gpt-6-luna"),
}

# Weights total exactly 100. Fundamentals and macro lead; the backtest is a
# sanity check and carries the least.
AGENT_WEIGHTS: dict[str, float] = {
    "jpm_fundamental":     28.0,
    "bridgewater_macro":   20.0,
    "gs_technical":        13.0,
    "citadel_quant":       13.0,
    "de_shaw_options":     11.0,
    "renaissance_pattern":  9.0,
    "two_sigma_backtest":   6.0,
}

ORCHESTRATOR_MODEL = ModelSpec("openai", "gpt-6.1-sol")

# House and discipline shown for each desk, heaviest weight first.
AGENT_PERSONAS: dict[str, tuple[str, str]] = {
    "jpm_fundamental":     ("J.P. Morgan",              "Fundamental"),
    "bridgewater_macro":   ("Bridgewater Associates",   "All-Weather Macro"),
    "gs_technical":        ("Goldman Sachs",            "Technical"),
    "citadel_quant":       ("Citadel",                  "Quantitative"),
    "de_shaw_options":     ("D. E. Shaw",               "Options-Derived"),
    "renaissance_pattern": ("Renaissance Technologies", "Pattern Recognition"),
    "two_sigma_backtest":  ("Two Sigma",                "Backtest Sanity"),
}

AGENT_NAMES = list(AGENT_PERSONAS)

# Desks that no longer sit on the committee. Runs saved while they did still
# carry their verdicts, and the grading report names them.
# Jane Street (ETF and basket flow, 10%) was retired on 2026-11-22: free data
# has no creation or redemption flows, and what was left repeated the
# technical desk's relative-strength reading.
RETIRED_DESKS: dict[str, str] = {"jane_street_etf": "Jane Street"}
