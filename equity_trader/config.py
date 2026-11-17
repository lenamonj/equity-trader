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
    "jane_street_etf":     ModelSpec("openai",   "gpt-6-luna"),
    "renaissance_pattern": ModelSpec("openai",   "gpt-6-luna"),
    "de_shaw_options":     ModelSpec("openai",   "gpt-6-luna"),
    "two_sigma_backtest":  ModelSpec("openai",   "gpt-6-luna"),
}

# Weights total exactly 100. Jane Street takes 10 by trimming the agents
# whose discipline most overlaps with ETF flow / sector rotation: Citadel
# (cross-sectional) -3, GS Technical (relative strength) -3, Bridgewater
# (macro regime partially captured by sector ETFs) -2, Renaissance
# (statistical signals) -2.
AGENT_WEIGHTS: dict[str, float] = {
    "jpm_fundamental":     25.0,
    "bridgewater_macro":   18.0,
    "gs_technical":        12.0,
    "citadel_quant":       12.0,
    "jane_street_etf":     10.0,
    "de_shaw_options":     10.0,
    "renaissance_pattern":  8.0,
    "two_sigma_backtest":   5.0,
}

ORCHESTRATOR_MODEL = ModelSpec("openai", "gpt-6.1-sol")

# House and discipline shown for each desk, heaviest weight first.
AGENT_PERSONAS: dict[str, tuple[str, str]] = {
    "jpm_fundamental":     ("J.P. Morgan",              "Fundamental"),
    "bridgewater_macro":   ("Bridgewater Associates",   "All-Weather Macro"),
    "gs_technical":        ("Goldman Sachs",            "Technical"),
    "citadel_quant":       ("Citadel",                  "Quantitative"),
    "jane_street_etf":     ("Jane Street",              "ETF & Basket Flow"),
    "de_shaw_options":     ("D. E. Shaw",               "Options-Derived"),
    "renaissance_pattern": ("Renaissance Technologies", "Pattern Recognition"),
    "two_sigma_backtest":  ("Two Sigma",                "Backtest Sanity"),
}

AGENT_NAMES = list(AGENT_PERSONAS)
