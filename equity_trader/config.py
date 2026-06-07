from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "groq", "deepseek", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


AGENT_MODELS: dict[str, ModelSpec] = {
    "jpm_fundamental":     ModelSpec("openai",   "gpt-4o-mini"),
    "bridgewater_macro":   ModelSpec("openai",   "gpt-4o-mini"),
    "gs_technical":        ModelSpec("openai",   "gpt-4o-mini"),
    "citadel_quant":       ModelSpec("openai",   "gpt-4o-mini"),
    "jane_street_etf":     ModelSpec("openai",   "gpt-4o-mini"),
    "renaissance_pattern": ModelSpec("openai",   "gpt-4o-mini"),
    "de_shaw_options":     ModelSpec("openai",   "gpt-4o-mini"),
    "two_sigma_backtest":  ModelSpec("openai",   "gpt-4o-mini"),
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

ORCHESTRATOR_MODEL = ModelSpec("openai", "gpt-4o")
