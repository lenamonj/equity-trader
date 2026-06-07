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
    "renaissance_pattern": ModelSpec("openai",   "gpt-4o-mini"),
    "de_shaw_options":     ModelSpec("openai",   "gpt-4o-mini"),
    "two_sigma_backtest":  ModelSpec("openai",   "gpt-4o-mini"),
}

AGENT_WEIGHTS: dict[str, float] = {
    "jpm_fundamental":     25.0,
    "bridgewater_macro":   20.0,
    "gs_technical":        15.0,
    "citadel_quant":       15.0,
    "renaissance_pattern": 10.0,
    "de_shaw_options":     10.0,
    "two_sigma_backtest":   5.0,
}

ORCHESTRATOR_MODEL = ModelSpec("openai", "gpt-4o")
