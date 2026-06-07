from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "groq", "deepseek", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


AGENT_MODELS: dict[str, ModelSpec] = {
    "jpm_fundamental":     ModelSpec("gemini",   "gemini-2.5-pro"),
    "bridgewater_macro":   ModelSpec("gemini",   "gemini-2.5-pro"),
    "gs_technical":        ModelSpec("deepseek", "deepseek-chat"),
    "citadel_quant":       ModelSpec("groq",     "llama-3.3-70b-versatile"),
    "renaissance_pattern": ModelSpec("groq",     "llama-3.3-70b-versatile"),
    "de_shaw_options":     ModelSpec("deepseek", "deepseek-chat"),
    "two_sigma_backtest":  ModelSpec("groq",     "llama-3.3-70b-versatile"),
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
