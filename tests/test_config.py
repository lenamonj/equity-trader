from equity_trader.config import AGENT_WEIGHTS, AGENT_MODELS, ORCHESTRATOR_MODEL


def test_weights_sum_to_100():
    assert abs(sum(AGENT_WEIGHTS.values()) - 100.0) < 1e-6


def test_all_seven_agents_have_models():
    expected = {"jpm_fundamental", "bridgewater_macro", "gs_technical",
                "citadel_quant", "renaissance_pattern", "de_shaw_options",
                "two_sigma_backtest"}
    assert set(AGENT_WEIGHTS.keys()) == expected
    assert set(AGENT_MODELS.keys()) == expected


def test_orchestrator_model_set():
    assert ORCHESTRATOR_MODEL.provider in {"openai", "groq", "deepseek", "gemini"}
    assert ORCHESTRATOR_MODEL.model
