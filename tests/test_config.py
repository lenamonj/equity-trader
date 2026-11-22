from equity_trader.config import (
    AGENT_MODELS,
    AGENT_PERSONAS,
    AGENT_WEIGHTS,
    ORCHESTRATOR_MODEL,
    RETIRED_DESKS,
)


def test_weights_sum_to_100():
    assert abs(sum(AGENT_WEIGHTS.values()) - 100.0) < 1e-6


def test_the_seven_desks_and_their_weights():
    expected = {"jpm_fundamental": 28.0, "bridgewater_macro": 20.0, "gs_technical": 13.0,
                "citadel_quant": 13.0, "de_shaw_options": 11.0, "renaissance_pattern": 9.0,
                "two_sigma_backtest": 6.0}
    assert AGENT_WEIGHTS == expected
    assert set(AGENT_MODELS) == set(expected) == set(AGENT_PERSONAS)


def test_a_retired_desk_keeps_its_name_for_old_runs():
    assert RETIRED_DESKS == {"jane_street_etf": "Jane Street"}
    assert not set(RETIRED_DESKS) & set(AGENT_WEIGHTS)


def test_orchestrator_model_set():
    assert ORCHESTRATOR_MODEL.provider in {"openai", "groq", "deepseek", "gemini"}
    assert ORCHESTRATOR_MODEL.model
