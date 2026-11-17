import importlib

import pytest

from equity_trader.agents import ALL_AGENTS
from equity_trader.agents.shared_tools import SHARED_TOOLS

SHARED = {tool.name for tool in SHARED_TOOLS}

# The tools that make each desk what it is.
DESK_TOOLS = {
    "jpm_fundamental": {"tool_get_fundamentals", "tool_get_earnings_release",
                        "tool_get_estimates", "tool_get_analyst_view",
                        "tool_get_insider_activity", "tool_get_recent_8ks"},
    "bridgewater_macro": {"tool_get_macro_sensitivity", "tool_get_stress_history",
                          "tool_get_cross_asset_board", "tool_get_sector_rotation"},
    "gs_technical": {"tool_compute_technicals", "tool_get_key_levels",
                     "tool_get_volume_analysis", "tool_get_relative_strength",
                     "tool_get_price_history"},
    "citadel_quant": {"tool_get_factor_scorecard", "tool_get_estimates",
                      "tool_get_style_board", "tool_get_risk_stats"},
    "jane_street_etf": {"tool_get_etf_landscape", "tool_get_etf_exposure",
                        "tool_get_relative_strength", "tool_get_sector_rotation",
                        "tool_get_index_event_calendar"},
    "de_shaw_options": {"tool_get_options_surface", "tool_get_volatility_premium",
                        "tool_get_market_volatility", "tool_get_options_chain",
                        "tool_get_index_event_calendar"},
    "renaissance_pattern": {"tool_get_seasonality", "tool_get_earnings_event_study",
                            "tool_get_forward_return_study", "tool_get_volatility_and_gaps"},
    "two_sigma_backtest": {"tool_backtest_rules"},
}


@pytest.mark.parametrize("name", [name for name, _ in ALL_AGENTS])
def test_every_desk_has_the_shared_context_and_its_own_tools(name):
    agent = importlib.import_module(f"equity_trader.agents.{name}").agent
    names = [tool.name for tool in agent.tools]
    assert len(names) == len(set(names))
    assert set(names) == SHARED | DESK_TOOLS[name]


@pytest.mark.parametrize("name", [name for name, _ in ALL_AGENTS])
def test_every_tool_a_prompt_names_is_one_the_desk_has(name):
    module = importlib.import_module(f"equity_trader.agents.{name}")
    have = {tool.name for tool in module.agent.tools}
    import re
    named = set(re.findall(r"tool_[a-z0-9_]+", module.INSTRUCTIONS))
    assert named <= have, named - have


def test_every_tool_describes_itself_to_the_model():
    for name, _ in ALL_AGENTS:
        for tool in importlib.import_module(f"equity_trader.agents.{name}").agent.tools:
            assert len(tool.description) > 40, tool.name
