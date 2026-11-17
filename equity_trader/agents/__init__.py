from equity_trader.agents import (
    bridgewater_macro,
    citadel_quant,
    de_shaw_options,
    gs_technical,
    jane_street_etf,
    jpm_fundamental,
    renaissance_pattern,
    two_sigma_backtest,
)
from equity_trader.config import AGENT_NAMES  # noqa: F401  (re-exported)

# Order matches default-weight ranking, heaviest first.
ALL_AGENTS = [
    ("jpm_fundamental",     jpm_fundamental.run),
    ("bridgewater_macro",   bridgewater_macro.run),
    ("gs_technical",        gs_technical.run),
    ("citadel_quant",       citadel_quant.run),
    ("jane_street_etf",     jane_street_etf.run),
    ("de_shaw_options",     de_shaw_options.run),
    ("renaissance_pattern", renaissance_pattern.run),
    ("two_sigma_backtest",  two_sigma_backtest.run),
]
