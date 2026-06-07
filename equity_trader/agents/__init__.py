from equity_trader.agents import (
    jpm_fundamental, bridgewater_macro, gs_technical,
    citadel_quant, renaissance_pattern, de_shaw_options,
    two_sigma_backtest,
)

ALL_AGENTS = [
    ("jpm_fundamental",     jpm_fundamental.run),
    ("bridgewater_macro",   bridgewater_macro.run),
    ("gs_technical",        gs_technical.run),
    ("citadel_quant",       citadel_quant.run),
    ("renaissance_pattern", renaissance_pattern.run),
    ("de_shaw_options",     de_shaw_options.run),
    ("two_sigma_backtest",  two_sigma_backtest.run),
]

AGENT_NAMES = [name for name, _ in ALL_AGENTS]
