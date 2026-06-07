from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import fred_tools as fred


AGENT_NAME = "citadel_quant"


@function_tool
def tool_get_peer_set(ticker: str) -> list[str]:
    return yft.get_peer_set(ticker)


@function_tool
def tool_compute_factor_loads(ticker: str, peers: list[str]) -> dict:
    return yft.compute_factor_loads(ticker, peers)


@function_tool
def tool_get_macro_snapshot() -> dict:
    return fred.get_macro_snapshot()


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


INSTRUCTIONS = f"""You are a quantitative equity analyst in the Citadel
tradition. You score the name on factor exposures (value, quality, momentum,
low-vol) and relative value vs peers. For the 3-6 month horizon, pay attention
to momentum (12-1), realized volatility regime, and rate sensitivity.

Use the peer set returned by tool_get_peer_set when computing relative metrics.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_peer_set, tool_compute_factor_loads,
           tool_get_macro_snapshot, tool_get_quote],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
