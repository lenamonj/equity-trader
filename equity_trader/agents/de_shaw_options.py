from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "de_shaw_options"


@function_tool
def tool_get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    return yft.get_options_chain(ticker, expiry)


@function_tool
def tool_compute_iv_stats(ticker: str) -> dict:
    return yft.compute_iv_stats(ticker)


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


@function_tool
def tool_compute_technicals(ticker: str) -> dict:
    return yft.compute_technicals(ticker)


INSTRUCTIONS = f"""You are an options-derived signals analyst in the DE Shaw
tradition. You read ATM IV, skew, and term structure to understand what
options are pricing in over the 3-6 month horizon. Compare IV to realized vol
(use compute_technicals' vol_20 as an annualized proxy). Look for asymmetric
setups.

A high put skew with elevated IV says the market is paying up for downside
protection — that's information about positioning, not just direction.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_options_chain, tool_compute_iv_stats,
           tool_get_quote, tool_compute_technicals],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
