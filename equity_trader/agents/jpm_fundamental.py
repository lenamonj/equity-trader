from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import edgar_tools as edg


AGENT_NAME = "jpm_fundamental"


@function_tool
def tool_get_fundamentals(ticker: str) -> dict:
    return yft.get_fundamentals(ticker)


@function_tool
def tool_get_analyst_targets(ticker: str) -> dict:
    return yft.get_analyst_targets(ticker)


@function_tool
def tool_get_latest_10k(ticker: str) -> dict:
    return edg.get_latest_10k(ticker)


@function_tool
def tool_get_latest_10q(ticker: str) -> dict:
    return edg.get_latest_10q(ticker)


@function_tool
def tool_get_recent_8k_summaries(ticker: str, n: int = 5) -> list[dict]:
    return edg.get_recent_8k_summaries(ticker, n)


INSTRUCTIONS = f"""You are a senior equity research analyst in the style of
JPM fundamental coverage. You analyze companies on revenue and FCF trajectory,
margin trends, balance sheet quality, capital allocation, and the credibility
of management guidance.

Use the EDGAR tools to read the latest 10-K and 10-Q. Use yfinance tools for
analyst targets and recent fundamentals. Form a view on the 3-6 month window
specifically - what catalysts hit in that window, what does the next quarter
look like, where does consensus sit vs your read.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_fundamentals, tool_get_analyst_targets,
           tool_get_latest_10k, tool_get_latest_10q,
           tool_get_recent_8k_summaries],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
