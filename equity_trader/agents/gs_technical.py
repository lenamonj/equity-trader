from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "gs_technical"


@function_tool
def tool_get_price_history(ticker: str, period: str = "2y") -> list[dict]:
    df = yft.get_price_history(ticker, period=period)
    out = df[["Open", "High", "Low", "Close", "Volume"]].tail(120).reset_index()
    return out.assign(Date=lambda d: d["Date"].astype(str)).to_dict(orient="records")


@function_tool
def tool_compute_technicals(ticker: str) -> dict:
    return yft.compute_technicals(ticker)


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


INSTRUCTIONS = f"""You are a technical analyst in the GS tradition. You read
trend, momentum, support/resistance, and volume. You explicitly call out the
3-6 month setup: is the stock in a base, breaking out, rolling over, or
chopping. Cite concrete levels (50dma, 200dma, recent swing high/low) and
specific indicator values (RSI, MACD position vs signal).

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_price_history, tool_compute_technicals, tool_get_quote],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
