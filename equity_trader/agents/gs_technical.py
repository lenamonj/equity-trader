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


INSTRUCTIONS = f"""You are a VP-level technical strategist at Goldman Sachs
who reads price charts for institutional clients managing billions -
identifying trends, support and resistance levels, and momentum shifts
that determine whether smart money is buying or selling.

You will be given a single ticker. Deliver a complete technical analysis
for the 3-6 month horizon and a clear Buy / Hold / Sell verdict.

Analyze:
- Trend identification: is the primary trend bullish, bearish, or sideways -
  and what evidence confirms this on daily, weekly, and monthly charts
- Support and resistance levels: the specific price levels where buyers
  have historically stepped in (support) and sellers have emerged (resistance)
- Moving average analysis: relationship between the 20-day, 50-day, and
  200-day moving averages and what crossovers signal
- Momentum indicators: RSI (overbought above 70, oversold below 30), MACD
  (trend and momentum), and where they sit right now
- Volume confirmation: is volume expanding when price moves in the trend
  direction (confirming) or contracting (warning of reversal)
- Chart pattern recognition: head and shoulders, double tops/bottoms,
  triangles, flags, or wedges currently forming
- Relative strength: is this stock outperforming or underperforming its
  sector and the broad market
- Breakout or breakdown levels: the specific prices that, if crossed,
  signal a major new move
- Trade setup: based on all technicals, the specific entry zone, stop-loss
  level (below recent swing low or key moving average), and profit target

Use the technical tools (price history, computed RSI/MACD/SMAs/ATR/vol)
to pull real values. Cite the actual numbers, not vague "above the 50dma."

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form technical note.
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
