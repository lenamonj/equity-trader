from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.agents.shared_tools import SHARED_TOOLS, tool_get_relative_strength
from equity_trader.config import AGENT_MODELS
from equity_trader.data import technical_tools
from equity_trader.data import yfinance_tools as yft
from equity_trader.schemas import AgentVerdict

AGENT_NAME = "gs_technical"


@function_tool
def tool_get_price_history(ticker: str) -> list[dict]:
    """The last 120 daily bars (open, high, low, close, volume) for reading
    chart patterns."""
    df = yft.get_price_history(ticker, period="2y")
    out = df[["Open", "High", "Low", "Close", "Volume"]].tail(120).round(2).reset_index()
    return out.assign(Date=lambda d: d["Date"].astype(str).str[:10]).to_dict(orient="records")


@function_tool
def tool_compute_technicals(ticker: str) -> dict:
    """Trend and momentum on daily, weekly and monthly bars: 20, 50, 100 and
    200-day averages with slopes, the last 50/200 cross, RSI, MACD, Bollinger
    bands, ATR, ADX, volatility and drawdown from the 52-week high."""
    return technical_tools.compute_technicals(ticker)


@function_tool
def tool_get_key_levels(ticker: str) -> dict:
    """Support and resistance from the last year: swing highs and lows, 52-week
    and all-time highs, the heaviest volume zones and unfilled gaps, each with
    its distance from the current price in percent and in ATRs."""
    return technical_tools.key_levels(ticker)


@function_tool
def tool_get_volume_analysis(ticker: str) -> dict:
    """Whether volume confirms the trend: up against down volume, on-balance
    volume, accumulation and distribution days, and the heaviest recent days."""
    return technical_tools.volume_analysis(ticker)


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
  (trend and momentum), ADX (trend strength, above 25 is a strong trend) and
  where they sit right now
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

YOUR TOOLS:
- tool_compute_technicals: every indicator above, on three timeframes.
- tool_get_key_levels: computed support, resistance, volume zones and gaps.
  Take your levels from here. A broken swing low that now sits above the
  price acts as resistance.
- tool_get_volume_analysis: the volume evidence.
- tool_get_relative_strength: the stock against the market and its sector.
- tool_get_price_history: the last 120 daily bars, for chart patterns only.

Cite the actual numbers, not vague "above the 50dma."

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form technical note.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[*SHARED_TOOLS, tool_compute_technicals, tool_get_key_levels,
           tool_get_volume_analysis, tool_get_relative_strength, tool_get_price_history],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(
        agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
