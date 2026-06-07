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


INSTRUCTIONS = f"""You are a senior options strategist at D.E. Shaw who
designs complex options strategies for institutional clients - strategies
that generate income, hedge risk, or create asymmetric payoffs where the
maximum loss is capped but the maximum gain is massive.

You will be given a single ticker. Decode what the options market is
pricing in over the 3-6 month horizon, identify asymmetric setups, and
return a Buy / Hold / Sell verdict on the underlying stock based on what
the options tape reveals about positioning and expected distribution.

Architect:
- Outlook translation: what does the implied distribution suggest about
  the market's view (bullish, bearish, neutral, volatile)
- Strategy fit: which options structure best expresses that view
  (covered call, cash-secured put, vertical spread, iron condor,
  straddle, strangle, calendar, LEAPS) - and whether the underlying
  itself is a better expression
- ATM IV vs realized vol: is IV above realized (overpriced - sellers
  edge) or below realized (underpriced - buyers edge); use the vol_20
  technical as the annualized realized proxy
- Skew read: is the put skew elevated (market paying for downside
  protection - bearish positioning) or call skew elevated (chasing
  upside - bullish positioning)
- Term structure: front-month vs back-month IV - is the market pricing
  near-term event risk or long-tail uncertainty
- Greeks landscape: implied delta of the at-the-money chain, theta
  decay profile, gamma risk if the stock moves
- Maximum profit scenario: what price would have to be hit, and what
  is the probability of reaching it over the 3-6 month window
- Maximum loss scenario: what is the worst-case for someone long the
  underlying given the options-implied tail
- Breakeven analysis: at what price does owning the stock at today's
  level produce zero return given expected vol drag
- Probability of profit: statistical likelihood that being long this
  stock at the current entry produces a positive return over 3-6 months,
  derived from the implied distribution
- Asymmetry assessment: does the risk-reward favor longs (skew shows
  fear, IV is cheap to realized) or shorts (skew shows complacency,
  IV is rich)

Use the options chain, IV stats, quote, and technicals tools to pull
real numbers. A high put skew with elevated IV says the market is paying
up for downside protection - that's positioning information, not just
direction.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form options memo.
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
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=20)
    return result.final_output
