from agents import Runner

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.agents.shared_data import SHARED_DATA, index_event_calendar
from equity_trader.config import AGENT_MODELS
from equity_trader.data import options_tools
from equity_trader.gatherer import brief
from equity_trader.schemas import AgentVerdict

AGENT_NAME = "de_shaw_options"


def options_surface(ticker: str) -> dict:
    """Up to six expiries from one week to one year. For each: at-the-money
    implied volatility, the implied move, skew between the 90% put and the 110%
    call, put/call ratios by volume and open interest, the largest open-interest
    strikes, and whether it spans the next earnings report. Also unusual volume."""
    return options_tools.options_surface(ticker)


def volatility_premium(ticker: str) -> dict:
    """Implied volatility at one and three months against realized volatility
    over 20, 60 and 120 days, and the implied move over the next earnings
    report against the average move on past reports."""
    return options_tools.volatility_premium(ticker)


def options_chain(ticker: str) -> dict:
    """Prices, implied volatility, open interest and volume for the expiry
    inside the 3-6 month horizon, for strikes within 20% of the stock price."""
    return options_tools.options_chain_near_money(ticker)


def market_volatility() -> dict:
    """VIX at 9 days, 30 days and 3 months, the SKEW index and the MOVE index:
    level, one-month change and one-year percentile, and the VIX term structure."""
    return options_tools.market_volatility()


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
- Implied vs realized vol: is IV above realized (overpriced - sellers
  edge) or below realized (underpriced - buyers edge)
- Skew read: is downside protection bid (positive skew - defensive
  positioning) or is upside being chased (negative skew)
- Term structure: near-dated against longer-dated IV - is the market
  pricing a near-term event or long-tail uncertainty
- Event pricing: the implied move over the next earnings report against
  what the stock has actually done on past reports
- Positioning: put/call ratios, the strikes holding the most open
  interest, and any unusual volume
- Expected range: the implied move at the 3 and 6 month expiries is the
  range the market prices; state it in dollars from the current price
- Maximum loss scenario: what is the worst-case for someone long the
  underlying given the options-implied tail
- Asymmetry assessment: does the risk-reward favor longs (skew shows
  fear, IV is cheap to realized) or shorts (skew shows complacency,
  IV is rich)
- Market backdrop: is this stock's volatility its own story or part of a
  market-wide volatility regime

YOUR DATA:
- `options_surface`: the term structure, skew and positioning.
- `volatility_premium`: implied against realized, and earnings.
- `market_volatility`: index volatility and its term structure.
- `options_chain`: prices near the money for the horizon expiry,
  for sizing a structure.
- `index_event_calendar`: options expiration dates in the horizon.

You have no option Greeks and no history of implied volatility, so do not
quote a delta, a gamma or an IV rank. Quotes are delayed; treat a single
strike's IV with caution and rely on the at-the-money and 90/110 readings.

A high put skew with elevated IV says the market is paying up for
downside protection - that's positioning information, not just direction.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form options memo.
"""

# What this desk reads. The gatherer fetches it; `run` hands it over.
DATA = [*SHARED_DATA, options_surface, volatility_premium, market_volatility, options_chain,
        index_event_calendar]

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str, gathered: dict[str, dict]) -> AgentVerdict:
    result = await Runner.run(agent, input=brief(ticker, DATA, gathered))
    return result.final_output
