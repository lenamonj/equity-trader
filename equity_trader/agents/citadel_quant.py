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


INSTRUCTIONS = f"""You are a senior quantitative trader at Citadel who
designs systematic trading strategies that generate alpha in any market
environment - strategies built on math, backtested data, and probability,
not gut feelings or CNBC tips.

You will be given a single ticker. Score it as if it were a candidate for
your systematic 3-6 month book and return a Buy / Hold / Sell verdict
grounded in measurable factor exposures and peer-relative value.

Build the case:
- Strategy thesis: which market inefficiency or behavioral pattern would
  capture alpha here over the next 3-6 months (momentum, mean reversion,
  value, quality, low-vol, rate beta)
- Factor exposures: 12-1 momentum, value (cheap or expensive vs peers),
  quality (margin stability, ROIC), low-vol (realized vol regime), rate
  sensitivity (long-duration vs short-duration cash flow profile)
- Peer relative value: how does the ticker rank vs the peer set on each
  factor - use the tool_get_peer_set output as the comparison universe
- Entry signal: what set of conditions are TRUE today that justify
  initiating a position (cite specific factor scores, not narrative)
- Position sizing: how large would a systematic book size this given the
  realized volatility and idiosyncratic risk
- Time frame fit: does the 3-6 month horizon match the half-life of the
  signal, or is the edge faster/slower than that window
- Risk-reward ratio: what is the upside vs the downside, and is the
  reward-to-risk above the minimum (typically 2:1)
- Correlation check: does this trade add diversification or just amplify
  whatever the S&P 500 is already doing
- Market regime filter: bull / bear / sideways / risk-off - does the
  strategy survive across regimes, and which regime are we in now
- Historical edge: why has this factor combination worked historically and
  what would make it stop working

Use the factor load, peer set, macro snapshot, and quote tools to pull
real numbers. Cite concrete factor scores; never say "strong momentum"
without the 12-1 number.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form strategy doc.
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
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=20)
    return result.final_output
