from agents import Runner

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.agents.shared_data import SHARED_DATA, estimates
from equity_trader.config import AGENT_MODELS
from equity_trader.data import company_tools, market_tools
from equity_trader.gatherer import brief
from equity_trader.schemas import AgentVerdict

AGENT_NAME = "citadel_quant"


def factor_scorecard(ticker: str) -> dict:
    """Value, quality, growth, momentum and low-volatility metrics for the
    ticker and its five largest industry peers. Each metric has the ticker's
    value, the peer median and its rank, where rank 1 is the most attractive."""
    return company_tools.get_factor_scorecard(ticker)


def risk_stats(ticker: str) -> dict:
    """One-year return, volatility, drawdown, skew, beta and correlation to the
    S&P 500, and how much of the stock's risk is specific to it."""
    return company_tools.get_risk_stats(ticker)


def style_board() -> dict:
    """Momentum, quality, value, low-volatility and small-cap ETFs against the
    S&P 500 over 1, 3 and 6 months: which styles the market is paying for."""
    return market_tools.style_board()


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
  value, quality, low-vol, estimate revisions, rate beta)
- Factor exposures: momentum, value, quality, growth and low volatility,
  each as a rank against peers
- Peer relative value: where the ticker ranks on each factor, and whether
  the ranks agree or conflict
- Estimate revisions and surprise: are estimates rising or falling, and
  has the company been beating them
- Factor regime: which styles the market is paying for now, and whether
  this stock's profile matches them
- Entry signal: what set of conditions are TRUE today that justify
  initiating a position (cite specific factor scores, not narrative)
- Position sizing: how large would a systematic book size this given the
  realized volatility and stock-specific risk
- Time frame fit: does the 3-6 month horizon match the half-life of the
  signal, or is the edge faster/slower than that window
- Risk-reward ratio: what is the upside vs the downside, and is the
  reward-to-risk above the minimum (typically 2:1)
- Correlation check: does this trade add diversification or just amplify
  whatever the S&P 500 is already doing
- Market regime filter: bull / bear / sideways / risk-off - does the
  strategy survive across regimes, and which regime are we in now

YOUR DATA:
- `factor_scorecard`: the ticker and its five largest industry peers
  on every factor, with ranks. Rank 1 of N is the most attractive. A metric
  with `of` below 4 has too few peers to rank; say so.
- `estimates`: 90-day estimate change, revision counts and surprises.
- `style_board`: which styles are being paid.
- `risk_stats`: volatility, drawdown, beta and stock-specific risk.
- `macro_dashboard`: the regime filter.

Cite concrete factor scores; never say "strong momentum" without the
number and its rank.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form strategy doc.
"""

# What this desk reads. The gatherer fetches it; `run` hands it over.
DATA = [*SHARED_DATA, factor_scorecard, estimates, style_board, risk_stats]

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str, gathered: dict[str, dict]) -> AgentVerdict:
    result = await Runner.run(agent, input=brief(ticker, DATA, gathered))
    return result.final_output
