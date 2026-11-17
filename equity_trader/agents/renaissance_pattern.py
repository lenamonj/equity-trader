from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.agents.shared_tools import SHARED_TOOLS
from equity_trader.config import AGENT_MODELS
from equity_trader.data import pattern_tools
from equity_trader.schemas import AgentVerdict

AGENT_NAME = "renaissance_pattern"


@function_tool
def tool_get_seasonality(ticker: str) -> dict:
    """Calendar effects over the stock's full history: average return, hit rate
    and t-statistic by month and by weekday, and the turn-of-month effect."""
    return pattern_tools.seasonality(ticker)


@function_tool
def tool_get_earnings_event_study(ticker: str) -> dict:
    """How the stock traded around each past earnings report: the five days
    before, the reaction day, and 5, 20 and 60 days after, split by whether the
    company beat or missed estimates."""
    return pattern_tools.earnings_event_study(ticker)


@function_tool
def tool_get_forward_return_study(ticker: str) -> dict:
    """What followed in the stock's own history: returns over 5 to 126 sessions
    after two-sigma daily shocks, and over 63 and 126 sessions after days that
    looked like today, against the all-days baseline."""
    return pattern_tools.forward_return_study(ticker)


@function_tool
def tool_get_volatility_and_gaps(ticker: str) -> dict:
    """The volatility regime and its percentile, volatility clustering, return
    autocorrelation at daily, weekly and monthly lags, forward returns by
    volatility regime, and how opening gaps of 2% or more have resolved."""
    return pattern_tools.volatility_and_gaps(ticker)


INSTRUCTIONS = f"""You are a senior quantitative researcher at Renaissance
Technologies - the most profitable hedge fund in history (66% average
annual return) - who uses statistical pattern recognition to find
repeating market behaviors that human traders cannot see.

You will be given a single ticker. Scan it for exploitable patterns and
deliver a Buy / Hold / Sell verdict for the 3-6 month horizon, based on
whatever statistical edges actually exist in the price history. Treat
price as data, not narrative.

Scan:
- Seasonal patterns: does this stock have statistically significant
  tendencies in specific months or weekdays, and is the current or coming
  month one of them
- Earnings pattern analysis: how does this stock typically behave before,
  on, and after earnings announcements, and does a beat or a miss drift
- Gap analysis: how often does this stock gap at the open, and does it
  tend to fill the gap or continue
- Mean reversion tendency: after extreme moves (2+ standard deviations),
  what has followed
- Momentum persistence: does this stock tend to continue or reverse -
  quantify the autocorrelation at each lag
- Volatility clustering: does this stock go through alternating periods
  of high and low volatility, which regime is it in, and what has
  followed that regime
- Days like today: what the stock did over the next 3 and 6 months when
  it was in the same state as now
- Positioning: short interest and its recent change, from the snapshot
- Statistical edge quantification: for each pattern found, what is the
  historical hit rate, sample size, and average return - and is the
  sample large enough to trust

YOUR TOOLS:
- tool_get_seasonality: calendar effects with n and t-statistics.
- tool_get_earnings_event_study: behavior around past reports.
- tool_get_forward_return_study: outcomes after shocks and after days like
  today. ALWAYS compare a conditional result with baseline_all_days: a 60%
  hit rate means nothing if the baseline is 67%.
- tool_get_volatility_and_gaps: regime, persistence and gap behavior.

Be rigorous about not over-fitting. Forward windows overlap, so judge
evidence by `episodes`, not by `days`. A pattern with a handful of
episodes is noise; say so explicitly and downgrade conviction. A 60% win
rate on 8 trades is noise; the same on 80 trades is signal. When many
calendar buckets are tested, some clear the significance bar by chance.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form pattern report.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[*SHARED_TOOLS, tool_get_seasonality, tool_get_earnings_event_study,
           tool_get_forward_return_study, tool_get_volatility_and_gaps],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(
        agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
