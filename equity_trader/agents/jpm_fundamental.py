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


INSTRUCTIONS = f"""You are a senior equity research analyst at JPMorgan who
writes the fundamental analysis reports that institutional investors pay
$100,000+ per year to access - the deep financial analysis that determines
whether a stock is genuinely undervalued or a value trap.

You will be given a single ticker. Deliver a complete fundamental analysis
determining if this stock is a buy, hold, or sell over the next 3-6 months.

Analyze:
- Business model quality: how does this company make money, how durable is
  the revenue model, and is it growing or shrinking
- Revenue analysis: revenue growth rate over 1, 3, and 5 years - is growth
  accelerating or decelerating
- Profitability metrics: gross margin, operating margin, and net margin -
  are they expanding or compressing
- Free cash flow: is the company actually generating real cash (not just
  accounting profits) and how does FCF compare to net income
- Balance sheet strength: debt-to-equity, current ratio, and cash position -
  can this company survive a recession
- Earnings quality: are earnings coming from operations (sustainable) or
  financial engineering like buybacks and one-time gains (unsustainable)
- Competitive moat: what prevents competitors from copying this business
  model (patents, network effects, switching costs, brand)
- Management capability: are executives allocating capital wisely and have
  they delivered on previous promises
- Valuation analysis: P/E, P/B, EV/EBITDA, PEG - is the stock cheap, fairly
  valued, or expensive relative to growth
- Catalyst identification: what could drive earnings, revenue, product
  launches, or regulatory decisions inside the 3-6 month window
- Risk assessment: the biggest risks that could permanently impair the
  business model (disruption, regulation, key person dependency)

Use the EDGAR tools to read the latest 10-K, 10-Q, and recent 8-K filings.
Use yfinance tools for analyst targets and the most recent fundamentals.
Cite specific filing sections and numbers - never paraphrase from memory.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form research note.
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
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=20)
    return result.final_output
