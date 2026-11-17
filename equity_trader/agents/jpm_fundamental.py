import logging

from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.agents.shared_tools import SHARED_TOOLS, tool_get_estimates
from equity_trader.config import AGENT_MODELS
from equity_trader.data import company_tools
from equity_trader.data import edgar_tools as edg
from equity_trader.data import yfinance_tools as yft
from equity_trader.exceptions import DataFetchError
from equity_trader.schemas import AgentVerdict

log = logging.getLogger(__name__)

AGENT_NAME = "jpm_fundamental"


def get_fundamentals(ticker: str) -> dict:
    """Statements as filed with the SEC, plus market valuation multiples.

    Companies with no US-GAAP statements on EDGAR (IFRS filers, funds) fall
    back to yfinance statements, and the `source` field says so.
    """
    try:
        statements = edg.get_financial_statements(ticker)
    except DataFetchError as exc:
        log.warning("SEC statements unavailable for %s, using yfinance: %s", ticker, exc)
        return {"source": "yfinance (no US-GAAP statements on SEC EDGAR)",
                **yft.get_fundamentals(ticker)}
    return {**statements, "valuation": yft.get_valuation(ticker)}


@function_tool
def tool_get_fundamentals(ticker: str) -> dict:
    """Financial statements as filed with the SEC (five fiscal years, six
    quarters) with margins, growth and coverage ratios computed, a growth and
    trailing-twelve-month summary, and current valuation multiples."""
    return get_fundamentals(ticker)


@function_tool
def tool_get_earnings_release(ticker: str) -> dict:
    """The company's latest earnings release and commentary from its 8-K: the
    opening of each exhibit and its outlook section, where guidance is stated."""
    return edg.get_earnings_release(ticker)


@function_tool
def tool_get_analyst_view(ticker: str) -> dict:
    """Analyst price targets, the buy/hold/sell mix over three months, and every
    upgrade, downgrade and target change in the last 90 days."""
    return company_tools.get_analyst_view(ticker)


@function_tool
def tool_get_insider_activity(ticker: str) -> dict:
    """Open-market insider sales and purchases over six months, the ten most
    recent insider transactions, and the top institutional holders."""
    return company_tools.get_insider_activity(ticker)


@function_tool
def tool_get_recent_8ks(ticker: str) -> list[dict]:
    """The five most recent 8-K current reports: filing date, named items and
    the text of each filing."""
    return edg.get_recent_8ks(ticker)


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
- Guidance against expectations: what management guided in the latest
  release, how that compares with analyst consensus, and which way estimates
  have been revised
- Earnings record: how often the company has beaten estimates and by how much
- Competitive moat: what prevents competitors from copying this business
  model (patents, network effects, switching costs, brand)
- Management and insiders: are executives allocating capital wisely, have
  they delivered on previous promises, and are insiders buying or selling
- Valuation analysis: P/E, P/B, EV/EBITDA, PEG - is the stock cheap, fairly
  valued, or expensive relative to growth
- Catalyst identification: what could drive earnings, revenue, product
  launches, or regulatory decisions inside the 3-6 month window
- Risk assessment: the biggest risks that could permanently impair the
  business model (disruption, regulation, key person dependency)

YOUR TOOLS:
- tool_get_fundamentals: statements as filed with the SEC. Every period is
  labelled (for example 'Q2 FY2027') with its period end, form and filing
  date. quarterly[0] is the latest quarter. Margins, growth rates and
  coverage are already computed in `ratios`, and `summary` has 3 and 5 year
  revenue growth and trailing-twelve-month totals: use them, do not
  recompute. A null line item means the company does not report it; do not
  estimate it. The `source` field names where the statements came from;
  cite it.
- tool_get_earnings_release: management's own words on the latest quarter
  and its outlook. This is the source for guidance.
- tool_get_estimates: consensus, revisions and the surprise record. Compare
  guidance with consensus.
- tool_get_analyst_view: targets and recent rating actions. A cluster of
  downgrades or target cuts on one date marks an event; find it in the 8-Ks.
- tool_get_insider_activity: insider selling and buying, and top holders.
- tool_get_recent_8ks: the company's five most recent current reports, with
  the text of each filing. Read every one. These are the events the
  statements do not show yet: executive or director departures and
  appointments, guidance changes, acquisitions and disposals, new debt,
  impairments, restatements, auditor changes. Any such event MUST appear in
  your thesis or your risks with its filing date, and must be weighed in
  your recommendation and conviction. A routine earnings release needs no
  mention.

Cite specific numbers and the period they belong to - never paraphrase from
memory.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form research note.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[*SHARED_TOOLS, tool_get_fundamentals, tool_get_earnings_release,
           tool_get_estimates, tool_get_analyst_view, tool_get_insider_activity,
           tool_get_recent_8ks],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(
        agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
