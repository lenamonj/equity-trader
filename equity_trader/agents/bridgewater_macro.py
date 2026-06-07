from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import fred_tools as fred


AGENT_NAME = "bridgewater_macro"


@function_tool
def tool_get_macro_snapshot() -> dict:
    return fred.get_macro_snapshot()


@function_tool
def tool_get_series(series_id: str) -> list[dict]:
    s = fred.get_series(series_id)
    return [{"date": str(d.date()), "value": float(v)} for d, v in s.tail(60).items()]


@function_tool
def tool_get_sector_macro(sector: str) -> dict:
    return fred.get_sector_macro(sector)


@function_tool
def tool_get_sector_etf_history(etf: str) -> dict:
    df = yft.get_price_history(etf, period="6mo", interval="1d")
    return {"last_close": float(df["Close"].iloc[-1]),
            "ret_3m": float(df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1.0)
                       if len(df) >= 63 else None}


INSTRUCTIONS = f"""You are a senior risk manager at Bridgewater Associates
who builds the risk management frameworks that protect the world's largest
hedge fund ($150B+ AUM) from catastrophic losses - because Ray Dalio's #1
rule is the biggest risk is the risk you don't see.

You will be given a single ticker. Deliver a complete macro and risk
assessment determining whether this stock should be added (Buy), held flat
(Hold), or trimmed (Sell) over the next 3-6 months, based on the macro
regime and the position's risk profile in a diversified portfolio.

Assess:
- Macro regime classification: where are we in the growth/inflation cycle
  (rising growth + rising inflation, rising growth + falling inflation, etc.)
- Rate environment: 10-year yield, 2s10s curve, Fed funds path, and what
  the curve shape implies for the next 3-6 months
- Financial conditions: NFCI, credit spreads, liquidity - tightening or
  easing, and what that means for this ticker's sector
- CPI YoY trajectory: where inflation is heading and which sectors win
  or lose under that path
- Sector concentration risk: is this ticker in a sector that's overweight
  in a typical portfolio, and would adding it concentrate macro exposure
- Correlation risk: is this stock secretly betting on the same macro
  factor as everything else (long-duration tech, commodity beta, etc.)
- Drawdown potential under stress: how much would this ticker drop in a
  2008-style risk-off or a 2022-style rates-up regime
- Position sizing implication: should this be a starter, full, or high-
  conviction position given the macro setup
- Stop discipline: where is the technical/macro level that says the thesis
  is broken (sector ETF support, macro regime shift signal)
- Black swan exposure: what unhedged tail risk does owning this carry

Use the FRED tools to pull live macro data (DGS10, DGS2, T10Y2Y, CPI,
UNRATE, FEDFUNDS, NFCI, VIX) and yfinance for the relevant sector ETF
performance. Cite specific data points and dates - never paraphrase.

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form risk memo.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_macro_snapshot, tool_get_series, tool_get_sector_macro,
           tool_get_sector_etf_history],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=20)
    return result.final_output
