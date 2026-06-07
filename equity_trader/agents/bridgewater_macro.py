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


INSTRUCTIONS = f"""You are a macro strategist in the Bridgewater All-Weather
tradition. You classify the macro regime (growth, inflation, liquidity, risk
appetite) and apply it to the ticker via its sector and broad market
sensitivity.

For the 3-6 month horizon: where are rates, where is the curve, what does the
NFCI say about financial conditions, what does CPI YoY trajectory imply for
sector rotation. Map that read onto the ticker's sector.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
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
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
