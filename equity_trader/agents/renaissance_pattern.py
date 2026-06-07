from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "renaissance_pattern"


@function_tool
def tool_get_price_history(ticker: str, period: str = "2y") -> list[dict]:
    df = yft.get_price_history(ticker, period=period)
    out = df[["Close", "Volume"]].tail(252).reset_index()
    return out.assign(Date=lambda d: d["Date"].astype(str)).to_dict(orient="records")


@function_tool
def tool_compute_statistical_patterns(ticker: str) -> dict:
    return yft.compute_statistical_patterns(ticker)


INSTRUCTIONS = f"""You are a statistical pattern researcher in the
Renaissance tradition. You treat price as data, not narrative. Identify
seasonality, autocorrelation regime (trending vs mean-reverting), and any
anomalies that might persist over the 3-6 month horizon.

Be rigorous about not over-fitting. If a 'pattern' relies on a handful of
samples, say so and downgrade conviction.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_price_history, tool_compute_statistical_patterns],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
