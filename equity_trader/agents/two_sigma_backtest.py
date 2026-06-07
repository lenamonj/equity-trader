import pandas as pd
import numpy as np
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "two_sigma_backtest"


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _summarize_trades(returns: pd.Series) -> dict:
    if returns.empty:
        return {"trades": 0, "hit_rate": None, "avg_return": None}
    return {
        "trades": int(len(returns)),
        "hit_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
    }


def backtest_simple_rules(ticker: str) -> dict:
    df = yft.get_price_history(ticker, period="2y")
    close = df["Close"]
    # RSI mean reversion: buy when RSI<30, hold 20 sessions
    rsi = _rsi_series(close)
    signals = rsi < 30
    rets = []
    for i in signals[signals].index:
        loc = close.index.get_loc(i)
        if loc + 20 < len(close):
            rets.append(close.iloc[loc + 20] / close.iloc[loc] - 1.0)
    rsi_summary = _summarize_trades(pd.Series(rets))

    # 50/200 SMA crossover: buy when 50dma crosses above 200dma, hold 60 sessions
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    cross = (sma50.shift() <= sma200.shift()) & (sma50 > sma200)
    rets2 = []
    for i in cross[cross].index:
        loc = close.index.get_loc(i)
        if loc + 60 < len(close):
            rets2.append(close.iloc[loc + 60] / close.iloc[loc] - 1.0)
    sma_summary = _summarize_trades(pd.Series(rets2))

    return {"rsi_mean_reversion": rsi_summary, "sma_crossover": sma_summary}


@function_tool
def tool_backtest_simple_rules(ticker: str) -> dict:
    return backtest_simple_rules(ticker)


INSTRUCTIONS = f"""You are a backtest-grounded analyst in the Two Sigma
tradition. You evaluate a small handful of simple rules on this ticker's
history and report what worked. You acknowledge that single-ticker backtests
are statistically weak - your role is a sanity check, not a primary signal.

If hit rates are 45-55%, treat as no edge. Only call BUY/SELL when a rule
shows a >60% hit rate over >=10 trades AND the most recent signal matches.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_backtest_simple_rules],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
