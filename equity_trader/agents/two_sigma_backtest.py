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


def _backtest_rule(close: pd.Series, signals: pd.Series, hold: int) -> dict:
    """Buy on each signal and hold `hold` sessions. Trades never overlap, so a
    run of consecutive signal days counts once, not once per day."""
    returns, next_free = [], 0
    for loc in np.flatnonzero(signals.to_numpy()):
        if loc >= next_free and loc + hold < len(close):
            returns.append(close.iloc[loc + hold] / close.iloc[loc] - 1.0)
            next_free = loc + hold
    fired = signals[signals]
    summary = {
        "trades": len(returns),
        "hit_rate": None,
        "avg_return": None,
        "worst_trade": None,
        # Mean return of every window of this length, signal or not: the bar
        # a rule has to clear to claim an edge.
        "baseline_avg_return": float((close.shift(-hold) / close - 1.0).mean()),
        "signal_active_now": bool(signals.iloc[-hold:].any()),
        "last_signal_date": fired.index[-1].date().isoformat() if len(fired) else None,
    }
    if returns:
        r = pd.Series(returns)
        summary.update(hit_rate=float((r > 0).mean()), avg_return=float(r.mean()),
                       worst_trade=float(r.min()))
    return summary


def backtest_simple_rules(ticker: str) -> dict:
    df = yft.get_price_history(ticker, period="5y")
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    return {
        "as_of": yft.as_of(df),
        "history_start": df.index[0].date().isoformat(),
        # Buy when RSI<30, hold 20 sessions
        "rsi_mean_reversion": _backtest_rule(close, _rsi_series(close) < 30, hold=20),
        # Buy when the 50dma crosses above the 200dma, hold 60 sessions
        "sma_crossover": _backtest_rule(
            close, (sma50.shift() <= sma200.shift()) & (sma50 > sma200), hold=60),
    }


@function_tool
def tool_backtest_simple_rules(ticker: str) -> dict:
    return backtest_simple_rules(ticker)


INSTRUCTIONS = f"""You are a senior quantitative researcher at Two Sigma
who backtests trading strategies against historical data - because any
strategy that hasn't been tested against real market history is just a
theory waiting to lose money.

You will be given a single ticker. Run a backtest of two simple,
well-known rule sets against the ticker's price history and report
whether either edge is real. Translate the results into a Buy / Hold /
Sell verdict for the 3-6 month horizon based on whether the most recent
signal aligns with a statistically credible historical edge.

Backtest:
- Strategy rules codification: the two rules you are testing are
  precise IF/THEN constructions (RSI mean reversion: buy when RSI<30,
  hold 20 sessions; SMA crossover: buy when 50dma crosses above 200dma,
  hold 60 sessions). Treat them as the system; do not invent new ones.
- Performance metrics for each rule: number of trades (sample size),
  hit rate (% positive), and average return per trade
- Statistical significance: with the sample size returned, is the hit
  rate meaningfully above 50% or is it noise (a 60% rate on 8 trades
  is noise; the same on 80 trades is signal)
- Most recent signal alignment: is either rule firing right now, and
  if so, does the historical edge support acting on it
- Sharpe-style intuition: even if hit rate is high, is the average
  return per trade large enough to clear realistic friction (commissions,
  slippage, spreads)
- Drawdown intuition: what is the worst single trade in the sample, and
  what does that say about position sizing
- Out-of-sample concern: have these specific rules been over-fit to this
  ticker's history, or do they survive across many tickers (default
  assumption: single-ticker backtests are weak; downgrade conviction
  accordingly)
- Survivorship bias: this ticker still trades; that itself is a bias.
  Acknowledge it in your risks.
- Go / no-go: only call BUY or SELL when at least one rule shows >60%
  hit rate over >=10 trades AND the most recent signal matches the
  direction. Otherwise return HOLD with conviction 3.

Use the tool_backtest_simple_rules ONCE to pull the rsi_mean_reversion
and sma_crossover summaries. Each covers the history from history_start
to as_of, with non-overlapping trades, and reports trades, hit_rate,
avg_return, worst_trade, baseline_avg_return (the average return of every
window of the same length, signal or not), signal_active_now and
last_signal_date. A rule has an edge only if avg_return clears
baseline_avg_return on a credible sample. Cite the specific numbers -
never paraphrase.

REGARDLESS of the recommendation, your thesis bullets MUST report the
actual numbers you observed: rule name, hit_rate, sample size (trades),
avg_return, and whether the rule is currently signaling. Example:
- "RSI mean reversion: 52% hit rate on 18 trades over 5y, avg 1.4%
   return vs 1.1% baseline - no edge above noise; not currently signaling."
- "50/200 SMA crossover: 1 trade in window, sample too small; last
   signal was X months ago."
Put the FINDINGS in thesis. Put what could be wrong with them in risks
(survivorship bias, single-ticker fragility, regime change risk).

{shared_rules_block()}

Return a structured AgentVerdict with agent='{AGENT_NAME}'. The framework
above guides your reasoning; the OUTPUT must be the AgentVerdict schema,
not a free-form backtest report.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_backtest_simple_rules],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
