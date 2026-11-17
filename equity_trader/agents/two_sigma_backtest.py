import logging

import numpy as np
import pandas as pd
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.data import yfinance_tools as yft
from equity_trader.data.technical_tools import YEAR, history, rsi
from equity_trader.data.util import num
from equity_trader.schemas import AgentVerdict

log = logging.getLogger(__name__)

AGENT_NAME = "two_sigma_backtest"


HOLDS = (20, 63, 126)  # sessions: one month, and the two ends of the horizon
COST = 0.001  # assumed round-trip trading cost, as a fraction


def _crosses_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift() <= b.shift()) & (a > b)


def _signals(df: pd.DataFrame) -> dict[str, tuple[int, pd.Series]]:
    """Rule name -> (direction, days the rule fires). +1 is long, -1 is short."""
    close = df["Close"]
    rsi_14 = rsi(close)
    sma_50, sma_200 = close.rolling(50).mean(), close.rolling(200).mean()
    high_52w, low_52w = close.rolling(YEAR).max(), close.rolling(YEAR).min()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    drawdown = close / high_52w - 1.0
    return {
        "rsi_below_30": (1, rsi_14 < 30),
        "golden_cross": (1, _crosses_above(sma_50, sma_200)),
        "reclaims_200_day_average": (1, _crosses_above(close, sma_200)),
        "new_52_week_high": (1, close > high_52w.shift()),
        "falls_20pct_below_52_week_high": (1, (drawdown.shift() > -0.20) & (drawdown <= -0.20)),
        "macd_bullish_cross": (1, _crosses_above(macd, macd_signal)),
        "rsi_above_70": (-1, rsi_14 > 70),
        "death_cross": (-1, _crosses_above(sma_200, sma_50)),
        "loses_200_day_average": (-1, _crosses_above(sma_200, close)),
        "new_52_week_low": (-1, close < low_52w.shift()),
        "macd_bearish_cross": (-1, _crosses_above(macd_signal, macd)),
    }


def _backtest_rule(close: pd.Series, signals: pd.Series, hold: int,
                   direction: int = 1) -> dict:
    """Trade each signal and hold `hold` sessions. Trades never overlap, so a
    run of consecutive signal days counts once, not once per day.

    `edge` is the average trade return beyond the baseline, in the rule's own
    direction: positive means the rule worked, long or short.
    """
    entries, next_free = [], 0
    for loc in np.flatnonzero(signals.to_numpy()):
        if loc >= next_free and loc + hold < len(close):
            entries.append(loc)
            next_free = loc + hold
    # Mean return of every window of this length, signal or not: the bar a
    # rule has to clear to claim an edge.
    baseline = float((close.shift(-hold) / close - 1.0).mean())
    summary = {"trades": len(entries), "hit_rate": None, "avg_return": None,
               "baseline_avg_return": num(baseline), "edge": None, "edge_after_cost": None,
               "t_stat": None, "worst_trade": None,
               "edge_first_half": None, "edge_second_half": None}
    if not entries:
        return summary

    returns = pd.Series([close.iloc[e + hold] / close.iloc[e] - 1.0 for e in entries],
                        index=close.index[entries])
    edge = direction * (returns - baseline)
    midpoint = close.index[len(close) // 2]
    first, second = edge[edge.index < midpoint], edge[edge.index >= midpoint]
    summary.update(
        hit_rate=num((direction * returns > 0).mean(), 2),
        avg_return=num(returns.mean()),
        edge=num(edge.mean()),
        edge_after_cost=num(edge.mean() - COST),
        t_stat=(num(edge.mean() / (edge.std() / len(edge) ** 0.5), 2)
                if len(edge) >= 3 and edge.std() > 0 else None),
        worst_trade=num((direction * returns).min()),
        edge_first_half=num(first.mean()) if len(first) else None,
        edge_second_half=num(second.mean()) if len(second) else None,
    )
    return summary


def _run_rules(df: pd.DataFrame, holds: tuple[int, ...]) -> dict:
    close = df["Close"]
    out = {}
    for name, (direction, signals) in _signals(df).items():
        fired = signals[signals]
        out[name] = {
            "direction": "long" if direction > 0 else "short",
            "fired_in_last_20_sessions": bool(signals.iloc[-20:].any()),
            "last_signal_date": fired.index[-1].date().isoformat() if len(fired) else None,
            "holds": {f"{h}d": _backtest_rule(close, signals, h, direction) for h in holds},
        }
    return out


def backtest_rules(ticker: str) -> dict:
    df = history(ticker)
    close = df["Close"]
    years = (df.index[-1] - df.index[0]).days / 365.25
    rules = _run_rules(df, HOLDS)

    # The same rules on the market and the sector: an edge that shows up only
    # in one ticker's history is more likely to be chance.
    sector_etf = yft.SECTOR_ETF.get(yft.get_info(ticker).get("sector"))
    elsewhere: dict = {}
    for symbol in ["SPY"] + ([sector_etf] if sector_etf else []):
        try:
            other = _run_rules(history(symbol), (63,))
        except Exception as exc:
            log.warning("Backtest cross-check unavailable for %s: %s", symbol, exc)
            continue
        elsewhere[symbol] = {name: {k: rule["holds"]["63d"][k]
                                    for k in ("trades", "edge", "t_stat")}
                             for name, rule in other.items()}
    return {
        "as_of": yft.as_of(df),
        "history_start": df.index[0].date().isoformat(),
        "cost_assumption": COST,
        "tests_run": len(rules) * len(HOLDS),
        "buy_and_hold_annual_return": num((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1.0),
        "rules": rules,
        "same_rules_63d_elsewhere": elsewhere,
    }


@function_tool
def tool_backtest_rules(ticker: str) -> dict:
    return backtest_rules(ticker)


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
    tools=[tool_backtest_rules],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(
        agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
