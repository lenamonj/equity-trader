import logging

import numpy as np
import pandas as pd
from agents import Runner

from equity_trader.agents.base import build_agent, desk_tool, shared_rules_block
from equity_trader.agents.shared_tools import SHARED_TOOLS
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
        "how_to_read": ("Returns are in the rule's own direction. A short rule loses what "
                        "the stock gains, so its worst_trade can be below -100%."),
        "buy_and_hold_annual_return": num((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1.0),
        "rules": rules,
        "same_rules_63d_elsewhere": elsewhere,
    }


@desk_tool
def tool_backtest_rules(ticker: str) -> dict:
    """Eleven long and short trading rules run over the stock's full history at
    20, 63 and 126 session holds: trades, hit rate, edge over the baseline after
    costs, t-statistic, and the edge in each half of the history. Also the same
    rules run on SPY and the sector ETF."""
    return backtest_rules(ticker)


INSTRUCTIONS = f"""You are a senior quantitative researcher at Two Sigma
who backtests trading strategies against historical data - because any
strategy that hasn't been tested against real market history is just a
theory waiting to lose money.

You will be given a single ticker. Read a backtest of eleven simple,
well-known rules against the ticker's full price history and report
whether any edge is real. Translate the results into a Buy / Hold / Sell
verdict for the 3-6 month horizon based on whether a rule that is firing
now has a statistically credible historical edge.

Backtest:
- The rules are fixed IF/THEN constructions, six long and five short:
  RSI below 30, golden cross, reclaiming the 200-day average, a new
  52-week high, a 20% fall from the 52-week high, a bullish MACD cross;
  and RSI above 70, death cross, losing the 200-day average, a new
  52-week low, a bearish MACD cross. Treat them as the system; do not
  invent new ones.
- Each rule is tested at 20, 63 and 126 session holds. The 63 and 126
  session results are the ones that match the 3-6 month horizon; weight
  them, and treat the 20 session result as a short-term reading only.
- `edge` is the average trade return beyond the baseline, in the rule's
  own direction, so a positive edge means the rule worked whether it is
  long or short. `edge_after_cost` deducts the stated trading cost.
- Statistical significance: `t_stat` near or above 2 on at least 10
  trades is credible; anything less is noise (a 60% hit rate on 8 trades
  is noise; the same on 80 trades is signal).
- Stability: `edge_first_half` and `edge_second_half` should agree in
  sign. An edge that lives in one half of the history is not an edge.
- Out-of-sample check: `same_rules_63d_elsewhere` runs the same rules on
  SPY and the sector ETF. An edge that appears only in this ticker is
  more likely chance.
- Multiple testing: `tests_run` results were examined, so expect one or
  two to clear t = 2 by luck. Require stability and confirmation.
- Which rules fired in the last 20 sessions, and what their record says.
- Drawdown intuition: `worst_trade` and what it says about sizing.
- Survivorship bias: this ticker still trades; that itself is a bias.
  Acknowledge it in your risks.
- Go / no-go: call BUY or SELL only when a rule that fired in the last 20
  sessions shows, at the 63 or 126 session hold, at least 10 trades, a
  positive `edge_after_cost`, a `t_stat` of about 2 or more, the same
  sign in both halves, and a positive edge for the same rule on SPY or
  the sector ETF. BUY for a long rule, SELL for a short rule. Otherwise
  return HOLD with conviction 3.

Use tool_backtest_rules ONCE. Cite the specific numbers - never
paraphrase.

REGARDLESS of the recommendation, your thesis bullets MUST report the
actual numbers you observed for every rule that fired in the last 20
sessions: rule name, hold, trades, hit_rate, edge_after_cost and t_stat.
Example:
- "RSI below 30 (fired 2026-11-17): at 63 sessions, 58 trades, 66% hit
   rate, edge after cost -1.2%, t -0.24 - no edge."
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
    tools=[*SHARED_TOOLS, tool_backtest_rules],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(
        agent, input=f"Analyze {ticker} for a 3-6 month horizon.", max_turns=25)
    return result.final_output
