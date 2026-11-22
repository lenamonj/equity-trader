"""Grade every saved verdict against what the stock did next.

  uv run python grade.py

Reads `runs/*.json`, fetches adjusted closes once, and prints each caller's
record at three and six months. No model is called.
"""
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from equity_trader import grading
from equity_trader.config import AGENT_PERSONAS
from equity_trader.data import yfinance_tools as yft

# A redirected stdout on Windows defaults to cp1252.
sys.stdout.reconfigure(encoding="utf-8")
console = Console()

_TITLES = {"3m": "3 months", "6m": "6 months"}
_NOTE = ("A call is a hit when the stock's own return over the horizon has the sign of the "
         "call. HOLD calls are graded for return, not for hits. Several runs on one stock on "
         "one day are one observation: read the ticker-days column before the hit rate.")


def _pct(x: float, signed: bool = True) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.0f}%"


def _caller(who: str) -> str:
    return "Committee" if who == "committee" else AGENT_PERSONAS.get(who, (who,))[0]


def render(graded: pd.DataFrame) -> None:
    summary = grading.summarize(graded)
    for horizon, sessions in grading.HORIZONS.items():
        table = Table(title=f"{_TITLES[horizon]}: {sessions} sessions after the run",
                      title_justify="left")
        for column in ("Caller", "Graded", "Pending", "Ticker-days", "Buy/Sell calls",
                       "Hit rate", "Avg return of the call", f"Against {grading.BENCHMARK}"):
            table.add_column(column)
        for row in summary[summary["horizon"] == horizon].itertuples():
            table.add_row(_caller(row.who), str(row.graded), str(row.pending),
                          str(row.ticker_days), str(row.directional),
                          _pct(row.hit_rate, signed=False), _pct(row.avg_signed_return),
                          _pct(row.avg_signed_excess))
        console.print(table)
        console.print()

    calls = Table(title="Committee calls", title_justify="left")
    for column in ("Ticker", "Run (UTC)", "Entry close", "Call", "Conviction",
                   "3m return", f"3m vs {grading.BENCHMARK}",
                   "6m return", f"6m vs {grading.BENCHMARK}"):
        calls.add_column(column)
    committee = graded[graded["who"] == "committee"]
    for (ticker, run), rows in committee.groupby(["ticker", "run"], sort=False):
        by_horizon = rows.set_index("horizon")
        cells = []
        for horizon in grading.HORIZONS:
            row = by_horizon.loc[horizon]
            pending = pd.isna(row["stock_return"])
            cells += ["pending" if pending else _pct(row["stock_return"]),
                      "-" if pending else _pct(row["excess_return"])]
        first = rows.iloc[0]
        calls.add_row(ticker, run[:16].replace("T", " "), first["entry"].date().isoformat(),
                      first["call"], str(first["conviction"]), *cells)
    console.print(calls)
    console.print(f"\n{_NOTE}")


def main() -> None:
    runs = grading.load_runs()
    if not runs:
        console.print("No saved runs in runs/.")
        return
    tickers = sorted({run["ticker"] for run in runs} | {grading.BENCHMARK})
    render(grading.grade(runs, yft.get_closes(tickers, period="max")))


if __name__ == "__main__":
    main()
