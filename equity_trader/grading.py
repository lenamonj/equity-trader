"""Grade saved verdicts against what the stock did next.

Every run is saved as JSON. Once three or six months have passed, the call
the committee made, and the call each desk made, can be checked against the
stock's return over that horizon.
"""
import json
import math
from pathlib import Path

import pandas as pd

from equity_trader.exceptions import DataFetchError

# The 3 to 6 month horizon, in trading sessions.
HORIZONS = {"3m": 63, "6m": 126}
BENCHMARK = "SPY"
_SIGN = {"BUY": 1, "HOLD": 0, "SELL": -1}
_MARKET_TIME = "America/New_York"
_CLOSE_HOUR = 16
_COLUMNS = ["ticker", "run", "who", "call", "conviction", "horizon", "entry", "exit",
            "stock_return", "benchmark_return", "excess_return", "signed_return",
            "signed_excess", "hit"]


def _utc(timestamp: str) -> pd.Timestamp:
    """A run's timestamp. Early runs were saved in UTC with no zone marked."""
    when = pd.Timestamp(timestamp)
    return when.tz_localize("UTC") if when.tzinfo is None else when


def load_runs(runs_dir: str | Path = "runs") -> list[dict]:
    """Every saved verdict, oldest first."""
    runs = []
    for file in Path(runs_dir).glob("*.json"):
        raw = file.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Early runs were written in the Windows default encoding.
            text = raw.decode("cp1252")
        runs.append(json.loads(text))
    return sorted(runs, key=lambda run: _utc(run["run_timestamp"]))


def _entry(timestamp: str, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """The session whose close is the last one before the run."""
    local = _utc(timestamp).tz_convert(_MARKET_TIME).tz_localize(None)
    day = local.normalize() - pd.Timedelta(days=int(local.hour < _CLOSE_HOUR))
    return sessions[sessions <= day][-1]


def grade(runs: list[dict], closes: pd.DataFrame) -> pd.DataFrame:
    """One row per run, horizon and caller: the committee and each desk that
    reported.

    `closes` holds adjusted daily closes, one column per ticker and one for
    the benchmark. A call is a hit when the stock's own return over the
    horizon has the sign of the call; a HOLD is neither hit nor miss. The
    return against the benchmark is given beside it, so a rising market is
    not read as skill. A horizon that has not elapsed has no returns yet.
    """
    rows = []
    for run in runs:
        ticker = run["ticker"]
        prices = closes[ticker].dropna() if ticker in closes else pd.Series(dtype=float)
        if prices.empty:
            raise DataFetchError(f"No price history for {ticker}")
        entry = _entry(run["run_timestamp"], prices.index)
        callers = [("committee", run["final_recommendation"], run["conviction"])] + [
            (desk["agent"], desk["recommendation"], desk["conviction"])
            for desk in run["agent_verdicts"] if not desk.get("error_note")]
        for horizon, sessions in HORIZONS.items():
            at = prices.index.get_loc(entry) + sessions
            exit_, stock, benchmark = pd.NaT, math.nan, math.nan
            if at < len(prices):
                exit_ = prices.index[at]
                stock = prices[exit_] / prices[entry] - 1.0
                benchmark = closes[BENCHMARK][exit_] / closes[BENCHMARK][entry] - 1.0
            for who, call, conviction in callers:
                sign = _SIGN[call]
                rows.append({
                    "ticker": ticker, "run": run["run_timestamp"], "who": who, "call": call,
                    "conviction": conviction, "horizon": horizon, "entry": entry,
                    "exit": exit_, "stock_return": stock, "benchmark_return": benchmark,
                    "excess_return": stock - benchmark,
                    "signed_return": sign * stock,
                    "signed_excess": sign * (stock - benchmark),
                    "hit": (None if sign == 0 or math.isnan(stock)
                            else bool(sign * stock > 0)),
                })
    graded = pd.DataFrame(rows, columns=_COLUMNS)
    # Kept as objects so a HOLD or a pending row stays None beside True and False.
    graded["hit"] = graded["hit"].astype(object)
    return graded


def summarize(graded: pd.DataFrame) -> pd.DataFrame:
    """Per horizon and caller: how many calls were graded, how many are still
    pending, the hit rate over BUY and SELL calls, and the average return of
    acting on the call, outright and against the benchmark.

    `ticker_days` counts distinct stock and entry-date pairs. Several runs on
    one stock on one day are one observation, not several.
    """
    out = []
    for (horizon, who), group in graded.groupby(["horizon", "who"], sort=False):
        done = group.dropna(subset=["stock_return"])
        directional = done[done["call"] != "HOLD"]
        hits = sum(hit is True for hit in directional["hit"])
        out.append({
            "horizon": horizon, "who": who, "graded": len(done),
            "pending": len(group) - len(done),
            "ticker_days": len(done[["ticker", "entry"]].drop_duplicates()),
            "directional": len(directional), "hits": hits,
            "hit_rate": hits / len(directional) if len(directional) else math.nan,
            "avg_signed_return": done["signed_return"].mean(),
            "avg_signed_excess": done["signed_excess"].mean(),
        })
    return pd.DataFrame(out)
