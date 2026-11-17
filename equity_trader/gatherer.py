"""The gatherer: fetches everything the desks will read before any desk runs.

It is code, not a model. What to fetch is fixed by the desks' own tool lists,
so there is nothing for a model to decide. It calls each tool once, one at a
time, which keeps the load on the data providers low and even. The results
land in the run cache, and the desks' own tool calls are answered from it.
A desk therefore never calls a data provider: every tool takes the ticker or
nothing, so there is no request a model could invent.
"""
import inspect
import logging
import time

from equity_trader import agents as desks
from equity_trader.agents.base import TOOL_FUNCTIONS

log = logging.getLogger(__name__)


def tool_names() -> list[str]:
    """Every tool the desks hold, shared ones first, each named once."""
    names = [tool.name for name, _ in desks.ALL_AGENTS
             for tool in getattr(desks, name).agent.tools]
    return list(dict.fromkeys(names))


def gather(ticker: str) -> list[dict]:
    """Run every desk tool in turn.

    Returns one row per tool: its name, how long it took, and the error if it
    failed. A failure is recorded, not raised: the desk that asks for that
    tool tries again and reports what it gets.
    """
    report = []
    for name in tool_names():
        function = TOOL_FUNCTIONS[name]
        takes_ticker = bool(inspect.signature(function).parameters)
        started = time.monotonic()
        error = None
        try:
            function(*([ticker] if takes_ticker else []))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Gatherer: %s failed for %s: %s", name, ticker, error)
        report.append({"tool": name, "seconds": round(time.monotonic() - started, 1),
                       "error": error})
    failed = [row["tool"] for row in report if row["error"]]
    log.info("Gatherer: %d tools for %s in %.0fs, %d failed %s", len(report), ticker,
             sum(row["seconds"] for row in report), len(failed), failed or "")
    return report
