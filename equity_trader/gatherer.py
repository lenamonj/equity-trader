"""The gatherer: fetches everything the desks will read before any desk runs.

It is code, not a model. What to fetch is fixed by the desks' own data lists,
so there is nothing for a model to decide. It calls each data function once,
one at a time, which keeps the load on the data providers low and even. Each
desk is then handed its sections in its request and answers in one model
call. A desk has no tools and never calls a data provider.
"""
import inspect
import json
import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


def gather(ticker: str, sections: list[Callable]) -> dict[str, dict]:
    """Run each data function in turn.

    Returns, by section name, its `data` or the `error` that stopped it, and
    the `seconds` it took. A failure is recorded, not raised: the desk is told
    that section is unavailable.
    """
    gathered = {}
    for function in sections:
        takes_ticker = bool(inspect.signature(function).parameters)
        started = time.monotonic()
        entry = {"data": None, "error": None}
        try:
            entry["data"] = function(*([ticker] if takes_ticker else []))
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("Gatherer: %s failed for %s: %s", function.__name__, ticker,
                        entry["error"])
        entry["seconds"] = round(time.monotonic() - started, 1)
        gathered[function.__name__] = entry
    failed = [name for name, entry in gathered.items() if entry["error"]]
    log.info("Gatherer: %d sections for %s in %.0fs, %d failed %s", len(gathered), ticker,
             sum(entry["seconds"] for entry in gathered.values()), len(failed), failed or "")
    return gathered


def brief(ticker: str, sections: list[Callable], gathered: dict[str, dict]) -> str:
    """A desk's request: the task, then each of its sections and what it holds."""
    parts = [f"Analyze {ticker} for a 3-6 month horizon.", "",
             "DATA, gathered for this run. This is everything you have."]
    for function in sections:
        entry = gathered[function.__name__]
        body = (f"UNAVAILABLE: {entry['error']}" if entry["error"]
                else json.dumps(entry["data"], separators=(",", ":"), default=str))
        parts += ["", f"## {function.__name__}", " ".join(function.__doc__.split()), body]
    return "\n".join(parts)
