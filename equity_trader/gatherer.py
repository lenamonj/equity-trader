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
import math
import re
import time
from collections.abc import Callable, Iterator
from decimal import Decimal
from numbers import Real

from equity_trader.schemas import CheckedEvidence, Evidence

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


_NUMBER = re.compile(r"\d[\d,]*\.?\d*")
# A desk may give a fraction as a percentage, or a large amount in thousands,
# millions, billions or trillions.
_SCALES = (1.0, 100.0, 1e-3, 1e-6, 1e-9, 1e-12)


def _numbers(data) -> Iterator[float]:
    """Every number in a section: its numeric values and the numbers in its text."""
    if isinstance(data, bool) or data is None:
        return
    if isinstance(data, Real):  # includes numpy's numbers
        yield abs(float(data))
    elif isinstance(data, str):
        for token in _NUMBER.findall(data):
            yield float(token.replace(",", ""))
    elif isinstance(data, dict):
        for value in data.values():
            yield from _numbers(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _numbers(value)


def _places(value: float) -> int:
    """How many decimal places a cited number was given to."""
    exponent = Decimal(repr(float(value))).normalize().as_tuple().exponent
    return max(-exponent, 0)


def in_data(value: float, data) -> bool:
    """Whether a cited number is in a section's data.

    It is found when some number in the section, rounded to the precision the
    desk gave, equals it. Sign is ignored, and so are the two common
    rescalings: a fraction as a percentage, a large amount in thousands up to
    trillions.
    """
    places, cited = _places(value), abs(value)
    return any(math.isclose(round(number * scale, places), cited, rel_tol=1e-9, abs_tol=1e-12)
               for number in _numbers(data) for scale in _SCALES)


def check(evidence: list[Evidence], gathered: dict[str, dict]) -> list[CheckedEvidence]:
    """Mark each reading a desk cited with whether its value is in the section it names."""
    return [CheckedEvidence(**item.model_dump(),
                            in_data=in_data(item.value, gathered[item.source]["data"]))
            for item in evidence]
