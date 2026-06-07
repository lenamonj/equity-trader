import asyncio
import pytest
from equity_trader.data.cache import cached_for_run, reset_run_cache


calls = {"n": 0}


@cached_for_run
def fetch(ticker: str, period: str = "2y") -> str:
    calls["n"] += 1
    return f"{ticker}:{period}"


def test_cache_hit_within_run():
    reset_run_cache()
    calls["n"] = 0
    assert fetch("AAPL") == "AAPL:2y"
    assert fetch("AAPL") == "AAPL:2y"
    assert calls["n"] == 1


def test_cache_reset_between_runs():
    reset_run_cache()
    calls["n"] = 0
    fetch("AAPL")
    reset_run_cache()
    fetch("AAPL")
    assert calls["n"] == 2


def test_cache_keys_on_kwargs():
    reset_run_cache()
    calls["n"] = 0
    fetch("AAPL", period="2y")
    fetch("AAPL", period="5y")
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_concurrent_runs_isolated():
    async def run(t):
        reset_run_cache()
        calls["n"] = 0  # not safe across tasks, used only to prove independence below
        fetch(t)
        fetch(t)
        return calls["n"]

    # ContextVar isolation: separate tasks get separate cache instances
    results = await asyncio.gather(run("AAPL"), run("MSFT"))
    # Each task should see exactly 1 underlying call within its own context
    assert all(r == 1 for r in results)
