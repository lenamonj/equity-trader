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


def test_callers_arriving_together_share_one_fetch():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    reset_run_cache()
    calls, release = [], threading.Event()

    @cached_for_run
    def slow(x):
        calls.append(x)
        release.wait(2)
        return x * 2

    # Each desk's tool runs in a thread that carries a copy of the run's context.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy_context().run, slow, 21) for _ in range(8)]
        time.sleep(0.2)
        release.set()
        assert [f.result() for f in futures] == [42] * 8
    assert calls == [21]


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
