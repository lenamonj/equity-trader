import asyncio
import logging
from typing import Callable, Awaitable

from equity_trader import orchestrator
from equity_trader.agents import ALL_AGENTS
from equity_trader.data.cache import reset_run_cache
from equity_trader.data.yfinance_tools import get_quote
from equity_trader.persistence import persist
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict


log = logging.getLogger(__name__)


def _error_verdict(agent: str, ticker: str, exc: Exception) -> AgentVerdict:
    return AgentVerdict(
        agent=agent, ticker=ticker, recommendation="HOLD", conviction=1,
        thesis=["Agent failed; HOLD placeholder."],
        risks=[f"Agent {agent} unavailable for this run."],
        data_cited=["n/a"],
        error_note=f"{type(exc).__name__}: {exc}",
    )


async def _run_one(name: str, fn: Callable[[str], Awaitable[AgentVerdict]],
                    ticker: str) -> AgentVerdict:
    try:
        return await fn(ticker)
    except Exception as e:
        log.exception("Agent %s failed: %s", name, e)
        return _error_verdict(name, ticker, e)


async def analyze_ticker(ticker: str) -> OrchestratorVerdict:
    reset_run_cache()
    quote = get_quote(ticker)
    current_price = quote["last_price"]

    verdicts = await asyncio.gather(
        *(_run_one(name, fn, ticker) for name, fn in ALL_AGENTS)
    )
    final = await orchestrator.run(ticker, verdicts, current_price)
    persist(final)
    return final
