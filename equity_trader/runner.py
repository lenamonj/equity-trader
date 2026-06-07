import asyncio
import logging
from typing import Callable, Awaitable

import openai

from equity_trader import orchestrator
from equity_trader.agents import ALL_AGENTS
from equity_trader.data.cache import reset_run_cache
from equity_trader.data.yfinance_tools import get_quote
from equity_trader.persistence import persist
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict


log = logging.getLogger(__name__)


# Cap parallel LLM calls so we stay under OpenAI per-org token/request rate
# limits. With 7 ~3-4k-token system prompts firing simultaneously, Tier-1
# accounts hit 429s; 3 concurrent keeps headroom while still finishing a
# full ticker run in ~20-40s.
_AGENT_CONCURRENCY = 3

# Exponential backoff schedule (seconds) for OpenAI 429s. Total wait ~56s
# across 3 retries, which comfortably exceeds the typical 60s reset window.
_RATE_LIMIT_BACKOFF = [8.0, 16.0, 32.0]


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
    last_rate_limit: Exception | None = None
    for attempt, delay in enumerate(_RATE_LIMIT_BACKOFF + [None]):
        try:
            return await fn(ticker)
        except openai.RateLimitError as e:
            last_rate_limit = e
            if delay is None:
                log.warning("Agent %s: rate-limited after %d retries",
                            name, len(_RATE_LIMIT_BACKOFF))
                return _error_verdict(name, ticker, e)
            log.info("Agent %s: 429, retry %d/%d in %.0fs",
                      name, attempt + 1, len(_RATE_LIMIT_BACKOFF), delay)
            await asyncio.sleep(delay)
        except Exception as e:
            log.exception("Agent %s failed: %s", name, e)
            return _error_verdict(name, ticker, e)
    # Defensive: loop should always return above
    return _error_verdict(name, ticker,
                           last_rate_limit or RuntimeError("retry loop exited"))


async def analyze_ticker(ticker: str) -> OrchestratorVerdict:
    reset_run_cache()
    quote = get_quote(ticker)
    current_price = quote["last_price"]

    sem = asyncio.Semaphore(_AGENT_CONCURRENCY)

    async def _gated(name: str,
                     fn: Callable[[str], Awaitable[AgentVerdict]]) -> AgentVerdict:
        async with sem:
            return await _run_one(name, fn, ticker)

    verdicts = await asyncio.gather(
        *(_gated(name, fn) for name, fn in ALL_AGENTS)
    )
    final = await orchestrator.run(ticker, verdicts, current_price)
    persist(final)
    return final
