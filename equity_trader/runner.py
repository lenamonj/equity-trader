import asyncio
import logging
from collections.abc import Awaitable, Callable

import openai

from equity_trader import orchestrator
from equity_trader.agents import ALL_AGENTS, ALL_DATA
from equity_trader.data.cache import reset_run_cache
from equity_trader.data.yfinance_tools import get_quote
from equity_trader.gatherer import gather
from equity_trader.persistence import persist
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict

log = logging.getLogger(__name__)


# Cap parallel LLM calls to stay under OpenAI per-org rate limits: eight
# agents firing at once hit 429s on Tier-1 accounts. Three at a time keeps
# headroom, and a full ticker run takes about a minute and a half.
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


def _stamp(v: AgentVerdict, name: str, ticker: str) -> AgentVerdict:
    # error_note is the system's failure marker and drops a desk from the
    # score. Models fill it with data caveats, so a returned verdict never
    # keeps one; the caveat moves to risks when there is room.
    risks = v.risks
    if v.error_note and len(risks) < 4:
        risks = [*risks, v.error_note]
    return v.model_copy(update={"agent": name, "ticker": ticker,
                                "risks": risks, "error_note": None})


Desk = Callable[[str, dict], Awaitable[AgentVerdict]]


async def _run_one(name: str, fn: Desk, ticker: str, gathered: dict) -> AgentVerdict:
    last_rate_limit: Exception | None = None
    for attempt, delay in enumerate(_RATE_LIMIT_BACKOFF + [None]):
        try:
            return _stamp(await fn(ticker, gathered), name, ticker)
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


async def analyze_ticker(
    ticker: str,
    on_verdict: Callable[[AgentVerdict], None] | None = None,
    on_gathered: Callable[[dict], None] | None = None,
) -> OrchestratorVerdict:
    """Run the committee: the gatherer, then the desks, then the chair.

    `on_gathered` is called with what the gatherer fetched once the data is
    in, and `on_verdict` as each desk reports.
    """
    reset_run_cache()
    quote = await asyncio.to_thread(get_quote, ticker)
    current_price = quote["last_price"]

    # Every desk's data is fetched here, in sequence, and handed to the desk
    # in its request. No desk calls a data provider.
    gathered = await asyncio.to_thread(gather, ticker, ALL_DATA)
    if on_gathered:
        on_gathered(gathered)

    sem = asyncio.Semaphore(_AGENT_CONCURRENCY)

    async def _gated(name: str, fn: Desk) -> AgentVerdict:
        async with sem:
            verdict = await _run_one(name, fn, ticker, gathered)
        if on_verdict:
            on_verdict(verdict)
        return verdict

    verdicts = await asyncio.gather(
        *(_gated(name, fn) for name, fn in ALL_AGENTS)
    )
    final = await orchestrator.run(ticker, verdicts, current_price)
    persist(final)
    return final
