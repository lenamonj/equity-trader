# EquityTrader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Per-project rule (overrides global git default):** Every "Commit" step in this plan is a request for the user's approval, not an autonomous commit. Stage changes, show the proposed message, and ask before running `git commit`. Never push.

**Goal:** Build a multi-agent equity research system that produces a single Buy/Hold/Sell verdict (3-6 month horizon) for a ticker by fanning out to 7 specialist trader agents and synthesizing their structured outputs.

**Architecture:** OpenAI Agents SDK, parallel fan-out via `asyncio.gather` over 7 agents (each with its own model and tools), then a single orchestrator (GPT-4o) that does deterministic weighted scoring plus an LLM-judged synthesis. Per-run in-memory tool cache. SQLite + JSON snapshot persistence. Gradio UI and Rich CLI both wrap the same async engine.

**Tech Stack:** Python 3.11+, `uv`, `openai-agents`, `openai`, `google-genai`, `yfinance`, `httpx` (for SEC EDGAR), `fredapi`, `pandas`, `numpy`, `pydantic`, `gradio`, `rich`, `pytest`, `pytest-asyncio`, `respx`.

**Reference spec:** `docs/superpowers/specs/2026-06-07-equitytrader-design.md`

---

## Phase 0 - Project Scaffolding

### Task 1: Initialize project with uv and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`
Expected: prints a version string. If missing, install per https://docs.astral.sh/uv/.

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "equity-trader"
version = "0.1.0"
description = "Multi-agent equity research producing 3-6 month Buy/Hold/Sell verdicts"
requires-python = ">=3.11"
dependencies = [
  "openai-agents>=0.0.10",
  "openai>=1.50.0",
  "google-genai>=0.3.0",
  "yfinance>=0.2.40",
  "httpx>=0.27.0",
  "fredapi>=0.5.2",
  "pandas>=2.2.0",
  "numpy>=1.26.0",
  "pydantic>=2.8.0",
  "gradio>=4.40.0",
  "rich>=13.7.0",
  "python-dotenv>=1.0.1",
]

[dependency-groups]
dev = [
  "pytest>=8.0.0",
  "pytest-asyncio>=0.23.0",
  "respx>=0.21.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
  "slow: integration tests that hit real APIs (deselect with -m 'not slow')",
]
testpaths = ["tests"]
```

- [ ] **Step 3: Pin Python version**

Write `.python-version` with content `3.11`.

- [ ] **Step 4: Sync the environment**

Run: `uv sync`
Expected: `.venv` created, lockfile written, no errors.

- [ ] **Step 5: Verify imports**

Run: `uv run python -c "import yfinance, fredapi, agents, openai, pydantic, gradio; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Request commit**

Stage: `git add pyproject.toml .python-version uv.lock`
Propose message: `chore: scaffold uv project with deps`
Ask user before committing.

---

### Task 2: Create package layout

**Files:**
- Create: `equity_trader/__init__.py`
- Create: `equity_trader/data/__init__.py`
- Create: `equity_trader/agents/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/agents/__init__.py`
- Create: `runs/.gitkeep`

- [ ] **Step 1: Create directory layout**

Use Write tool to create each `__init__.py` with content `"""EquityTrader package."""` (top-level) or empty string (sub-packages). Create `runs/.gitkeep` as an empty file.

- [ ] **Step 2: Verify layout**

Run: `uv run python -c "import equity_trader, equity_trader.data, equity_trader.agents; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Request commit**

Stage all new files; propose message: `chore: add package layout`. Ask user before committing.

---

### Task 3: Project .gitignore and .env.template

**Files:**
- Create: `.gitignore`
- Create: `.env.template`

- [ ] **Step 1: Write .gitignore**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.coverage
.env
runs/*.json
equity_trader.db
equity_trader.db-shm
equity_trader.db-wal
.DS_Store
```

- [ ] **Step 2: Write .env.template** (no real keys)

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
DEEPSEEK_API_KEY=
FRED_API_KEY=
SEC_EDGAR_USER_AGENT_EMAIL=you@example.com
```

- [ ] **Step 3: Confirm `.env` is ignored**

Run: `git check-ignore -v .env` (if `.env` exists)
Expected: shows the `.gitignore` line. If `.env` doesn't exist yet, skip.

- [ ] **Step 4: Request commit**

Stage `.gitignore` and `.env.template`. Propose message: `chore: add gitignore and env template`. Ask user before committing.

---

## Phase 1 - Schemas and Exceptions

These are the foundation. Every other module imports from here.

### Task 4: Custom exception types

**Files:**
- Create: `equity_trader/exceptions.py`
- Test: `tests/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exceptions.py
import pytest
from equity_trader.exceptions import (
    DataFetchError,
    TickerNotFoundError,
    RateLimitError,
    AgentExecutionError,
)


def test_exception_hierarchy():
    assert issubclass(TickerNotFoundError, DataFetchError)
    assert issubclass(RateLimitError, DataFetchError)
    assert issubclass(AgentExecutionError, Exception)


def test_ticker_not_found_carries_ticker():
    err = TickerNotFoundError("XYZAB")
    assert err.ticker == "XYZAB"
    assert "XYZAB" in str(err)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/exceptions.py
class DataFetchError(Exception):
    """Base for data layer failures."""


class TickerNotFoundError(DataFetchError):
    def __init__(self, ticker: str):
        super().__init__(f"Ticker not found: {ticker}")
        self.ticker = ticker


class RateLimitError(DataFetchError):
    """Raised when an upstream provider rate-limits us."""


class AgentExecutionError(Exception):
    """Raised when an agent fails after retries; carries the agent name."""

    def __init__(self, agent: str, cause: Exception):
        super().__init__(f"Agent {agent} failed: {cause}")
        self.agent = agent
        self.cause = cause
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add custom exception types`. Ask before committing.

---

### Task 5: AgentVerdict and CatalystEvent schemas

**Files:**
- Create: `equity_trader/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
import pytest
from datetime import date
from pydantic import ValidationError
from equity_trader.schemas import AgentVerdict, CatalystEvent, Recommendation


def test_agent_verdict_minimal_ok():
    v = AgentVerdict(
        agent="jpm_fundamental",
        ticker="NVDA",
        recommendation="BUY",
        conviction=8,
        price_target_6mo=215.0,
        thesis=["FCF expanding", "Capex visibility"],
        risks=["China exports"],
        data_cited=["10-Q Q1"],
    )
    assert v.recommendation == "BUY"
    assert v.error_note is None


def test_agent_verdict_conviction_range():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="BUY", conviction=11,
            thesis=["x"], risks=["y"], data_cited=["z"],
        )


def test_agent_verdict_recommendation_literal():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="STRONG_BUY", conviction=5,
            thesis=["x"], risks=["y"], data_cited=["z"],
        )


def test_agent_verdict_thesis_min_one():
    with pytest.raises(ValidationError):
        AgentVerdict(
            agent="a", ticker="T", recommendation="BUY", conviction=5,
            thesis=[], risks=["y"], data_cited=["z"],
        )


def test_catalyst_event():
    e = CatalystEvent(event="Q2 earnings", date=date(2026, 8, 27),
                      expected_impact="HIGH", notes="Guidance update")
    assert e.expected_impact == "HIGH"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/schemas.py
from datetime import date, datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

Recommendation = Literal["BUY", "HOLD", "SELL"]
Impact = Literal["LOW", "MEDIUM", "HIGH"]
PositionSize = Literal[
    "not sizeable", "starter (~1%)", "full (~3%)", "high conviction (~5%)"
]


class AgentVerdict(BaseModel):
    agent: str
    ticker: str
    recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    price_target_6mo: Optional[float] = None
    thesis: list[str] = Field(min_length=1, max_length=6)
    risks: list[str] = Field(min_length=1, max_length=4)
    data_cited: list[str] = Field(min_length=1)
    error_note: Optional[str] = None


class CatalystEvent(BaseModel):
    event: str
    date: date
    expected_impact: Impact
    notes: Optional[str] = None
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add AgentVerdict and CatalystEvent schemas`. Ask first.

---

### Task 6: OrchestratorVerdict schema

**Files:**
- Modify: `equity_trader/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_schemas.py`:

```python
from datetime import datetime
from equity_trader.schemas import OrchestratorVerdict


def _sample_verdict(agent="a", rec="BUY"):
    return AgentVerdict(
        agent=agent, ticker="NVDA", recommendation=rec, conviction=7,
        price_target_6mo=200.0, thesis=["t"], risks=["r"], data_cited=["d"],
    )


def test_orchestrator_verdict_minimal_ok():
    ov = OrchestratorVerdict(
        ticker="NVDA",
        run_timestamp=datetime.utcnow(),
        current_price=190.0,
        final_recommendation="BUY",
        conviction=7,
        price_target_6mo=210.0,
        weighted_score=4.2,
        weights_used={"jpm_fundamental": 25.0},
        weight_overrides_rationale=None,
        synthesis="Strong fundamentals; macro mixed but acceptable for horizon.",
        key_agreements=["FCF strong"],
        key_disagreements=["Macro vs technical"],
        dominant_drivers=["jpm_fundamental", "gs_technical"],
        red_flags=["Customer concentration"],
        position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0,
        catalyst_calendar=[],
        agent_verdicts=[_sample_verdict()],
    )
    assert ov.final_recommendation == "BUY"


def test_orchestrator_verdict_position_sizing_literal():
    with pytest.raises(ValidationError):
        OrchestratorVerdict(
            ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
            final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
            weighted_score=4.2, weights_used={"a": 100.0},
            weight_overrides_rationale=None, synthesis="x",
            key_agreements=[], key_disagreements=[], dominant_drivers=[],
            red_flags=[], position_sizing_suggestion="MAX OUT",
            stop_loss_level=None, catalyst_calendar=[], agent_verdicts=[],
        )
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_schemas.py::test_orchestrator_verdict_minimal_ok -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement** (append to `equity_trader/schemas.py`)

```python
class OrchestratorVerdict(BaseModel):
    ticker: str
    run_timestamp: datetime
    current_price: float
    final_recommendation: Recommendation
    conviction: int = Field(ge=1, le=10)
    price_target_6mo: Optional[float] = None
    weighted_score: float
    weights_used: dict[str, float]
    weight_overrides_rationale: Optional[str] = None
    synthesis: str = Field(min_length=50, max_length=3000)
    key_agreements: list[str]
    key_disagreements: list[str]
    dominant_drivers: list[str]
    red_flags: list[str]
    position_sizing_suggestion: PositionSize
    stop_loss_level: Optional[float] = None
    catalyst_calendar: list[CatalystEvent]
    agent_verdicts: list[AgentVerdict]
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add OrchestratorVerdict schema`. Ask first.

---

## Phase 2 - Data Layer

### Task 7: Per-run cache via ContextVar

**Files:**
- Create: `equity_trader/data/cache.py`
- Test: `tests/data/test_cache.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_cache.py
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
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/data/cache.py
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Any

_run_cache: ContextVar[dict | None] = ContextVar("run_cache", default=None)


def reset_run_cache() -> None:
    """Start a fresh per-run cache for the current ContextVar context."""
    _run_cache.set({})


def _get_cache() -> dict:
    cache = _run_cache.get()
    if cache is None:
        cache = {}
        _run_cache.set(cache)
    return cache


def cached_for_run(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__, args, tuple(sorted(kwargs.items())))
        cache = _get_cache()
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add per-run cache via ContextVar`. Ask first.

---

### Task 8: yfinance core tools (price, quote, fundamentals, analyst targets)

**Files:**
- Create: `equity_trader/data/yfinance_tools.py`
- Test: `tests/data/test_yfinance_tools.py`

- [ ] **Step 1: Write tests** (unit tests with monkeypatch + one `slow`-marked integration test)

```python
# tests/data/test_yfinance_tools.py
import pytest
import pandas as pd
from equity_trader.data import yfinance_tools as yft
from equity_trader.exceptions import TickerNotFoundError


def test_get_price_history_empty_raises(monkeypatch):
    class FakeTicker:
        def history(self, period, interval):
            return pd.DataFrame()

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    with pytest.raises(TickerNotFoundError):
        yft.get_price_history("ZZZZ")


def test_get_quote_returns_dict(monkeypatch):
    class FakeTicker:
        fast_info = {"lastPrice": 190.5, "marketCap": 4.6e12,
                     "dayHigh": 192.0, "dayLow": 188.0}

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    q = yft.get_quote("NVDA")
    assert q["last_price"] == 190.5
    assert q["market_cap"] == 4.6e12


@pytest.mark.slow
def test_get_price_history_real_aapl():
    df = yft.get_price_history("AAPL", period="1mo", interval="1d")
    assert len(df) > 10
    assert "Close" in df.columns
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/data/yfinance_tools.py
import time
import yfinance as yf
import pandas as pd

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import TickerNotFoundError, DataFetchError

_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.1  # seconds


def _throttle() -> None:
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL["t"] = time.monotonic()


@cached_for_run
def get_price_history(ticker: str, period: str = "2y",
                     interval: str = "1d") -> pd.DataFrame:
    _throttle()
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise TickerNotFoundError(ticker)
    return df


@cached_for_run
def get_quote(ticker: str) -> dict:
    _throttle()
    info = yf.Ticker(ticker).fast_info
    try:
        return {
            "last_price": float(info["lastPrice"]),
            "market_cap": float(info.get("marketCap") or 0.0),
            "day_high": float(info.get("dayHigh") or 0.0),
            "day_low": float(info.get("dayLow") or 0.0),
        }
    except (KeyError, TypeError) as e:
        raise DataFetchError(f"Quote unavailable for {ticker}: {e}")


@cached_for_run
def get_fundamentals(ticker: str) -> dict:
    _throttle()
    t = yf.Ticker(ticker)
    return {
        "income_stmt": t.income_stmt.to_dict() if t.income_stmt is not None else {},
        "balance_sheet": t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
        "cash_flow": t.cashflow.to_dict() if t.cashflow is not None else {},
    }


@cached_for_run
def get_analyst_targets(ticker: str) -> dict:
    _throttle()
    info = yf.Ticker(ticker).info or {}
    return {
        "mean_target": info.get("targetMeanPrice"),
        "high_target": info.get("targetHighPrice"),
        "low_target": info.get("targetLowPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
    }
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: 2 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add yfinance core tools (price, quote, fundamentals, targets)`. Ask first.

---

### Task 9: yfinance computed tools (technicals, statistical patterns)

**Files:**
- Modify: `equity_trader/data/yfinance_tools.py`
- Modify: `tests/data/test_yfinance_tools.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/data/test_yfinance_tools.py`:

```python
import numpy as np


def _fake_history(monkeypatch, n=300):
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    df = pd.DataFrame({
        "Open": prices, "High": prices + 0.5, "Low": prices - 0.5,
        "Close": prices, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))

    class FakeTicker:
        def history(self, period, interval):
            return df

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    yft.reset_run_cache() if hasattr(yft, "reset_run_cache") else None
    return df


def test_compute_technicals_returns_keys(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()
    _fake_history(monkeypatch)
    out = yft.compute_technicals("AAPL")
    assert {"rsi_14", "macd", "macd_signal", "sma_50", "sma_200",
            "atr_14", "last_close", "vol_20"}.issubset(out.keys())
    assert 0 <= out["rsi_14"] <= 100


def test_compute_statistical_patterns_returns_keys(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()
    _fake_history(monkeypatch)
    out = yft.compute_statistical_patterns("AAPL")
    assert {"autocorr_1d", "autocorr_5d", "monthly_seasonality",
            "regime", "realized_vol_30d"}.issubset(out.keys())
    assert out["regime"] in {"trending", "mean_reverting", "mixed"}
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: 2 new tests FAIL.

- [ ] **Step 3: Implement** (append to `equity_trader/data/yfinance_tools.py`)

```python
import numpy as np


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


@cached_for_run
def compute_technicals(ticker: str) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return {
        "last_close": float(close.iloc[-1]),
        "sma_50": float(close.rolling(50).mean().iloc[-1]),
        "sma_200": float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None,
        "rsi_14": _rsi(close),
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "atr_14": _atr(df),
        "vol_20": float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)),
    }


@cached_for_run
def compute_statistical_patterns(ticker: str) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    rets = df["Close"].pct_change().dropna()
    ac1 = float(rets.autocorr(lag=1) or 0.0)
    ac5 = float(rets.autocorr(lag=5) or 0.0)
    by_month = rets.groupby(rets.index.month).mean().to_dict()
    realized_vol = float(rets.tail(30).std() * (252 ** 0.5))
    if ac1 > 0.05:
        regime = "trending"
    elif ac1 < -0.05:
        regime = "mean_reverting"
    else:
        regime = "mixed"
    return {
        "autocorr_1d": ac1,
        "autocorr_5d": ac5,
        "monthly_seasonality": {int(k): float(v) for k, v in by_month.items()},
        "realized_vol_30d": realized_vol,
        "regime": regime,
    }
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: 4 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add technical and statistical-pattern computations`. Ask first.

---

### Task 10: yfinance options, IV stats, peer set, factor loads

**Files:**
- Modify: `equity_trader/data/yfinance_tools.py`
- Modify: `tests/data/test_yfinance_tools.py`

- [ ] **Step 1: Add failing tests**

```python
def test_get_peer_set(monkeypatch):
    class FakeTicker:
        info = {"sector": "Technology", "industry": "Semiconductors"}

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    peers = yft.get_peer_set("NVDA")
    assert isinstance(peers, list)
    assert len(peers) >= 1
    assert "NVDA" not in peers


def test_compute_factor_loads_runs(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()
    _fake_history(monkeypatch)
    out = yft.compute_factor_loads("NVDA", peers=["AMD", "INTC"])
    assert {"momentum_12_1", "volatility_60d", "rel_strength_vs_peers"}.issubset(out.keys())
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: new tests FAIL.

- [ ] **Step 3: Implement** (append)

```python
# Simple sector-peer heuristic. Real peer discovery would use a richer source;
# yfinance's `info` doesn't expose peers directly, so we use a small hardcoded
# map keyed on sector/industry. Override at call-site if richer peers needed.
_SECTOR_PEERS = {
    ("Technology", "Semiconductors"): ["AMD", "INTC", "AVGO", "TSM", "QCOM"],
    ("Technology", "Software - Infrastructure"): ["MSFT", "ORCL", "CRM", "ADBE"],
    ("Technology", "Software - Application"): ["CRM", "NOW", "SNOW", "DDOG"],
    ("Communication Services", "Internet Content & Information"):
        ["GOOGL", "META", "PINS", "SNAP"],
    ("Consumer Cyclical", "Internet Retail"): ["AMZN", "MELI", "EBAY"],
    ("Financial Services", "Banks - Diversified"): ["JPM", "BAC", "C", "WFC"],
}


@cached_for_run
def get_peer_set(ticker: str) -> list[str]:
    _throttle()
    info = yf.Ticker(ticker).info or {}
    key = (info.get("sector"), info.get("industry"))
    peers = _SECTOR_PEERS.get(key, [])
    return [p for p in peers if p.upper() != ticker.upper()]


@cached_for_run
def get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    _throttle()
    t = yf.Ticker(ticker)
    expiries = list(t.options or [])
    if not expiries:
        return {"expiries": [], "calls": [], "puts": [], "selected_expiry": None}
    chosen = expiry or expiries[min(2, len(expiries) - 1)]  # ~near-term but not weeklies
    chain = t.option_chain(chosen)
    return {
        "expiries": expiries,
        "selected_expiry": chosen,
        "calls": chain.calls[["strike", "lastPrice", "impliedVolatility",
                              "openInterest", "volume"]].to_dict(orient="records"),
        "puts": chain.puts[["strike", "lastPrice", "impliedVolatility",
                            "openInterest", "volume"]].to_dict(orient="records"),
    }


@cached_for_run
def compute_iv_stats(ticker: str) -> dict:
    chain = get_options_chain(ticker)
    calls = chain["calls"]
    puts = chain["puts"]
    if not calls or not puts:
        return {"atm_iv": None, "put_call_iv_skew": None,
                "selected_expiry": chain["selected_expiry"]}
    quote = get_quote(ticker)
    spot = quote["last_price"]
    atm_call = min(calls, key=lambda c: abs(c["strike"] - spot))
    atm_put = min(puts, key=lambda p: abs(p["strike"] - spot))
    return {
        "atm_iv": float(atm_call["impliedVolatility"]),
        "put_call_iv_skew": float(atm_put["impliedVolatility"]
                                   - atm_call["impliedVolatility"]),
        "selected_expiry": chain["selected_expiry"],
    }


@cached_for_run
def compute_factor_loads(ticker: str, peers: list[str]) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    close = df["Close"]
    # 12-1 momentum: return from t-252 to t-21
    if len(close) >= 252:
        mom_12_1 = float(close.iloc[-21] / close.iloc[-252] - 1.0)
    else:
        mom_12_1 = float(close.iloc[-1] / close.iloc[0] - 1.0)
    vol_60 = float(close.pct_change().tail(60).std() * (252 ** 0.5))

    # Relative strength vs peers over last 90 sessions
    base_ret = float(close.iloc[-1] / close.iloc[-90] - 1.0) if len(close) >= 90 else 0.0
    peer_rets = []
    for p in peers:
        try:
            pdf = get_price_history(p, period="6mo", interval="1d")
            peer_rets.append(float(pdf["Close"].iloc[-1] / pdf["Close"].iloc[0] - 1.0))
        except Exception:
            continue
    rel = base_ret - (sum(peer_rets) / len(peer_rets)) if peer_rets else 0.0

    return {
        "momentum_12_1": mom_12_1,
        "volatility_60d": vol_60,
        "rel_strength_vs_peers": rel,
        "peer_count_used": len(peer_rets),
    }
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_yfinance_tools.py -v -m "not slow"`
Expected: 6 PASS total.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add options, IV, peer set, and factor load tools`. Ask first.

---

### Task 11: SEC EDGAR tools

**Files:**
- Create: `equity_trader/data/edgar_tools.py`
- Test: `tests/data/test_edgar_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_edgar_tools.py
import pytest
import respx
import httpx
from equity_trader.data import edgar_tools as et


_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}


@respx.mock
def test_ticker_to_cik():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON)
    )
    cik = et._ticker_to_cik("AAPL")
    assert cik == "0000320193"


@respx.mock
def test_get_recent_8k_summaries():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON)
    )
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={
            "filings": {"recent": {
                "form": ["8-K", "10-Q", "8-K"],
                "filingDate": ["2026-05-01", "2026-04-15", "2026-03-10"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "accessionNumber": ["1-1", "2-2", "3-3"],
                "items": ["2.02", "", "8.01"],
            }}
        })
    )
    out = et.get_recent_8k_summaries("AAPL", n=2)
    assert len(out) == 2
    assert out[0]["form"] == "8-K"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_edgar_tools.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/data/edgar_tools.py
import os
import time
import httpx

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import TickerNotFoundError, DataFetchError

_BASE = "https://www.sec.gov"
_DATA = "https://data.sec.gov"
_TICKERS_URL = f"{_BASE}/files/company_tickers.json"
_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.1  # ~10 req/sec ceiling per SEC


def _user_agent() -> str:
    email = os.environ.get("SEC_EDGAR_USER_AGENT_EMAIL", "anon@example.com")
    return f"EquityTrader {email}"


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": _user_agent()}, timeout=15.0)


def _throttle() -> None:
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL["t"] = time.monotonic()


@cached_for_run
def _ticker_to_cik(ticker: str) -> str:
    _throttle()
    with _client() as c:
        r = c.get(_TICKERS_URL)
        r.raise_for_status()
        data = r.json()
    upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == upper:
            return f"{int(entry['cik_str']):010d}"
    raise TickerNotFoundError(ticker)


def _submissions(cik: str) -> dict:
    _throttle()
    with _client() as c:
        r = c.get(f"{_DATA}/submissions/CIK{cik}.json")
        r.raise_for_status()
        return r.json()


@cached_for_run
def get_recent_8k_summaries(ticker: str, n: int = 5) -> list[dict]:
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    out: list[dict] = []
    for i, form in enumerate(recent["form"]):
        if form != "8-K":
            continue
        out.append({
            "form": form,
            "filing_date": recent["filingDate"][i],
            "accession_no": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            "items": recent.get("items", [""] * len(recent["form"]))[i],
        })
        if len(out) >= n:
            break
    return out


def _latest_form(ticker: str, form: str) -> dict:
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    for i, f in enumerate(recent["form"]):
        if f == form:
            return {
                "form": f,
                "filing_date": recent["filingDate"][i],
                "accession_no": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "cik": cik,
            }
    raise DataFetchError(f"No {form} found for {ticker}")


@cached_for_run
def get_latest_10k(ticker: str) -> dict:
    return _latest_form(ticker, "10-K")


@cached_for_run
def get_latest_10q(ticker: str) -> dict:
    return _latest_form(ticker, "10-Q")


@cached_for_run
def get_filing_text(accession_no: str, cik: str, primary_document: str) -> str:
    _throttle()
    acc_clean = accession_no.replace("-", "")
    url = f"{_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_document}"
    with _client() as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_edgar_tools.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add SEC EDGAR data tools`. Ask first.

---

### Task 12: FRED tools

**Files:**
- Create: `equity_trader/data/fred_tools.py`
- Test: `tests/data/test_fred_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_fred_tools.py
import pytest
import pandas as pd
from equity_trader.data import fred_tools as ft


class FakeFred:
    def __init__(self, *_, **__):
        pass

    def get_series(self, series_id, observation_start=None):
        idx = pd.date_range("2024-01-01", periods=10, freq="MS")
        return pd.Series([1.0 + i for i in range(10)], index=idx, name=series_id)


def test_get_series(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    s = ft.get_series("DGS10")
    assert s.iloc[-1] == 10.0
    assert s.name == "DGS10"


def test_get_macro_snapshot(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    snap = ft.get_macro_snapshot()
    for k in ["dgs10", "dgs2", "t10y2y", "cpi_yoy", "unrate", "fedfunds", "nfci", "vix"]:
        assert k in snap
        assert snap[k] is None or isinstance(snap[k], (int, float))
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/data/test_fred_tools.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/data/fred_tools.py
import os
import pandas as pd
from fredapi import Fred

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import DataFetchError


def _client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise DataFetchError("FRED_API_KEY not set")
    return Fred(api_key=key)


@cached_for_run
def get_series(series_id: str, start: str | None = None) -> pd.Series:
    s = _client().get_series(series_id, observation_start=start)
    if s is None or len(s) == 0:
        raise DataFetchError(f"FRED series empty: {series_id}")
    return s


def _last_or_none(series_id: str) -> float | None:
    try:
        s = get_series(series_id)
        v = s.dropna().iloc[-1]
        return float(v)
    except Exception:
        return None


@cached_for_run
def get_macro_snapshot() -> dict:
    cpi = get_series("CPIAUCSL")
    cpi_yoy = None
    try:
        cpi_yoy = float((cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0)
    except Exception:
        pass
    return {
        "dgs10": _last_or_none("DGS10"),
        "dgs2": _last_or_none("DGS2"),
        "t10y2y": _last_or_none("T10Y2Y"),
        "cpi_yoy": cpi_yoy,
        "unrate": _last_or_none("UNRATE"),
        "fedfunds": _last_or_none("FEDFUNDS"),
        "nfci": _last_or_none("NFCI"),
        "vix": _last_or_none("VIXCLS"),
    }


# Sector → macro series mapping for the Bridgewater agent
_SECTOR_SERIES = {
    "Financials": ["DGS10", "T10Y2Y", "DRCCLACBS"],
    "Technology": ["UMCSENT", "INDPRO"],
    "Energy": ["DCOILWTICO", "DHHNGSP"],
    "Consumer Cyclical": ["UMCSENT", "RSXFS"],
    "Industrials": ["INDPRO", "ISRATIO"],
    "Healthcare": ["CPIAUCSL"],
    "Utilities": ["DGS10"],
    "Real Estate": ["MORTGAGE30US", "DGS10"],
}


@cached_for_run
def get_sector_macro(sector: str) -> dict:
    series_ids = _SECTOR_SERIES.get(sector, [])
    return {sid.lower(): _last_or_none(sid) for sid in series_ids}
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/data/test_fred_tools.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add FRED macro tools`. Ask first.

---

## Phase 3 - Agent Infrastructure

### Task 13: config.py - model routing and weights

**Files:**
- Create: `equity_trader/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
from equity_trader.config import AGENT_WEIGHTS, AGENT_MODELS, ORCHESTRATOR_MODEL


def test_weights_sum_to_100():
    assert abs(sum(AGENT_WEIGHTS.values()) - 100.0) < 1e-6


def test_all_seven_agents_have_models():
    expected = {"jpm_fundamental", "bridgewater_macro", "gs_technical",
                "citadel_quant", "renaissance_pattern", "de_shaw_options",
                "two_sigma_backtest"}
    assert set(AGENT_WEIGHTS.keys()) == expected
    assert set(AGENT_MODELS.keys()) == expected


def test_orchestrator_model_set():
    assert ORCHESTRATOR_MODEL.provider in {"openai", "groq", "deepseek", "gemini"}
    assert ORCHESTRATOR_MODEL.model
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/config.py
from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "groq", "deepseek", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


AGENT_MODELS: dict[str, ModelSpec] = {
    "jpm_fundamental":     ModelSpec("gemini",   "gemini-2.5-pro"),
    "bridgewater_macro":   ModelSpec("gemini",   "gemini-2.5-pro"),
    "gs_technical":        ModelSpec("deepseek", "deepseek-chat"),
    "citadel_quant":       ModelSpec("groq",     "llama-3.3-70b-versatile"),
    "renaissance_pattern": ModelSpec("groq",     "llama-3.3-70b-versatile"),
    "de_shaw_options":     ModelSpec("deepseek", "deepseek-chat"),
    "two_sigma_backtest":  ModelSpec("groq",     "llama-3.3-70b-versatile"),
}

AGENT_WEIGHTS: dict[str, float] = {
    "jpm_fundamental":     25.0,
    "bridgewater_macro":   20.0,
    "gs_technical":        15.0,
    "citadel_quant":       15.0,
    "renaissance_pattern": 10.0,
    "de_shaw_options":     10.0,
    "two_sigma_backtest":   5.0,
}

ORCHESTRATOR_MODEL = ModelSpec("openai", "gpt-4o")
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add agent model routing and weights config`. Ask first.

---

### Task 14: agents/base.py - agent factory

**Files:**
- Create: `equity_trader/agents/base.py`
- Test: `tests/agents/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agents/test_base.py
import os
import pytest
from equity_trader.config import ModelSpec
from equity_trader.agents.base import build_agent, _client_for


def test_client_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    c = _client_for(ModelSpec("openai", "gpt-4o"))
    assert c.base_url is None or "openai.com" in str(c.base_url)


def test_client_for_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    c = _client_for(ModelSpec("groq", "llama-3.3-70b-versatile"))
    assert "groq.com" in str(c.base_url)


def test_client_for_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    c = _client_for(ModelSpec("deepseek", "deepseek-chat"))
    assert "deepseek.com" in str(c.base_url)


def test_client_for_gemini_raises_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _client_for(ModelSpec("gemini", "gemini-2.5-pro"))


def test_build_agent_constructs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    a = build_agent(
        name="test_agent",
        instructions="You are a test.",
        tools=[],
        spec=ModelSpec("groq", "llama-3.3-70b-versatile"),
        output_type=dict,
    )
    assert a.name == "test_agent"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_base.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/base.py
import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel

from equity_trader.config import ModelSpec


_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _client_for(spec: ModelSpec) -> AsyncOpenAI:
    env_key = _ENV_KEYS[spec.provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} not set")
    if spec.provider == "openai":
        return AsyncOpenAI(api_key=api_key)
    return AsyncOpenAI(api_key=api_key, base_url=_BASE_URLS[spec.provider])


def build_agent(*, name: str, instructions: str, tools: list,
                spec: ModelSpec, output_type):
    client = _client_for(spec)
    model = OpenAIChatCompletionsModel(model=spec.model, openai_client=client)
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model,
        output_type=output_type,
    )
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_base.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add agent factory with multi-provider client routing`. Ask first.

---

### Task 15: Shared agent prompt skeleton helper

**Files:**
- Modify: `equity_trader/agents/base.py`
- Modify: `tests/agents/test_base.py`

- [ ] **Step 1: Add failing test**

```python
from equity_trader.agents.base import shared_rules_block


def test_shared_rules_block_mentions_horizon_and_schema():
    text = shared_rules_block()
    assert "3-6 month" in text
    assert "data_cited" in text
    assert "thesis" in text
    assert "risks" in text
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_base.py::test_shared_rules_block_mentions_horizon_and_schema -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (append to `equity_trader/agents/base.py`)

```python
def shared_rules_block() -> str:
    return (
        "HARD RULES:\n"
        "1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.\n"
        "2. Cite specific data points in `data_cited` (e.g., '10-Q Q1 revenue',\n"
        "   'DGS10 5/29 close', 'RSI 28 on 2026-06-05').\n"
        "3. Provide 1-6 thesis bullets and 1-4 concrete risks.\n"
        "4. Conviction is 1-10 where 10 = bet-the-book confidence.\n"
        "5. Output MUST conform to the AgentVerdict schema.\n"
        "6. If you genuinely lack data, return HOLD with conviction 3 and explain in risks.\n"
    )
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_base.py -v`
Expected: all PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add shared agent rules block`. Ask first.

---

## Phase 4 - Specialist Agents

Each agent task follows the same shape: define `INSTRUCTIONS`, declare tool list, build agent with `build_agent`, expose a `run(ticker) -> AgentVerdict` coroutine, unit-test with mocked LLM. Tools are passed to the SDK via the `function_tool` decorator. I'll show the pattern in Task 16 in full and then reference it concisely for 17-22.

### Task 16: JPM Fundamental agent

**Files:**
- Create: `equity_trader/agents/jpm_fundamental.py`
- Test: `tests/agents/test_jpm_fundamental.py`

- [ ] **Step 1: Write failing test (mocked LLM)**

```python
# tests/agents/test_jpm_fundamental.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import jpm_fundamental as mod


@pytest.mark.asyncio
async def test_jpm_fundamental_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="jpm_fundamental", ticker="NVDA", recommendation="BUY",
        conviction=8, price_target_6mo=215.0,
        thesis=["FCF expanding"], risks=["China"], data_cited=["10-Q"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "jpm_fundamental"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_jpm_fundamental.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/jpm_fundamental.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import edgar_tools as edg


AGENT_NAME = "jpm_fundamental"


@function_tool
def tool_get_fundamentals(ticker: str) -> dict:
    return yft.get_fundamentals(ticker)


@function_tool
def tool_get_analyst_targets(ticker: str) -> dict:
    return yft.get_analyst_targets(ticker)


@function_tool
def tool_get_latest_10k(ticker: str) -> dict:
    return edg.get_latest_10k(ticker)


@function_tool
def tool_get_latest_10q(ticker: str) -> dict:
    return edg.get_latest_10q(ticker)


@function_tool
def tool_get_recent_8k_summaries(ticker: str, n: int = 5) -> list[dict]:
    return edg.get_recent_8k_summaries(ticker, n)


INSTRUCTIONS = f"""You are a senior equity research analyst in the style of
JPM fundamental coverage. You analyze companies on revenue and FCF trajectory,
margin trends, balance sheet quality, capital allocation, and the credibility
of management guidance.

Use the EDGAR tools to read the latest 10-K and 10-Q. Use yfinance tools for
analyst targets and recent fundamentals. Form a view on the 3-6 month window
specifically - what catalysts hit in that window, what does the next quarter
look like, where does consensus sit vs your read.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_fundamentals, tool_get_analyst_targets,
           tool_get_latest_10k, tool_get_latest_10q,
           tool_get_recent_8k_summaries],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_jpm_fundamental.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit**

Stage; propose: `feat: add JPM fundamental agent`. Ask first.

---

### Task 17: Bridgewater All-Weather (macro) agent

**Files:**
- Create: `equity_trader/agents/bridgewater_macro.py`
- Test: `tests/agents/test_bridgewater_macro.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_bridgewater_macro.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import bridgewater_macro as mod


@pytest.mark.asyncio
async def test_bridgewater_macro_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="bridgewater_macro", ticker="NVDA", recommendation="HOLD",
        conviction=6, thesis=["Curve flat"], risks=["Recession"], data_cited=["FRED"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "bridgewater_macro"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_bridgewater_macro.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/bridgewater_macro.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import fred_tools as fred


AGENT_NAME = "bridgewater_macro"


@function_tool
def tool_get_macro_snapshot() -> dict:
    return fred.get_macro_snapshot()


@function_tool
def tool_get_series(series_id: str) -> list[dict]:
    s = fred.get_series(series_id)
    return [{"date": str(d.date()), "value": float(v)} for d, v in s.tail(60).items()]


@function_tool
def tool_get_sector_macro(sector: str) -> dict:
    return fred.get_sector_macro(sector)


@function_tool
def tool_get_sector_etf_history(etf: str) -> dict:
    df = yft.get_price_history(etf, period="6mo", interval="1d")
    return {"last_close": float(df["Close"].iloc[-1]),
            "ret_3m": float(df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1.0)
                       if len(df) >= 63 else None}


INSTRUCTIONS = f"""You are a macro strategist in the Bridgewater All-Weather
tradition. You classify the macro regime (growth, inflation, liquidity, risk
appetite) and apply it to the ticker via its sector and broad market
sensitivity.

For the 3-6 month horizon: where are rates, where is the curve, what does the
NFCI say about financial conditions, what does CPI YoY trajectory imply for
sector rotation. Map that read onto the ticker's sector.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_macro_snapshot, tool_get_series, tool_get_sector_macro,
           tool_get_sector_etf_history],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_bridgewater_macro.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add Bridgewater macro agent`.

---

### Task 18: GS Technical agent

**Files:**
- Create: `equity_trader/agents/gs_technical.py`
- Test: `tests/agents/test_gs_technical.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_gs_technical.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import gs_technical as mod


@pytest.mark.asyncio
async def test_gs_technical_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="gs_technical", ticker="NVDA", recommendation="BUY",
        conviction=7, price_target_6mo=208.0,
        thesis=["Above 50dma"], risks=["RSI hot"], data_cited=["RSI 62"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "gs_technical"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_gs_technical.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/gs_technical.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "gs_technical"


@function_tool
def tool_get_price_history(ticker: str, period: str = "2y") -> list[dict]:
    df = yft.get_price_history(ticker, period=period)
    out = df[["Open", "High", "Low", "Close", "Volume"]].tail(120).reset_index()
    return out.assign(Date=lambda d: d["Date"].astype(str)).to_dict(orient="records")


@function_tool
def tool_compute_technicals(ticker: str) -> dict:
    return yft.compute_technicals(ticker)


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


INSTRUCTIONS = f"""You are a technical analyst in the GS tradition. You read
trend, momentum, support/resistance, and volume. You explicitly call out the
3-6 month setup: is the stock in a base, breaking out, rolling over, or
chopping. Cite concrete levels (50dma, 200dma, recent swing high/low) and
specific indicator values (RSI, MACD position vs signal).

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_price_history, tool_compute_technicals, tool_get_quote],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_gs_technical.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add GS technical agent`.

---

### Task 19: Citadel Quant agent

**Files:**
- Create: `equity_trader/agents/citadel_quant.py`
- Test: `tests/agents/test_citadel_quant.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_citadel_quant.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import citadel_quant as mod


@pytest.mark.asyncio
async def test_citadel_quant_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="citadel_quant", ticker="NVDA", recommendation="BUY",
        conviction=6, price_target_6mo=205.0,
        thesis=["Strong momentum 12-1"], risks=["High vol"], data_cited=["mom=0.32"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "citadel_quant"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_citadel_quant.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/citadel_quant.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft
from equity_trader.data import fred_tools as fred


AGENT_NAME = "citadel_quant"


@function_tool
def tool_get_peer_set(ticker: str) -> list[str]:
    return yft.get_peer_set(ticker)


@function_tool
def tool_compute_factor_loads(ticker: str, peers: list[str]) -> dict:
    return yft.compute_factor_loads(ticker, peers)


@function_tool
def tool_get_macro_snapshot() -> dict:
    return fred.get_macro_snapshot()


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


INSTRUCTIONS = f"""You are a quantitative equity analyst in the Citadel
tradition. You score the name on factor exposures (value, quality, momentum,
low-vol) and relative value vs peers. For the 3-6 month horizon, pay attention
to momentum (12-1), realized volatility regime, and rate sensitivity.

Use the peer set returned by tool_get_peer_set when computing relative metrics.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_peer_set, tool_compute_factor_loads,
           tool_get_macro_snapshot, tool_get_quote],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_citadel_quant.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add Citadel quant agent`.

---

### Task 20: Renaissance Pattern agent

**Files:**
- Create: `equity_trader/agents/renaissance_pattern.py`
- Test: `tests/agents/test_renaissance_pattern.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_renaissance_pattern.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import renaissance_pattern as mod


@pytest.mark.asyncio
async def test_renaissance_pattern_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="renaissance_pattern", ticker="NVDA", recommendation="HOLD",
        conviction=5, thesis=["Mixed regime"], risks=["Small sample"],
        data_cited=["ac1=0.02"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "renaissance_pattern"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_renaissance_pattern.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/renaissance_pattern.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "renaissance_pattern"


@function_tool
def tool_get_price_history(ticker: str, period: str = "2y") -> list[dict]:
    df = yft.get_price_history(ticker, period=period)
    out = df[["Close", "Volume"]].tail(252).reset_index()
    return out.assign(Date=lambda d: d["Date"].astype(str)).to_dict(orient="records")


@function_tool
def tool_compute_statistical_patterns(ticker: str) -> dict:
    return yft.compute_statistical_patterns(ticker)


INSTRUCTIONS = f"""You are a statistical pattern researcher in the
Renaissance tradition. You treat price as data, not narrative. Identify
seasonality, autocorrelation regime (trending vs mean-reverting), and any
anomalies that might persist over the 3-6 month horizon.

Be rigorous about not over-fitting. If a 'pattern' relies on a handful of
samples, say so and downgrade conviction.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_price_history, tool_compute_statistical_patterns],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_renaissance_pattern.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add Renaissance pattern agent`.

---

### Task 21: DE Shaw Options agent

**Files:**
- Create: `equity_trader/agents/de_shaw_options.py`
- Test: `tests/agents/test_de_shaw_options.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_de_shaw_options.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import de_shaw_options as mod


@pytest.mark.asyncio
async def test_de_shaw_options_run_returns_agent_verdict():
    fake = AgentVerdict(
        agent="de_shaw_options", ticker="NVDA", recommendation="BUY",
        conviction=6, price_target_6mo=210.0,
        thesis=["IV cheap vs realized"], risks=["Skew elevated"],
        data_cited=["ATM IV 28%"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert isinstance(out, AgentVerdict)
        assert out.agent == "de_shaw_options"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/agents/test_de_shaw_options.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/de_shaw_options.py
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "de_shaw_options"


@function_tool
def tool_get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    return yft.get_options_chain(ticker, expiry)


@function_tool
def tool_compute_iv_stats(ticker: str) -> dict:
    return yft.compute_iv_stats(ticker)


@function_tool
def tool_get_quote(ticker: str) -> dict:
    return yft.get_quote(ticker)


@function_tool
def tool_compute_technicals(ticker: str) -> dict:
    return yft.compute_technicals(ticker)


INSTRUCTIONS = f"""You are an options-derived signals analyst in the DE Shaw
tradition. You read ATM IV, skew, and term structure to understand what
options are pricing in over the 3-6 month horizon. Compare IV to realized vol
(use compute_technicals' vol_20 as an annualized proxy). Look for asymmetric
setups.

A high put skew with elevated IV says the market is paying up for downside
protection - that's information about positioning, not just direction.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_get_options_chain, tool_compute_iv_stats,
           tool_get_quote, tool_compute_technicals],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_de_shaw_options.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add DE Shaw options agent`.

---

### Task 22: Two Sigma Backtest agent

**Files:**
- Create: `equity_trader/agents/two_sigma_backtest.py`
- Test: `tests/agents/test_two_sigma_backtest.py`

- [ ] **Step 1: Write tests for the in-module rule evaluator and the agent run**

```python
# tests/agents/test_two_sigma_backtest.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict
from equity_trader.agents import two_sigma_backtest as mod


def test_backtest_rsi_mean_reversion_returns_summary(monkeypatch):
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 500))
    df = pd.DataFrame({"Close": prices,
                        "Open": prices, "High": prices + 1, "Low": prices - 1,
                        "Volume": rng.integers(1e6, 5e6, 500)},
                       index=pd.date_range("2023-01-01", periods=500, freq="B"))
    monkeypatch.setattr(mod.yft, "get_price_history", lambda t, period="2y": df)
    summary = mod.backtest_simple_rules("AAPL")
    assert "rsi_mean_reversion" in summary
    assert "sma_crossover" in summary
    for k, v in summary.items():
        assert {"trades", "hit_rate", "avg_return"}.issubset(v.keys())


@pytest.mark.asyncio
async def test_two_sigma_backtest_run_returns_agent_verdict(monkeypatch):
    fake = AgentVerdict(
        agent="two_sigma_backtest", ticker="NVDA", recommendation="HOLD",
        conviction=4, thesis=["weak signal"], risks=["noisy"], data_cited=["bt"],
    )
    with patch.object(mod, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await mod.run("NVDA")
        assert out.agent == "two_sigma_backtest"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/two_sigma_backtest.py
import pandas as pd
import numpy as np
from agents import Runner, function_tool

from equity_trader.agents.base import build_agent, shared_rules_block
from equity_trader.config import AGENT_MODELS
from equity_trader.schemas import AgentVerdict
from equity_trader.data import yfinance_tools as yft


AGENT_NAME = "two_sigma_backtest"


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _summarize_trades(returns: pd.Series) -> dict:
    if returns.empty:
        return {"trades": 0, "hit_rate": None, "avg_return": None}
    return {
        "trades": int(len(returns)),
        "hit_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
    }


def backtest_simple_rules(ticker: str) -> dict:
    df = yft.get_price_history(ticker, period="2y")
    close = df["Close"]
    # RSI mean reversion: buy when RSI<30, hold 20 sessions
    rsi = _rsi_series(close)
    signals = rsi < 30
    rets = []
    for i in signals[signals].index:
        loc = close.index.get_loc(i)
        if loc + 20 < len(close):
            rets.append(close.iloc[loc + 20] / close.iloc[loc] - 1.0)
    rsi_summary = _summarize_trades(pd.Series(rets))

    # 50/200 SMA crossover: buy when 50dma crosses above 200dma, hold 60 sessions
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    cross = (sma50.shift() <= sma200.shift()) & (sma50 > sma200)
    rets2 = []
    for i in cross[cross].index:
        loc = close.index.get_loc(i)
        if loc + 60 < len(close):
            rets2.append(close.iloc[loc + 60] / close.iloc[loc] - 1.0)
    sma_summary = _summarize_trades(pd.Series(rets2))

    return {"rsi_mean_reversion": rsi_summary, "sma_crossover": sma_summary}


@function_tool
def tool_backtest_simple_rules(ticker: str) -> dict:
    return backtest_simple_rules(ticker)


INSTRUCTIONS = f"""You are a backtest-grounded analyst in the Two Sigma
tradition. You evaluate a small handful of simple rules on this ticker's
history and report what worked. You acknowledge that single-ticker backtests
are statistically weak - your role is a sanity check, not a primary signal.

If hit rates are 45-55%, treat as no edge. Only call BUY/SELL when a rule
shows a >60% hit rate over >=10 trades AND the most recent signal matches.

{shared_rules_block()}

Return an AgentVerdict with agent='{AGENT_NAME}'.
"""

agent = build_agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    tools=[tool_backtest_simple_rules],
    spec=AGENT_MODELS[AGENT_NAME],
    output_type=AgentVerdict,
)


async def run(ticker: str) -> AgentVerdict:
    result = await Runner.run(agent, input=f"Analyze {ticker} for a 3-6 month horizon.")
    return result.final_output
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/agents/test_two_sigma_backtest.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit** - `feat: add Two Sigma backtest agent`.

---

### Task 23: Agent registry

**Files:**
- Modify: `equity_trader/agents/__init__.py`
- Test: `tests/agents/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agents/test_registry.py
from equity_trader.agents import ALL_AGENTS, AGENT_NAMES


def test_seven_agents():
    assert len(ALL_AGENTS) == 7
    assert len(AGENT_NAMES) == 7


def test_each_entry_has_run_coroutine():
    import inspect
    for name, run_fn in ALL_AGENTS:
        assert inspect.iscoroutinefunction(run_fn), name
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/agents/__init__.py
from equity_trader.agents import (
    jpm_fundamental, bridgewater_macro, gs_technical,
    citadel_quant, renaissance_pattern, de_shaw_options,
    two_sigma_backtest,
)

ALL_AGENTS = [
    ("jpm_fundamental",     jpm_fundamental.run),
    ("bridgewater_macro",   bridgewater_macro.run),
    ("gs_technical",        gs_technical.run),
    ("citadel_quant",       citadel_quant.run),
    ("renaissance_pattern", renaissance_pattern.run),
    ("de_shaw_options",     de_shaw_options.run),
    ("two_sigma_backtest",  two_sigma_backtest.run),
]

AGENT_NAMES = [name for name, _ in ALL_AGENTS]
```

- [ ] **Step 4: Run and verify pass**

- [ ] **Step 5: Request commit** - `feat: add agent registry`.

---

## Phase 5 - Orchestrator

### Task 24: Deterministic weighted-score helper

**Files:**
- Create: `equity_trader/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring.py
import pytest
from equity_trader.schemas import AgentVerdict
from equity_trader.scoring import compute_weighted_score, redistribute_weights


def _v(agent, rec, conv=5, err=None):
    return AgentVerdict(agent=agent, ticker="T", recommendation=rec,
                        conviction=conv, thesis=["x"], risks=["y"],
                        data_cited=["z"], error_note=err)


def test_all_buy_unanimous():
    verdicts = [_v("jpm_fundamental", "BUY", 8), _v("gs_technical", "BUY", 7)]
    weights = {"jpm_fundamental": 60.0, "gs_technical": 40.0}
    score = compute_weighted_score(verdicts, weights)
    # 1*8*60 + 1*7*40 = 480 + 280 = 760; /100 = 7.6
    assert abs(score - 7.6) < 1e-6


def test_split_calls():
    verdicts = [_v("a", "BUY", 8), _v("b", "SELL", 6)]
    weights = {"a": 50.0, "b": 50.0}
    score = compute_weighted_score(verdicts, weights)
    # (1*8*50 + (-1)*6*50)/100 = (400-300)/100 = 1.0
    assert abs(score - 1.0) < 1e-6


def test_hold_contributes_zero():
    verdicts = [_v("a", "HOLD", 9)]
    score = compute_weighted_score(verdicts, {"a": 100.0})
    assert score == 0.0


def test_errored_agent_excluded_and_weights_redistributed():
    verdicts = [_v("a", "BUY", 8), _v("b", "BUY", 6, err="boom")]
    weights = {"a": 50.0, "b": 50.0}
    score = compute_weighted_score(verdicts, weights)
    # 'b' excluded; 'a' gets 100% weight: 1*8*100/100 = 8.0
    assert abs(score - 8.0) < 1e-6


def test_redistribute_weights():
    out = redistribute_weights({"a": 50.0, "b": 30.0, "c": 20.0}, excluded={"b"})
    # remaining: a=50, c=20; total=70; rescale to 100
    assert abs(out["a"] - 50.0 * 100.0 / 70.0) < 1e-6
    assert abs(out["c"] - 20.0 * 100.0 / 70.0) < 1e-6
    assert "b" not in out
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/scoring.py
from equity_trader.schemas import AgentVerdict

_SIGN = {"BUY": 1, "HOLD": 0, "SELL": -1}


def redistribute_weights(weights: dict[str, float],
                         excluded: set[str]) -> dict[str, float]:
    kept = {k: v for k, v in weights.items() if k not in excluded}
    total = sum(kept.values())
    if total <= 0:
        return kept
    factor = 100.0 / total
    return {k: v * factor for k, v in kept.items()}


def compute_weighted_score(verdicts: list[AgentVerdict],
                            weights: dict[str, float]) -> float:
    excluded = {v.agent for v in verdicts if v.error_note}
    effective = redistribute_weights(weights, excluded)
    score = 0.0
    for v in verdicts:
        if v.error_note:
            continue
        w = effective.get(v.agent, 0.0)
        score += _SIGN[v.recommendation] * v.conviction * w
    return score / 100.0
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Request commit** - `feat: add deterministic weighted-score helper`.

---

### Task 25: Orchestrator agent

**Files:**
- Create: `equity_trader/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader import orchestrator as orch


def _v(agent, rec, conv=7, tgt=200.0, err=None):
    return AgentVerdict(agent=agent, ticker="NVDA", recommendation=rec,
                        conviction=conv, price_target_6mo=tgt,
                        thesis=["t"], risks=["r"], data_cited=["d"],
                        error_note=err)


@pytest.mark.asyncio
async def test_orchestrator_run_returns_orchestrator_verdict():
    verdicts = [
        _v("jpm_fundamental", "BUY", 8, 215),
        _v("bridgewater_macro", "HOLD", 6, None),
        _v("gs_technical", "BUY", 7, 208),
        _v("citadel_quant", "BUY", 6, 205),
        _v("renaissance_pattern", "HOLD", 5, None),
        _v("de_shaw_options", "BUY", 6, 210),
        _v("two_sigma_backtest", "HOLD", 4, None),
    ]
    fake = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
        weighted_score=3.55, weights_used={"jpm_fundamental": 25.0},
        weight_overrides_rationale=None,
        synthesis="Strong fundamentals confirmed by technicals; macro acceptable for horizon.",
        key_agreements=["FCF and technicals align"],
        key_disagreements=["Macro vs fundamental on rate path"],
        dominant_drivers=["jpm_fundamental", "gs_technical"],
        red_flags=["Customer concentration"],
        position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[], agent_verdicts=verdicts,
    )
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert out.final_recommendation == "BUY"


@pytest.mark.asyncio
async def test_orchestrator_falls_back_on_llm_failure():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert "mechanical aggregation" in out.synthesis.lower()
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/orchestrator.py
import logging
from datetime import datetime
from agents import Runner

from equity_trader.agents.base import build_agent
from equity_trader.config import AGENT_WEIGHTS, ORCHESTRATOR_MODEL
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader.scoring import compute_weighted_score, redistribute_weights


log = logging.getLogger(__name__)


_INSTRUCTIONS = """You are the head of a multi-strategy investment committee.
Seven specialist agents have each analyzed the ticker. Your job is to
synthesize their structured verdicts into a final BUY / HOLD / SELL for the
3-6 month horizon.

You will be given:
- Each agent's full AgentVerdict (recommendation, conviction, target, thesis,
  risks, data_cited, error_note).
- The default weights for each agent.
- The deterministic weighted_score already computed from those weights.
- The current price of the ticker.

Hard rules:
1. Horizon is strictly 3-6 months. Reject theses pegged to longer windows.
2. You may use the default weights or override them. If you override, populate
   `weight_overrides_rationale` explaining why (e.g., 'downweighting macro
   because Fed meeting is post-horizon').
3. If your `final_recommendation` sign disagrees with the sign of
   `weighted_score`, you MUST populate `weight_overrides_rationale`.
4. Surface disagreements explicitly in `key_disagreements`. Do not paper over splits.
5. `price_target_6mo` is conviction-weighted average of agent targets. Agents
   without a target are excluded from the average.
6. You may NOT introduce new data not cited by an agent. You are a synthesizer,
   not an analyst.
7. `position_sizing_suggestion` must be one of: 'not sizeable', 'starter (~1%)',
   'full (~3%)', 'high conviction (~5%)'.
8. `stop_loss_level` is technical-anchored; populate only on BUY recommendations,
   otherwise null.
9. `catalyst_calendar` lists events within the 3-6 month window with expected_impact.

Write a 150-300 word synthesis in `synthesis` that reads like a PM memo -
direct, evidence-cited, no hedging filler.

Return an OrchestratorVerdict.
"""


_agent = build_agent(
    name="orchestrator",
    instructions=_INSTRUCTIONS,
    tools=[],
    spec=ORCHESTRATOR_MODEL,
    output_type=OrchestratorVerdict,
)


def _format_input(ticker: str, verdicts: list[AgentVerdict],
                   current_price: float, weighted_score: float) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Current price: {current_price}",
        f"Default weights: {AGENT_WEIGHTS}",
        f"Deterministic weighted_score (using defaults): {weighted_score:.4f}",
        "",
        "Agent verdicts:",
    ]
    for v in verdicts:
        lines.append(v.model_dump_json(indent=2))
    return "\n".join(lines)


def _mechanical_fallback(ticker: str, verdicts: list[AgentVerdict],
                          current_price: float,
                          weighted_score: float) -> OrchestratorVerdict:
    if weighted_score > 1.0:
        rec, sizing = "BUY", "starter (~1%)"
    elif weighted_score < -1.0:
        rec, sizing = "SELL", "not sizeable"
    else:
        rec, sizing = "HOLD", "not sizeable"
    targets = [(v.price_target_6mo, v.conviction) for v in verdicts
               if v.price_target_6mo and not v.error_note]
    tgt = None
    if targets:
        num = sum(t * c for t, c in targets)
        den = sum(c for _, c in targets)
        tgt = num / den if den else None
    return OrchestratorVerdict(
        ticker=ticker,
        run_timestamp=datetime.utcnow(),
        current_price=current_price,
        final_recommendation=rec,
        conviction=min(10, max(1, int(abs(weighted_score) + 1))),
        price_target_6mo=tgt,
        weighted_score=weighted_score,
        weights_used=AGENT_WEIGHTS,
        weight_overrides_rationale=None,
        synthesis=("LLM synthesis unavailable; mechanical aggregation below. "
                   f"Deterministic weighted score = {weighted_score:.2f} "
                   f"on default weights. Recommendation derived from score sign."),
        key_agreements=[],
        key_disagreements=[],
        dominant_drivers=[],
        red_flags=[],
        position_sizing_suggestion=sizing,
        stop_loss_level=None,
        catalyst_calendar=[],
        agent_verdicts=verdicts,
    )


async def run(ticker: str, verdicts: list[AgentVerdict],
              current_price: float) -> OrchestratorVerdict:
    weighted = compute_weighted_score(verdicts, AGENT_WEIGHTS)
    try:
        result = await Runner.run(_agent,
                                   input=_format_input(ticker, verdicts,
                                                       current_price, weighted))
        return result.final_output
    except Exception as e:
        log.exception("Orchestrator LLM failed; falling back to mechanical: %s", e)
        return _mechanical_fallback(ticker, verdicts, current_price, weighted)
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit** - `feat: add orchestrator with mechanical fallback`.

---

## Phase 6 - Runner and Persistence

### Task 26: Persistence (SQLite + JSON snapshots)

**Files:**
- Create: `equity_trader/persistence.py`
- Test: `tests/test_persistence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_persistence.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pytest
from equity_trader.schemas import OrchestratorVerdict, AgentVerdict
from equity_trader.persistence import persist


def _verdict():
    av = AgentVerdict(agent="jpm_fundamental", ticker="NVDA",
                       recommendation="BUY", conviction=8,
                       thesis=["x"], risks=["y"], data_cited=["z"])
    return OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime(2026, 6, 7, 12, 0, 0),
        current_price=190.0, final_recommendation="BUY", conviction=7,
        price_target_6mo=210.0, weighted_score=3.5,
        weights_used={"jpm_fundamental": 100.0}, weight_overrides_rationale=None,
        synthesis="A test synthesis sufficiently long to meet the minimum length.",
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[], agent_verdicts=[av],
    )


def test_persist_writes_sqlite_and_json(tmp_path):
    db = tmp_path / "et.db"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    v = _verdict()
    run_id = persist(v, db_path=str(db), runs_dir=str(runs_dir))
    assert run_id

    con = sqlite3.connect(db)
    row = con.execute("SELECT ticker, final_recommendation, snapshot_path FROM runs").fetchone()
    assert row[0] == "NVDA"
    assert row[1] == "BUY"
    snap = Path(row[2])
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["ticker"] == "NVDA"
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/persistence.py
import json
import sqlite3
import uuid
from pathlib import Path
from equity_trader.schemas import OrchestratorVerdict


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    current_price REAL NOT NULL,
    final_recommendation TEXT NOT NULL,
    conviction INTEGER NOT NULL,
    price_target_6mo REAL,
    weighted_score REAL NOT NULL,
    snapshot_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_ticker_ts ON runs(ticker, run_timestamp DESC);
"""


def _ensure_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()


def persist(verdict: OrchestratorVerdict,
             db_path: str = "equity_trader.db",
             runs_dir: str = "runs") -> str:
    _ensure_schema(db_path)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    ts = verdict.run_timestamp.strftime("%Y-%m-%d_%H%M")
    snap_path = Path(runs_dir) / f"{verdict.ticker}_{ts}.json"
    snap_path.write_text(verdict.model_dump_json(indent=2))

    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs (run_id, ticker, run_timestamp, current_price, "
            "final_recommendation, conviction, price_target_6mo, weighted_score, "
            "snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, verdict.ticker, verdict.run_timestamp.isoformat(),
             verdict.current_price, verdict.final_recommendation,
             verdict.conviction, verdict.price_target_6mo, verdict.weighted_score,
             str(snap_path)),
        )
        con.commit()
    finally:
        con.close()
    return run_id


def list_runs(db_path: str = "equity_trader.db",
               ticker: str | None = None,
               limit: int = 50) -> list[dict]:
    _ensure_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        if ticker:
            rows = con.execute(
                "SELECT * FROM runs WHERE ticker=? ORDER BY run_timestamp DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY run_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_persistence.py -v`
Expected: 1 PASS.

- [ ] **Step 5: Request commit** - `feat: add SQLite + JSON persistence`.

---

### Task 27: Runner - parallel fan-out

**Files:**
- Create: `equity_trader/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_runner.py
import pytest
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader import runner as run_mod


def _v(name, rec="BUY", conv=6, err=None):
    return AgentVerdict(agent=name, ticker="NVDA", recommendation=rec,
                        conviction=conv, thesis=["x"], risks=["y"],
                        data_cited=["z"], error_note=err)


@pytest.mark.asyncio
async def test_analyze_ticker_calls_all_agents_then_orchestrator(monkeypatch, tmp_path):
    from datetime import datetime
    from equity_trader.agents import AGENT_NAMES

    async def fake_agent_run(t):
        return _v(fake_agent_run._name)

    fake_funcs = []
    for n in AGENT_NAMES:
        f = AsyncMock(side_effect=lambda t, _n=n: _v(_n))
        fake_funcs.append((n, f))
    monkeypatch.setattr(run_mod, "ALL_AGENTS", fake_funcs)

    fake_quote = lambda t: {"last_price": 190.0, "market_cap": 0, "day_high": 0, "day_low": 0}
    monkeypatch.setattr(run_mod, "get_quote", fake_quote)

    fake_orch = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
        weighted_score=3.5, weights_used={}, weight_overrides_rationale=None,
        synthesis="x" * 60,
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[],
        agent_verdicts=[_v(n) for n in AGENT_NAMES],
    )
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value=fake_orch))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")

    out = await run_mod.analyze_ticker("NVDA")
    assert isinstance(out, OrchestratorVerdict)
    for _, fn in fake_funcs:
        fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_ticker_survives_one_agent_failure(monkeypatch):
    from datetime import datetime
    from equity_trader.agents import AGENT_NAMES

    async def good(t):
        return _v("g")

    async def bad(t):
        raise RuntimeError("boom")

    fake_funcs = [(n, good if i > 0 else bad) for i, n in enumerate(AGENT_NAMES)]
    monkeypatch.setattr(run_mod, "ALL_AGENTS", fake_funcs)
    monkeypatch.setattr(run_mod, "get_quote",
                         lambda t: {"last_price": 190.0, "market_cap": 0,
                                    "day_high": 0, "day_low": 0})

    fake_orch = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="HOLD", conviction=5, price_target_6mo=None,
        weighted_score=1.0, weights_used={}, weight_overrides_rationale=None,
        synthesis="x" * 60,
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="not sizeable",
        stop_loss_level=None, catalyst_calendar=[],
        agent_verdicts=[],
    )
    monkeypatch.setattr(run_mod.orchestrator, "run", AsyncMock(return_value=fake_orch))
    monkeypatch.setattr(run_mod, "persist", lambda v, **kw: "run-id")

    out = await run_mod.analyze_ticker("NVDA")
    assert isinstance(out, OrchestratorVerdict)
```

- [ ] **Step 2: Run and verify failure**

- [ ] **Step 3: Implement**

```python
# equity_trader/runner.py
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
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Request commit** - `feat: add runner with parallel fan-out and graceful failure`.

---

## Phase 7 - Interfaces

### Task 28: CLI (`run.py`)

**Files:**
- Create: `run.py`

- [ ] **Step 1: Implement** (no unit test; smoke-tested in Phase 8)

```python
# run.py
import argparse
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from equity_trader.runner import analyze_ticker

console = Console()


def render(v) -> None:
    table = Table(title=f"{v.ticker}  ·  {v.final_recommendation}  ·  Conviction {v.conviction}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Current price", f"${v.current_price:,.2f}")
    if v.price_target_6mo:
        table.add_row("6mo target", f"${v.price_target_6mo:,.2f}")
    table.add_row("Weighted score", f"{v.weighted_score:.2f}")
    table.add_row("Position sizing", v.position_sizing_suggestion)
    if v.stop_loss_level:
        table.add_row("Stop loss", f"${v.stop_loss_level:,.2f}")
    console.print(table)
    console.print(Panel(v.synthesis, title="Synthesis"))
    if v.key_agreements:
        console.print("[bold]Agreements:[/]")
        for a in v.key_agreements:
            console.print(f"  - {a}")
    if v.key_disagreements:
        console.print("[bold]Disagreements:[/]")
        for d in v.key_disagreements:
            console.print(f"  - {d}")
    if v.red_flags:
        console.print("[bold red]Red flags:[/]")
        for r in v.red_flags:
            console.print(f"  - {r}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    with console.status(f"Analyzing {args.ticker.upper()}..."):
        v = asyncio.run(analyze_ticker(args.ticker.upper()))
    render(v)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Static check**

Run: `uv run python -c "import run; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Request commit** - `feat: add CLI entry point`.

---

### Task 29: Gradio UI (`app.py`)

**Files:**
- Create: `app.py`

- [ ] **Step 1: Implement**

```python
# app.py
import asyncio
import gradio as gr
from dotenv import load_dotenv

from equity_trader.runner import analyze_ticker
from equity_trader.persistence import list_runs
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict

load_dotenv()


def _agent_md(v: AgentVerdict) -> str:
    target = f"${v.price_target_6mo:,.2f}" if v.price_target_6mo else "-"
    err = f"\n\n_Error: {v.error_note}_" if v.error_note else ""
    thesis = "\n".join(f"- {t}" for t in v.thesis)
    risks = "\n".join(f"- {r}" for r in v.risks)
    return (
        f"### {v.agent}\n"
        f"**{v.recommendation}**  ·  Conviction {v.conviction}/10  ·  Target {target}\n\n"
        f"**Thesis:**\n{thesis}\n\n"
        f"**Risks:**\n{risks}{err}"
    )


def _orch_md(v: OrchestratorVerdict) -> str:
    target = f"${v.price_target_6mo:,.2f}" if v.price_target_6mo else "-"
    stop = f"${v.stop_loss_level:,.2f}" if v.stop_loss_level else "-"
    agreements = "\n".join(f"- {a}" for a in v.key_agreements) or "_(none)_"
    disagree = "\n".join(f"- {d}" for d in v.key_disagreements) or "_(none)_"
    drivers = ", ".join(v.dominant_drivers) or "_(none)_"
    red = "\n".join(f"- {r}" for r in v.red_flags) or "_(none)_"
    catalysts = "\n".join(
        f"- **{c.date}** {c.event} ({c.expected_impact})" for c in v.catalyst_calendar
    ) or "_(none)_"
    return (
        f"# {v.ticker}  ·  {v.final_recommendation}  ·  Conviction {v.conviction}/10\n"
        f"**Current:** ${v.current_price:,.2f}  ·  **6mo Target:** {target}  ·  "
        f"**Weighted score:** {v.weighted_score:.2f}\n\n"
        f"## Synthesis\n{v.synthesis}\n\n"
        f"**Position sizing:** {v.position_sizing_suggestion}  ·  **Stop:** {stop}\n\n"
        f"**Dominant drivers:** {drivers}\n\n"
        f"### Agreements\n{agreements}\n\n"
        f"### Disagreements\n{disagree}\n\n"
        f"### Red flags\n{red}\n\n"
        f"### Catalysts\n{catalysts}"
    )


async def analyze(ticker: str):
    if not ticker:
        return "Enter a ticker.", *(["" for _ in range(7)])
    v = await analyze_ticker(ticker.upper())
    by_agent = {av.agent: av for av in v.agent_verdicts}
    cards = [_agent_md(by_agent[n]) if n in by_agent else ""
             for n in ["jpm_fundamental", "bridgewater_macro", "gs_technical",
                      "citadel_quant", "renaissance_pattern", "de_shaw_options",
                      "two_sigma_backtest"]]
    return _orch_md(v), *cards


def history_table():
    rows = list_runs(limit=50)
    return [[r["ticker"], r["run_timestamp"], r["final_recommendation"],
             r["conviction"], r["weighted_score"]] for r in rows]


with gr.Blocks(title="EquityTrader") as demo:
    gr.Markdown("# EquityTrader")
    with gr.Tab("Analyze"):
        with gr.Row():
            ticker = gr.Textbox(label="Ticker", placeholder="NVDA", scale=4)
            run_btn = gr.Button("Run", variant="primary", scale=1)
        with gr.Row():
            jpm = gr.Markdown()
            bw = gr.Markdown()
            gst = gr.Markdown()
        with gr.Row():
            cq = gr.Markdown()
            rp = gr.Markdown()
            dso = gr.Markdown()
        with gr.Row():
            tsb = gr.Markdown()
        gr.Markdown("---")
        orch_out = gr.Markdown()
        run_btn.click(analyze, inputs=ticker,
                      outputs=[orch_out, jpm, bw, gst, cq, rp, dso, tsb])

    with gr.Tab("History"):
        gr.Dataframe(value=history_table,
                     headers=["Ticker", "Timestamp", "Rec", "Conviction", "Score"],
                     every=10)


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Static check**

Run: `uv run python -c "import app; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Request commit** - `feat: add Gradio UI`.

---

## Phase 8 - End-to-End Smoke

### Task 30: Real run against AAPL

**Files:** none new.

- [ ] **Step 1: Confirm `.env` populated**

Verify all five keys exist: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `FRED_API_KEY`, plus `SEC_EDGAR_USER_AGENT_EMAIL`.

- [ ] **Step 2: Run unit suite end-to-end**

Run: `uv run pytest -v -m "not slow"`
Expected: full green.

- [ ] **Step 3: Run integration data tests** (real APIs)

Run: `uv run pytest -v -m slow`
Expected: green; AAPL price history fetched, real EDGAR call, real FRED call.

- [ ] **Step 4: CLI smoke**

Run: `uv run python run.py AAPL`
Expected: completes in 15-60 seconds; prints orchestrator verdict; SQLite row + JSON snapshot created. Verify with:
- `uv run python -c "from equity_trader.persistence import list_runs; print(list_runs())"`
- `ls runs/`

- [ ] **Step 5: Gradio smoke**

Run: `uv run python app.py`
Open `http://127.0.0.1:7860/`, enter `NVDA`, click Run. Confirm all 7 agent cards render, orchestrator card renders at bottom, History tab shows the run.

- [ ] **Step 6: Request commit** (any small fixes from smoke) - propose: `chore: post-smoke fixes`. Ask first.

- [ ] **Step 7: Done.**
