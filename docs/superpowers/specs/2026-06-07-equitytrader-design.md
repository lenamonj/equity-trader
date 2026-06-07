# EquityTrader Design Spec

**Date:** 2026-06-07
**Owner:** jlenamon
**Status:** Approved for implementation planning

## Purpose

A multi-agent equity research system that produces a single Buy / Hold / Sell verdict for a given ticker over a 3-6 month horizon. Seven specialist trader agents analyze the ticker in parallel from distinct angles. An orchestrator synthesizes their structured outputs into a final recommendation with conviction, price target, position sizing, stop level, and catalyst calendar.

Primary use case: help a portfolio manager make well-rounded equity decisions by surfacing where independent specialist views converge and disagree.

## Goals

- Single ticker in, structured verdict out, in 15-30 seconds.
- Each specialist agent reasons independently with its own tools and model.
- Orchestrator synthesis is auditable (deterministic weighted score) and judgmental (LLM override allowed with explicit rationale).
- All runs persisted for later review and future backtest scoring.
- Free market data only.

## Non-Goals (v1)

- Backtesting harness (data captured; scoring is a later script).
- Inter-agent debate / revision rounds.
- News sentiment ingestion (defer to Finnhub free tier if a gap appears).
- Multi-ticker watchlist UI (CLI scriptable; no UI).
- Position-tracking integration (Aladdin, Bloomberg).
- Prompt eval / regression framework.

## Architecture

Pattern: parallel fan-out with tool-level caching, then single orchestrator synthesis.

```
                ticker
                  |
        reset per-run cache
                  |
         asyncio.gather(
           jpm_fundamental,
           bridgewater_macro,
           gs_technical,
           citadel_quant,
           renaissance_pattern,
           de_shaw_options,
           two_sigma_backtest,
         )
                  |
         7 AgentVerdict objects
                  |
            Orchestrator
                  |
          OrchestratorVerdict
                  |
          persist (SQLite + JSON)
                  |
           return to UI / CLI
```

### Framework

OpenAI Agents SDK. Reasons:

- Smallest jump from the user's existing OpenAI SDK usage.
- Fan-out / fan-in maps cleanly to `asyncio.gather` over independent `Agent` instances.
- Native tool calling and structured output via Pydantic.
- Built-in tracing for debugging cross-agent reasoning.
- Accepts OpenAI-compatible endpoints, so each agent can use Groq, DeepSeek, or Gemini via the same client surface with its own `base_url`.

### Model Routing

Each agent uses the model that fits its workload. The user already holds keys for all of these.

| Agent | Model | Reason |
|---|---|---|
| JPM Fundamental | Gemini 2.5 Pro | Long context for full 10-K / 10-Q ingestion |
| Bridgewater Macro | Gemini 2.5 Pro | Long context for macro series and regime narrative |
| GS Technical | DeepSeek | Strong math, low cost for indicator reasoning |
| Citadel Quant | Groq Llama 3.3 70B | Fast and cheap for factor / peer comparison |
| Renaissance Pattern | Groq Llama 3.3 70B | Fast and cheap for pattern / statistics reasoning |
| DE Shaw Options | DeepSeek | Math-heavy IV / skew / term structure analysis |
| Two Sigma Backtest | Groq Llama 3.3 70B | Lightweight rule evaluation |
| Orchestrator | GPT-4o | Final judgment quality matters most |

Model assignment lives in `config.py` as a dict so it can be tuned without touching agent code.

### Project Structure

```
EquityTrader/
  .env
  pyproject.toml
  app.py                          # Gradio entry point
  run.py                          # CLI entry point
  equity_trader/
    __init__.py
    config.py                     # model routing, agent weights, settings
    schemas.py                    # Pydantic: AgentVerdict, OrchestratorVerdict, RunRecord
    data/
      __init__.py
      cache.py                    # per-run memoization (ContextVar)
      yfinance_tools.py
      edgar_tools.py
      fred_tools.py
    agents/
      __init__.py
      base.py                     # shared agent factory
      citadel_quant.py
      two_sigma_backtest.py
      bridgewater_macro.py
      renaissance_pattern.py
      gs_technical.py
      jpm_fundamental.py
      de_shaw_options.py
    orchestrator.py
    runner.py
    persistence.py
  runs/                           # JSON snapshots (gitignored)
  equity_trader.db                # SQLite (gitignored)
  tests/
    test_schemas.py
    test_data_tools.py
    test_agents.py
    test_orchestrator.py
```

One module per agent. Tuning one persona touches one file. Data tools are shared infrastructure.

## Agent Specifications

Weights are defaults the orchestrator sees. Sum to 100. Orchestrator may override with rationale.

### JPM Fundamental (weight 25)

- **Model:** Gemini 2.5 Pro
- **Tools:** `get_latest_10k`, `get_latest_10q`, `get_recent_8k_summaries`, `get_filing_text`, `get_fundamentals`, `get_analyst_targets`
- **Focus:** Revenue and FCF trajectory, margin trends, balance sheet quality, capital allocation, guidance credibility.
- **Output:** `AgentVerdict` with thesis grounded in cited filing sections and fundamental data points.

### Bridgewater All-Weather (weight 20)

- **Model:** Gemini 2.5 Pro
- **Tools:** `get_macro_snapshot`, `get_series`, `get_sector_macro`, sector ETF prices via `get_price_history`
- **Focus:** Macro regime, rate path, recession indicators, sector rotation read applied to the ticker.

### GS Technical (weight 15)

- **Model:** DeepSeek
- **Tools:** `get_price_history`, `compute_technicals`
- **Focus:** Trend, support / resistance, momentum, breakout or breakdown setups over the 3-6 month horizon.

### Citadel Quant (weight 15)

- **Model:** Groq Llama 3.3 70B
- **Tools:** `get_price_history` (multi-ticker), `get_peer_set`, `compute_factor_loads`, `get_macro_snapshot` (rates)
- **Focus:** Factor exposures (value, quality, momentum, low-vol), peer relative value, beta, idiosyncratic vol.

### Renaissance Pattern (weight 10)

- **Model:** Groq Llama 3.3 70B
- **Tools:** `get_price_history`, `compute_statistical_patterns`
- **Focus:** Statistical patterns, mean reversion vs trend regime, anomaly detection. Treats price as data, not narrative.

### DE Shaw Options (weight 10)

- **Model:** DeepSeek
- **Tools:** `get_options_chain`, `compute_iv_stats`, `get_price_history`
- **Focus:** IV vs realized vol, skew, term structure, what options are pricing in for 3-6 months. Flags asymmetric setups.

### Two Sigma Backtest (weight 5)

- **Model:** Groq Llama 3.3 70B
- **Tools:** `get_price_history`, simple rule evaluator (in-module helper, not LLM-called)
- **Focus:** Applies a handful of simple rules (RSI mean reversion, 50/200 dma crossover, breakout) to this ticker's history and reports historical hit rate. Sanity check only.

### Shared Agent Prompt Skeleton

Every agent prompt includes:

1. Persona description.
2. Hard rules:
   - Horizon is strictly 3-6 months.
   - Must cite specific data points in `data_cited`.
   - Must list 2-3 concrete risks.
   - Must produce structured `AgentVerdict` output (no free-form text outside the schema).
3. Tool descriptions and guidance on when to call each.
4. Output schema reminder.

## Orchestrator Specification

### Input

- 7 `AgentVerdict` objects (may include errored placeholders).
- Default weights dict from `config.py`.
- Current ticker price from yfinance.

### Process

1. **Deterministic preprocessing.** Build a summary table the LLM sees. Compute `weighted_score`:

   ```
   score = sum(
     sign(verdict.recommendation) * verdict.conviction * weights[agent_name]
     for verdict in non_errored_verdicts
   )
   ```

   where `sign(BUY) = +1`, `sign(HOLD) = 0`, `sign(SELL) = -1`. Errored agents are excluded and their weight is redistributed proportionally across the survivors.

2. **LLM synthesis.** GPT-4o receives the table, each agent's full thesis + risks + `data_cited`, the weights, and the current price. Produces `OrchestratorVerdict`.

### Hard Rules in Orchestrator Prompt

1. 3-6 month horizon. Reject theses pegged to longer time frames unless they map to this window.
2. If `final_recommendation` sign disagrees with `weighted_score` sign, must populate `weight_overrides_rationale`.
3. Surface disagreements explicitly in `key_disagreements`. Do not paper over splits.
4. `price_target_6mo` is conviction-weighted average of agent targets. Agents without a target are excluded from the average.
5. May not introduce new data not cited by an agent. Synthesizer, not analyst.
6. Position sizing suggestion uses categories: "not sizeable", "starter (~1%)", "full (~3%)", "high conviction (~5%)". User adjusts to their book.
7. Stop loss level should be technical-anchored (e.g., below recent swing low or key moving average), populated only on BUY recommendations.

### Output Schema (informative; final shape in `schemas.py`)

```python
class OrchestratorVerdict(BaseModel):
    ticker: str
    run_timestamp: datetime
    current_price: float
    final_recommendation: Literal["BUY", "HOLD", "SELL"]
    conviction: int  # 1-10
    price_target_6mo: float | None
    weighted_score: float
    weights_used: dict[str, float]
    weight_overrides_rationale: str | None
    synthesis: str  # 150-300 words
    key_agreements: list[str]
    key_disagreements: list[str]
    dominant_drivers: list[str]
    red_flags: list[str]
    position_sizing_suggestion: Literal[
        "not sizeable", "starter (~1%)", "full (~3%)", "high conviction (~5%)"
    ]
    stop_loss_level: float | None
    catalyst_calendar: list[CatalystEvent]
    agent_verdicts: list[AgentVerdict]
```

`AgentVerdict` includes: `agent`, `ticker`, `recommendation`, `conviction`, `price_target_6mo`, `thesis`, `risks`, `data_cited`, `error_note`.

## Data Layer

### `yfinance_tools.py`

```
get_price_history(ticker, period="2y", interval="1d") -> DataFrame
get_quote(ticker) -> dict
get_fundamentals(ticker) -> dict
get_analyst_targets(ticker) -> dict
get_options_chain(ticker, expiry=None) -> dict
get_peer_set(ticker) -> list[str]
compute_technicals(ticker) -> dict          # RSI, MACD, 50/200dma, ATR, volume profile
compute_statistical_patterns(ticker) -> dict # seasonality, autocorrelation, regime stats
compute_iv_stats(ticker) -> dict
compute_factor_loads(ticker, peers) -> dict
```

Raises `TickerNotFoundError`, `DataFetchError` on failure. 0.1 second sleep between consecutive calls within a run as a good-citizen rate limit.

### `edgar_tools.py`

```
get_latest_10k(ticker) -> dict
get_latest_10q(ticker) -> dict
get_recent_8k_summaries(ticker, n=5) -> list[dict]
get_filing_text(accession_no, section) -> str
```

Resolves ticker to CIK via EDGAR's ticker map JSON. Uses `User-Agent: EquityTrader jlenamon@gmail.com` per SEC requirement (email pulled from env). Honors the 10 req/sec limit.

### `fred_tools.py`

```
get_series(series_id, start=None) -> Series
get_macro_snapshot() -> dict   # DGS10, DGS2, T10Y2Y, CPI YoY, UNRATE, FEDFUNDS, NFCI, VIX
get_sector_macro(sector) -> dict
```

Uses `FRED_API_KEY` from env.

### `cache.py`

ContextVar holds per-run cache dict. `runner.py` resets at start of each ticker run. Cache key = `(func.__name__, args, frozenset(kwargs.items()))`. Memory only, no disk. Concurrent runs are isolated by ContextVar semantics.

## Runner

```python
async def analyze_ticker(ticker: str) -> OrchestratorVerdict:
    reset_run_cache()
    verdicts = await asyncio.gather(
        *(agent.run(ticker) for agent in ALL_AGENTS),
        return_exceptions=True,
    )
    verdicts = [
        v if isinstance(v, AgentVerdict) else error_verdict(agent, v)
        for agent, v in zip(ALL_AGENTS, verdicts)
    ]
    final = await orchestrator.run(ticker, verdicts)
    persist(final)
    return final
```

Used identically by Gradio and CLI. One bad agent never kills a run.

## Interface

### Gradio (`app.py`)

- Input: ticker text box, Run button.
- Output: 7 agent cards in a 3-column grid (4 + 3 layout), streaming in as each finishes. Each card shows recommendation, conviction, 6-month target, thesis bullets, risk bullets.
- Below the grid: orchestrator card prominent, with recommendation + conviction + target at top, synthesis memo, drivers, disagreements, red flags, position sizing, stop, catalyst calendar.
- Second tab: History, lists past runs from SQLite, click to reload a JSON snapshot.

### CLI (`run.py`)

`python run.py TICKER` runs the same engine. Rich-formatted progress indicator per agent, then formatted final verdict. Suitable for cron over a watchlist.

## Persistence

### SQLite (`equity_trader.db`)

```sql
CREATE TABLE runs (
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
CREATE INDEX idx_runs_ticker_ts ON runs(ticker, run_timestamp DESC);
```

### JSON snapshots

Full `OrchestratorVerdict` dumped to `runs/{TICKER}_{YYYY-MM-DD_HHMM}.json` for human inspection and future backtest scoring.

## Error Handling

- **Data layer:** Tools raise specific exceptions. No bare excepts, no silent fallbacks.
- **Agent layer:** LLM call failure after Agents SDK retries returns an `AgentVerdict` with `recommendation="HOLD"`, `conviction=0`, and `error_note` populated. Orchestrator excludes from `weighted_score` and redistributes weight.
- **Orchestrator:** On LLM failure, fall back to deterministic `weighted_score` with templated synthesis ("LLM synthesis unavailable; mechanical aggregation below..."). Run still completes and persists.
- All errors logged to console with stack trace.

## Testing

| Layer | Type | Notes |
|---|---|---|
| `schemas.py` | Unit | Pydantic validation of all field constraints. |
| `data/cache.py` | Unit | Hit / miss, reset between runs, ContextVar isolation across concurrent runs. |
| `data/*_tools.py` | Integration, marked `slow` | Hit real APIs with stable tickers (AAPL, NVDA). Skipped in default `pytest` run. |
| `agents/*` | Unit, mocked LLM | Freeze tool outputs, mock model client, assert well-formed `AgentVerdict`. Prompt quality not unit-tested. |
| `orchestrator.py` | Unit | Frozen `AgentVerdict` inputs, assert `weighted_score` math, assert override path requires rationale. |
| `runner.py` | Integration, mocked LLM | Full fan-out with mocked agents, assert one failed agent does not break the run, assert persistence writes SQLite row and JSON file. |
| End-to-end | Manual smoke | `python run.py AAPL` against real APIs and real LLMs before each commit. |

## Dependencies (`pyproject.toml`)

```
openai-agents
openai
google-genai
yfinance
httpx                       # direct calls to data.sec.gov (lighter than sec-edgar-downloader)
fredapi
pandas
numpy
pydantic
gradio
rich
python-dotenv
pytest
pytest-asyncio
respx                       # httpx mocking for tests
```

Managed with `uv`.

## Gitignore Additions

```
runs/
equity_trader.db
.env
```

## Out of Scope (v1)

- Backtesting harness.
- Inter-agent debate or revision rounds.
- News sentiment ingestion.
- Multi-ticker watchlist UI.
- Aladdin / Bloomberg integration.
- Prompt evaluation framework.
