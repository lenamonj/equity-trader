# EquityTrader design

This document describes the system as it is built. The code is the source of truth; file references point to where each rule lives.

## Purpose

EquityTrader produces one Buy, Hold or Sell verdict for a single ticker over a 3 to 6 month horizon. Eight specialist agents analyze the ticker independently, each from its own discipline. An orchestrator reads their structured verdicts and writes the final call with a conviction, a price target, a position size, a stop level and a catalyst calendar.

The point of the design is to show a portfolio manager where independent views converge and where they split, with every number traceable to code or to a dated data source.

## Run flow

```
ticker
  |
reset the per-run cache, fetch the quote        runner.py
  |
eight specialists, three at a time              runner.py, agents/
  |
eight AgentVerdict objects
  |
weighted score, computed in code                scoring.py
  |
orchestrator memo and final call (one LLM call) orchestrator.py
  |
deterministic fields stamped over the output    orchestrator.py
  |
SQLite row + JSON snapshot                      persistence.py
  |
web UI or terminal report                       app.py, run.py
```

`analyze_ticker(ticker, on_verdict=None)` in `runner.py` is the single entry point. The web UI and the CLI both call it, and both use `on_verdict` to show each desk's call as it arrives. A run takes about a minute and a half.

## The committee

Each desk is a standalone OpenAI Agents SDK agent with `AgentVerdict` as its output type. No desk sees another's work. Weights sum to 100 and are fixed in `config.py`.

| Desk | Module | Weight | Tools |
|---|---|---:|---|
| J.P. Morgan, Fundamental | `jpm_fundamental` | 25 | SEC financial statements, valuation multiples, analyst targets, recent 8-Ks |
| Bridgewater Associates, All-Weather Macro | `bridgewater_macro` | 18 | macro snapshot, FRED series, sector macro, sector ETF history |
| Goldman Sachs, Technical | `gs_technical` | 12 | price history, technicals, quote |
| Citadel, Quantitative | `citadel_quant` | 12 | peer set, factor loads, macro snapshot, quote |
| Jane Street, ETF & Basket Flow | `jane_street_etf` | 10 | ETF landscape, quote, price history |
| D. E. Shaw, Options-Derived | `de_shaw_options` | 10 | options chain, IV stats, quote, technicals |
| Renaissance Technologies, Pattern Recognition | `renaissance_pattern` | 8 | price history, statistical patterns |
| Two Sigma, Backtest Sanity | `two_sigma_backtest` | 5 | rule backtest |

Every prompt is a persona and framework followed by the shared rules in `agents/base.py`: a strict 3 to 6 month horizon, cited data points, thesis and risks on every call, one call per tool, and `error_note` left null.

The Two Sigma backtest runs in code, not in the model. It tests two fixed rules on five years of history (RSI below 30 held 20 sessions, 50/200-day crossover held 60 sessions) with non-overlapping trades, and reports each rule next to the unconditional baseline return for the same holding period.

## Models

| Role | Model | API |
|---|---|---|
| Eight specialists | `gpt-6-luna` | OpenAI Responses |
| Orchestrator | `gpt-6.1-sol` | OpenAI Responses |

Routing is one `ModelSpec(provider, model)` per agent in `config.py`. `build_agent` in `agents/base.py` uses the Responses API for OpenAI, because OpenAI's current models accept function tools only there, and the Chat Completions API for Groq, DeepSeek and Gemini through their OpenAI-compatible endpoints. Those three providers are supported by the factory and need their own key in `.env`; none is used by the default configuration.

## The verdict contract

Defined in `schemas.py`.

`AgentVerdict`: `agent`, `ticker`, `recommendation` (BUY, HOLD or SELL), `conviction` (1 to 10), `price_target_6mo` (optional), `thesis` (up to 6), `risks` (up to 4), `data_cited`, `error_note`.

`OrchestratorVerdict`: `ticker`, `run_timestamp`, `current_price`, `final_recommendation`, `conviction`, `price_target_6mo`, `weighted_score`, `weights_used`, `score_departure_rationale`, `synthesis`, `key_agreements`, `key_disagreements`, `dominant_drivers`, `red_flags`, `position_sizing_suggestion`, `stop_loss_level`, `catalyst_calendar`, `agent_verdicts`.

`error_note` belongs to the runner. It marks a desk that failed and removes it from the score. If a model writes a caveat there, the runner moves it into `risks` and clears the field.

## Scoring and what the model cannot change

```
score = sum( sign(recommendation) * conviction * weight ) / 100
```

`sign` is +1 for BUY, 0 for HOLD, -1 for SELL, so the score runs from -10 to +10. A failed desk is excluded and its weight is redistributed pro rata across the rest.

The orchestrator model writes the memo, the agreements and splits, the red flags, the position size, the stop level, the final recommendation and its conviction. The final call is a judgment and may depart from the score. A call against the score's sign must come with `score_departure_rationale`; without one, a red flag is added in code.

After the LLM call, `_stamp_deterministic` overwrites these fields with values computed in code:

| Field | Source |
|---|---|
| `ticker`, `current_price` | the runner's quote |
| `run_timestamp` | UTC clock |
| `weighted_score` | `scoring.compute_weighted_score` |
| `price_target_6mo` | conviction-weighted average of desk targets |
| `weights_used` | fixed weights after redistribution |
| `agent_verdicts` | the eight verdicts as received |
| `dominant_drivers` | filtered to valid desk names |
| `catalyst_calendar` | sourced catalysts merged in, anything outside today to today + 180 days dropped |

## Data layer

All market data is free. Every tool is memoized per run by `cached_for_run` (`data/cache.py`), a `ContextVar` cache reset at the start of each run, so eight agents share one fetch and concurrent runs stay isolated.

| Module | Provides |
|---|---|
| `yfinance_tools.py` | price history, quote, valuation multiples, analyst targets, technicals, statistical patterns, options chain and IV stats, peer set, factor loads, ETF landscape |
| `edgar_tools.py` | financial statements as filed (XBRL company facts), item codes for recent 8-Ks |
| `fred_tools.py` | macro snapshot (DGS10, DGS2, T10Y2Y, CPI year over year, UNRATE, FEDFUNDS, NFCI, VIX), single series, sector series |
| `calendar_tools.py` | upcoming earnings and dividend dates inside the horizon |

Financial statements come from the SEC, not from a market data vendor. `get_financial_statements` reads the XBRL company facts for a fixed set of lines (revenue, gross profit, operating income, pretax income, net income, diluted EPS, operating cash flow, capex, buybacks, cash, current assets and liabilities, total assets and liabilities, long-term debt, equity) over the last four fiscal years and five quarters.

- Each period carries its period end, form and first filing date. Values reflect any later restatement.
- Companies file different XBRL tags for the same line. The candidate tag reported most recently is used, and the output names it.
- Filings give cash flow items and fourth quarters only as year-to-date figures. A single quarter is the difference between consecutive year-to-date figures. Per-share figures are never derived.
- A company with no US-GAAP annual statements on EDGAR (IFRS filers, funds) falls back to yfinance statements. The `source` field in the tool output says which was used.

Rules the data layer holds to:

- Every reading carries its observation date, so a stale number is visible.
- The options chain uses the first expiry at least 90 days out and reports days to expiry.
- Peer relative strength is measured over the same 90 sessions for the ticker and each peer.
- An unknown ticker raises `TickerNotFoundError` before any LLM call is made.
- Failures are logged or raised, never swallowed.

## Failure handling

- **Bad ticker:** `get_quote` raises, the run stops, the UI shows an error card and the CLI exits 1. No LLM cost is incurred.
- **Rate limits:** a semaphore runs three agents at a time. An OpenAI 429 is retried after 8, 16 and 32 seconds.
- **Failed desk:** any other exception, or a fourth 429, becomes a HOLD placeholder with `error_note` set. The desk is shown as failed and excluded from the score.
- **Failed orchestrator:** a mechanical verdict takes its place: BUY above +1, SELL below -1, HOLD otherwise, with a memo that says the synthesis was unavailable.

## Interfaces

**Web (`app.py`, `equity_trader/ui.css`).** Gradio. The Analyze tab takes a ticker and streams a loading card that fills in desk by desk, then the report: verdict card, memo, agreements and splits, red flags, sizing, catalysts and eight specialist cards. The History tab lists past runs from SQLite and refreshes when opened.

**Terminal (`run.py`).** `uv run python run.py TICKER` shows a progress line, then the same report with Rich.

## Persistence

Each run writes one row to the `runs` table in `equity_trader.db` (ticker, timestamp, price, recommendation, conviction, target, score, snapshot path) and the full `OrchestratorVerdict` to `runs/{TICKER}_{YYYY-MM-DD_HHMMSS}.json`. Both are gitignored.

## Testing

`uv run pytest -m "not slow"` runs the unit tier with a mocked LLM and mocked data: agents, orchestrator, runner, scoring, persistence, cache, schemas, data tools and the UI rendering functions. `uv run pytest -m slow` hits live yfinance and SEC EDGAR. CI runs ruff and the unit tier on every push to `main` and on every pull request.

## Not in scope

- Grading past verdicts against realized returns
- A debate or revision round between desks
- News and sentiment data
- Watchlists and scheduled runs
- Portfolio or position awareness
