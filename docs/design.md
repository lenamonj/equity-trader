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

`analyze_ticker(ticker, on_verdict=None)` in `runner.py` is the single entry point. The web UI and the CLI both call it, and both use `on_verdict` to show each desk's call as it arrives.

## The committee

Each desk is a standalone OpenAI Agents SDK agent with `AgentVerdict` as its output type. No desk sees another's work. Weights sum to 100 and are fixed in `config.py`.

Two principles govern what a desk is given:

- **The model interprets; code computes.** Every tool returns a compact summary with its dates, sample sizes and, where a claim of edge is involved, a baseline to beat. No desk is asked to do arithmetic, count trades or recall a fact from memory.
- **Desks share context, never conclusions.** Every desk can read the same background. Each desk's analytical tools are its own.

### Shared context (`agents/shared_tools.py`)

| Tool | What it gives |
|---|---|
| Company snapshot | Today's date, price, market cap, sector, industry, 52-week range, beta, dollar volume, next earnings date, short interest and its change, ownership |
| Macro dashboard | About fifty FRED series in eight groups: rates, inflation, growth, labor, credit, liquidity, markets, recession. Each with its value now, three months ago and a year ago, and its ten-year percentile. Also the scheduled CPI, jobs, GDP and PCE releases |
| Sector macro | FRED series that drive the ticker's sector |
| FRED series and search | Any series by id, and a keyword search for the id |

### Desk tools

| Desk | Module | Weight | Its own tools |
|---|---|---:|---|
| J.P. Morgan, Fundamental | `jpm_fundamental` | 25 | SEC statements with ratios, earnings release and guidance, consensus and revisions, analyst actions, insider trades and holders, text of recent 8-Ks |
| Bridgewater Associates, All-Weather Macro | `bridgewater_macro` | 18 | Macro sensitivity of the stock, its record in past market drawdowns, cross-asset board, sector rotation |
| Goldman Sachs, Technical | `gs_technical` | 12 | Technicals on three timeframes, key levels, volume analysis, relative strength, last 120 daily bars |
| Citadel, Quantitative | `citadel_quant` | 12 | Factor scorecard ranked against industry peers, estimates and revisions, style board, risk statistics |
| Jane Street, ETF & Basket Flow | `jane_street_etf` | 10 | ETF landscape, ETF weights and liquidity, relative strength, sector rotation and breadth, index event calendar |
| D. E. Shaw, Options-Derived | `de_shaw_options` | 10 | Options surface, volatility premium, index volatility, near-the-money chain, index event calendar |
| Renaissance Technologies, Pattern Recognition | `renaissance_pattern` | 8 | Seasonality, earnings event study, forward-return study, volatility and gaps |
| Two Sigma, Backtest Sanity | `two_sigma_backtest` | 5 | Rule backtest |

Every prompt is a persona and framework, a description of that desk's tools, and the shared rules in `agents/base.py`: a strict 3 to 6 month horizon, the snapshot called first, cited data points, thesis and risks on every call, one call per tool, and `error_note` left null. A test checks that every tool a prompt names is one the desk has.

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

The orchestrator model writes the memo, the agreements and splits, the red flags, the position size, the stop level, the final recommendation and its conviction. It is given each desk's house name and told to use it in prose. The final call is a judgment and may depart from the score. A call against the score's sign must come with `score_departure_rationale`; without one, a red flag is added in code.

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

Sourced catalysts are the company's earnings and dividend dates and the scheduled CPI, jobs, GDP and PCE releases.

## Data layer

All market data is free. Tools are memoized per run by `cached_for_run` (`data/cache.py`), a `ContextVar` cache reset at the start of each run, so eight agents share one fetch and concurrent runs stay isolated. The macro dashboard is the exception: it is about fifty requests and does not change within a day, so a complete one is kept for the day.

| Module | Source | Provides |
|---|---|---|
| `edgar_tools.py` | SEC EDGAR | Financial statements as filed, earnings release text, recent 8-K text |
| `fred_tools.py` | FRED | Macro dashboard, sector series, any series, search, release calendar |
| `yfinance_tools.py` | yfinance | Raw access: prices, company profile, valuation multiples, earnings dates, ETF landscape, fallback statements |
| `company_tools.py` | yfinance | Snapshot, estimates and surprises, analyst view, insider activity, industry peers, factor scorecard, risk statistics |
| `technical_tools.py` | prices | Technicals, key levels, volume analysis, relative strength |
| `pattern_tools.py` | prices, earnings dates | Seasonality, earnings event study, forward-return study, volatility and gaps |
| `options_tools.py` | yfinance options | Surface across expiries, volatility premium, near-the-money chain, index volatility |
| `market_tools.py` | prices, FRED | Cross-asset, sector and style boards, ETF exposure, stress history, macro sensitivity, index event calendar |
| `calendar_tools.py` | yfinance, FRED | Catalysts inside the horizon |

### Financial statements

Statements come from the SEC, not from a market data vendor. `get_financial_statements` reads the XBRL company facts for a fixed set of income, cash flow and balance sheet lines over the last five fiscal years and six quarters.

- Each period carries its fiscal label (for example `Q2 FY2027`), how many periods ago it is, its period end, form and first filing date. Values reflect any later restatement.
- Margins, growth rates, coverage and leverage are computed per period, with a summary of three and five year revenue growth and trailing-twelve-month totals.
- Companies file different XBRL tags for the same line. The candidate tag reported most recently is used, and the output names it.
- Filings give cash flow items and fourth quarters only as year-to-date figures. A single quarter is the difference between consecutive year-to-date figures. Per-share figures are never derived.
- A company with no US-GAAP annual statements on EDGAR (IFRS filers, funds) falls back to a compact set of yfinance statements. The `source` field says which was used.

### Events and guidance

`get_recent_8ks` returns the five most recent 8-Ks, each with its filing date, its item codes translated to names in code (5.02 is a director or officer departure or appointment), and the filing's own text up to 6,000 characters. `get_earnings_release` returns the opening and the outlook section of the exhibits attached to the latest results 8-K, which is where guidance is stated. The J.P. Morgan desk must put any material event in its thesis or risks with the filing date.

### Statistical studies

The pattern and backtest desks work from the stock's full price history.

- Every result carries its sample size. Studies with overlapping windows report the number of separate episodes as well as the number of days.
- Conditional results sit next to the unconditional baseline for the same horizon.
- The backtest runs eleven fixed rules, six long and five short, at 20, 63 and 126 session holds with non-overlapping trades. Each result has the edge over the baseline after a stated trading cost, a t-statistic, the edge in each half of the history, and the same rule run on SPY and the sector ETF. The output states how many tests were run.

### Options

`options_surface` reads up to six expiries from one week to one year: at-the-money implied volatility, the implied move, skew between the 90% put and the 110% call, put/call ratios, the largest open-interest strikes and unusual volume. Quotes are delayed. There are no Greeks and no history of implied volatility, and the D. E. Shaw prompt says so.

### Rules the data layer holds to

- Every reading carries its observation date, so a stale number is visible.
- A value that is unavailable is `null`, and the reason is logged. Nothing is estimated.
- An unknown ticker raises `TickerNotFoundError` before any LLM call is made.
- Failures are logged or raised, never swallowed.

### Known limits of the free data

- ETF holdings cover each fund's ten largest positions.
- Creation and redemption flows are not available; fund volume against its average is the proxy.
- Credit spread history on FRED starts in November 2023, so those percentiles cover three years.
- The published macro release schedule usually runs to the end of the calendar year, so it can stop short of the horizon.

## Failure handling

- **Bad ticker:** `get_quote` raises, the run stops, the UI shows an error card and the CLI exits 1. No LLM cost is incurred.
- **Failed tool:** the error is returned to the model as the tool's result, and the desk carries on with what it has.
- **Rate limits:** a semaphore runs three agents at a time. An OpenAI 429 is retried after 8, 16 and 32 seconds.
- **Failed desk:** any other exception, or a fourth 429, becomes a HOLD placeholder with `error_note` set. The desk is shown as failed and excluded from the score.
- **Failed orchestrator:** a mechanical verdict takes its place: BUY above +1, SELL below -1, HOLD otherwise, with a memo that says the synthesis was unavailable.

## Interfaces

**Web (`app.py`, `equity_trader/ui.css`).** Gradio. The Analyze tab takes a ticker and streams a loading card that fills in desk by desk, then the report: verdict card, memo, agreements and splits, red flags, sizing, catalysts and eight specialist cards. The History tab lists past runs from SQLite and refreshes when opened.

**Terminal (`run.py`).** `uv run python run.py TICKER` shows a progress line, then the same report with Rich.

## Persistence

Each run writes one row to the `runs` table in `equity_trader.db` (ticker, timestamp, price, recommendation, conviction, target, score, snapshot path) and the full `OrchestratorVerdict` to `runs/{TICKER}_{YYYY-MM-DD_HHMMSS}.json`. Both are gitignored.

## Testing

`uv run pytest -m "not slow"` runs the unit tier with a mocked LLM and mocked data: every data tool and study on constructed inputs, the agents and their tool wiring, the orchestrator, runner, scoring, persistence, cache, schemas and the UI rendering functions. `uv run pytest -m slow` hits live yfinance and SEC EDGAR. CI runs ruff and the unit tier on every push to `main` and on every pull request.

## Not in scope

- Grading past verdicts against realized returns
- A debate or revision round between desks
- News and sentiment data
- Watchlists and scheduled runs
- Portfolio or position awareness
