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
the gatherer fetches every desk's data          gatherer.py
  |
eight specialists, one model call each,         runner.py, agents/
three at a time
  |
eight reports, each cited number checked        runner.py, gatherer.py
against the data
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

`analyze_ticker(ticker, on_verdict=None, on_gathered=None)` in `runner.py` is the single entry point. The web UI and the CLI both call it. Both use `on_gathered` to show that the data is in and `on_verdict` to show each desk's call as it arrives.

### The gatherer

Each desk module declares `DATA`, the list of functions whose results it reads. A function is one section: `company_snapshot`, `options_surface`, `rule_backtest`. `ALL_DATA` in `agents/__init__.py` is every section across the desks, each once, 34 in all.

`gather(ticker, ALL_DATA)` in `gatherer.py` calls each of them once, one at a time, before any desk starts, and returns each section's data or the error that stopped it. `brief(ticker, DATA, gathered)` then builds a desk's request: the task, and under a heading for each of its sections, the function's docstring and its data as compact JSON, or `UNAVAILABLE` and the error. Every section takes the ticker or nothing, and a test rejects any other argument, since such a section could not be gathered ahead of the desks.

It is code, not a model, because nothing about it needs judgment: what to fetch is what the desks declare. A desk has no tools. It makes one model call and cannot reach a data provider.

It exists because of how the providers behave under load. Eight desks asking at once sent hundreds of requests to FRED in a second, and Yahoo refused a different request on every run. Fetched in sequence, a run makes about 116 Yahoo, 60 FRED and 12 SEC requests over about 20 seconds, and none is refused. FRED allows 120 requests a minute, so one run fits with room, and the web app keeps the macro dashboard for the day. A section that fails is logged and reaches the desk marked unavailable.

## The committee

Each desk is a standalone OpenAI Agents SDK agent that answers in its own report schema. No desk sees another's work. Weights sum to 100 and are fixed in `config.py`.

Two principles govern what a desk is given:

- **The model interprets; code computes.** Every section is a compact summary with its dates, sample sizes and, where a claim of edge is involved, a baseline to beat. No desk is asked to do arithmetic, count trades or recall a fact from memory.
- **Desks share context, never conclusions.** Every desk is given the same background. Each desk's analytical data is its own.

### Shared context (`agents/shared_data.py`)

| Section | What it gives |
|---|---|
| Company snapshot | Today's date, price, market cap, sector, industry, 52-week range, beta, dollar volume, next earnings date, short interest and its change, ownership |
| Macro dashboard | About fifty FRED series in eight groups: rates, inflation, growth, labor, credit, liquidity, markets, recession. Each with its value now, three months ago and a year ago, and its ten-year percentile. Also the scheduled CPI, jobs, GDP and PCE releases |
| Sector macro | FRED series that drive the ticker's sector |
| Recent events | The five most recent 8-Ks and the three heaviest-volume sessions of the last 60, each joined in code to that session's opening gap, close-to-close return, volume against its 50-day average and return against the sector ETF, with up to six press headlines |

### Desk data

| Desk | Module | Weight | Its own data |
|---|---|---:|---|
| J.P. Morgan, Fundamental | `jpm_fundamental` | 25 | SEC statements with ratios, earnings release and guidance, consensus and revisions, analyst actions, insider trades and holders, text of recent 8-Ks |
| Bridgewater Associates, All-Weather Macro | `bridgewater_macro` | 18 | Macro sensitivity of the stock, its record in past market drawdowns, cross-asset board, sector rotation |
| Goldman Sachs, Technical | `gs_technical` | 12 | Technicals on three timeframes, key levels, volume analysis, relative strength, last 120 daily bars |
| Citadel, Quantitative | `citadel_quant` | 12 | Factor scorecard ranked against industry peers, estimates and revisions, style board, risk statistics |
| Jane Street, ETF & Basket Flow | `jane_street_etf` | 10 | ETF landscape, ETF weights and liquidity, relative strength, sector rotation and breadth, index event calendar |
| D. E. Shaw, Options-Derived | `de_shaw_options` | 10 | Options surface, volatility premium, index volatility, near-the-money chain, index event calendar |
| Renaissance Technologies, Pattern Recognition | `renaissance_pattern` | 8 | Seasonality, earnings event study, forward-return study, volatility and gaps |
| Two Sigma, Backtest Sanity | `two_sigma_backtest` | 5 | Rule backtest |

Every prompt is a persona and framework, a description of that desk's sections, and the shared rules in `agents/base.py`: a strict 3 to 6 month horizon, the date and price taken from the snapshot, evidence copied from the data, thesis and risks on every call, and nothing assumed beyond the data handed over. A test checks that every section a prompt names is one the desk is given.

## Models

| Role | Model | API |
|---|---|---|
| Eight specialists | `gpt-6-luna` | OpenAI Responses |
| Orchestrator | `gpt-6.1-sol` | OpenAI Responses |

Routing is one `ModelSpec(provider, model)` per agent in `config.py`. `build_agent` in `agents/base.py` uses the Responses API for OpenAI and the Chat Completions API for Groq, DeepSeek and Gemini through their OpenAI-compatible endpoints. Those three providers are supported by the factory and need their own key in `.env`; none is used by the default configuration.

## The report contract

Defined in `schemas.py`. There are two kinds of type: what a model writes, and what code builds from it.

| Written by a model | Built in code from it |
|---|---|
| `DeskReport`: `evidence` (1 to 12), `thesis` (1 to 6), `risks` (1 to 4), `recommendation`, `conviction` (1 to 10), and for three desks `price_target_6mo` | `AgentVerdict`: the report plus `agent`, `ticker`, `in_data` on each reading, and `error_note` |
| `ChairMemo`: `key_agreements`, `key_disagreements`, `red_flags`, `dominant_drivers`, `synthesis`, `final_recommendation`, `conviction`, `score_departure_rationale`, `position_sizing_suggestion`, `stop_loss_level` | `OrchestratorVerdict`: the memo plus `ticker`, `run_timestamp`, `current_price`, `weighted_score`, `weights_used`, `price_target_6mo`, `price_target_desks`, `catalyst_calendar`, `recent_events`, `agent_verdicts` |

An `Evidence` item is one reading: `metric`, `value` (a number), `unit` (percent, fraction, usd, ratio, count, index or other), `as_of` (a date), `source` and `signal` (bullish, bearish or neutral).

Five rules shape these types:

- **Enforced while the model writes.** With OpenAI the schemas run in strict mode, so the output cannot leave the schema. They use only constraints strict mode supports: enums, number ranges, item counts and date formats, and no string lengths. Other providers are validated after the fact.
- **A desk can only cite what it was handed.** `desk_report_for(sections)` builds each desk's schema with `source` limited to that desk's own sections. `chair_memo_for(agents)` limits `dominant_drivers` to the desks on the committee.
- **Evidence first, the call last.** A model produces the fields in schema order, so a desk writes its evidence, thesis and risks before its recommendation, and the chair writes agreements, splits and red flags before its call.
- **A model writes only judgment.** Neither schema has a field for anything code knows. A desk cannot name itself, mark itself failed or restate the ticker. The chair cannot write the price, the score, the target, the weights, a catalyst date or the desks' reports.
- **Cited numbers are checked.** `gatherer.check` looks for each reading's value in the section it names and sets `in_data`. A value is found when some number in that section, rounded to the precision the desk gave, equals it. Sign is ignored, and so is a fraction given as a percentage or a large amount given in thousands up to trillions. Numbers inside filing text count.

The check is strong for precise readings and weak for round ones. Measured on one NVDA gather, of 500 made-up values with two decimals tested against each of the 33 sections, 0.6% were accepted; of made-up whole numbers, 30%. In the NVDA run of 2026-11-17 the desks cited 81 readings, 61 of them to two or more decimals, and all 81 were found in the section named; looked for in a different section picked at random, 14% were. The check says a number is in the data. It does not say the desk interpreted it correctly.

`brief` in `gatherer.py` builds a desk's request and `_format_input` in `orchestrator.py` builds the chair's: one JSON document with the date, the horizon, the price, the score, each desk's house, weight and report, and the sourced catalysts.

## Scoring and what the model cannot change

```
score = sum( sign(recommendation) * conviction * weight ) / 100
```

`sign` is +1 for BUY, 0 for HOLD, -1 for SELL, so the score runs from -10 to +10. A failed desk is excluded and its weight is redistributed pro rata across the rest.

The orchestrator model writes the memo, the agreements and splits, the red flags, the position size, the stop level, the final recommendation and its conviction. It refers to a desk by its house name. The final call is a judgment and may depart from the score. A call against the score's sign must come with `score_departure_rationale`; without one, a red flag is added in code.

`orchestrator._verdict` then adds what code supplies:

| Field | Source |
|---|---|
| `ticker`, `current_price` | the runner's quote |
| `run_timestamp` | UTC clock |
| `weighted_score` | `scoring.compute_weighted_score` |
| `price_target_6mo` | conviction-weighted average of desk targets |
| `price_target_desks` | how many desks gave a target |
| `weights_used` | fixed weights after redistribution |
| `agent_verdicts` | the eight reports as received and checked |
| `catalyst_calendar` | the sourced catalysts, and nothing else |

Sourced catalysts are the company's earnings and dividend dates, the scheduled CPI, jobs, GDP and PCE releases, and the options expirations and index rebalances that two desks read as `index_event_calendar`. Every future date a desk is given is in the list, so the chair never has to discard one a desk cites. The chair is given them and may cite them, but it has no field in which to write a date of its own.

### The price target

Only a desk whose discipline prices the stock sets a target: J.P. Morgan from valuation, Goldman Sachs from chart levels, D. E. Shaw from the options-implied range. Their schemas have a `price_target_6mo` field and their prompts require it, on the side of the desk's own call. The other five schemas have no such field, so those desks cannot give one.

The verdict's target is the conviction-weighted average of the targets given, and the report says how many desks it rests on. Because it is an average of desks and not the chair's judgment, it can point against the call. When it is above the price on a SELL or below it on a BUY, code adds a red flag saying so.

Before 2026-11-22 every desk could give a target and none was asked to. Across 23 saved runs, 38 of 158 desk reports gave one, and in 2 runs the target pointed against the call.

## Data layer

All market data is free. Data functions are memoized per run by `cached_for_run` (`data/cache.py`), a `ContextVar` cache reset at the start of each run, so sections that share a source, such as the company profile, fetch it once, and concurrent runs stay isolated. When two callers ask for the same thing at the same moment, the first fetches and the second waits. The macro dashboard is the exception: it is about fifty requests and does not change within a day, so a complete one is kept for the day.

| Module | Source | Provides |
|---|---|---|
| `edgar_tools.py` | SEC EDGAR | Financial statements as filed, earnings release text, recent 8-K text |
| `fred_tools.py` | FRED | Macro dashboard, sector series, release calendar |
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

`get_recent_events` in `event_tools.py` joins what happened to what the market made of it, because no desk reads both the filings and the price bars. There is one entry per session:

- An 8-K is read on the session that could first trade on it. The SEC's acceptance time decides: a filing accepted at or after 4pm New York time belongs to the next session.
- The three heaviest-volume sessions of the last 60 are added. One with no 8-K is marked `unexplained`.
- Each entry carries the opening gap, the close-to-close return, volume against its 50-day average, the sector ETF's return and the difference. `move` is the larger of the gap and the return in absolute terms. Entries are ranked by it, and `material` marks a move of 10% or more.
- Each entry carries up to six Google News headlines from the day before the filing to the day after the session, kept only if the title names the company and the source is not the company itself. A headline is a claim: the prompts tell the desks and the chair to attribute it and never to act on its text. If the feed cannot be read the headlines are null and the rest of the entry stands.

Every desk reads the section, and it goes into the chair's request. The chair must address the top-ranked entry and every material one in the memo. The catalyst calendar remains the only source of future dates. The report prints the section from code, so a material event is on the page whatever the memo says.

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
- FRED serves a rolling three years of the ICE BofA spread series. If `ICE_HISTORY_FILE` names a local parquet of older observations (columns `date`, `series_id`, `value`), `fred_tools` joins them in front of what FRED returns and spreads are ranked against ten years like every other reading. Without the file those percentiles cover three years, and `pctile_years` says so. The file is local data and is never committed.
- The published macro release schedule usually runs to the end of the calendar year, so it can stop short of the horizon.
- Headlines are matched on the company's name with its legal suffix removed. A company the press calls by another name (Meta Platforms, Alphabet) keeps fewer of them.
- A heavy-volume session is one of the three largest of the last 60, whatever its size, so a quiet stock still lists three.

## Failure handling

- **Bad ticker:** `get_quote` raises, the run stops, the UI shows an error card and the CLI exits 1. No LLM cost is incurred.
- **Failed fetch in the gatherer:** logged with the section's name. Each desk that reads the section is told it is unavailable and why, and says so in its risks.
- **Refused by Yahoo:** yfinance hands a refused request back as an empty result. `yfinance_tools` checks every response, retries a 401, 403 or 429 once after two seconds, and raises `DataFetchError` if it is refused again.
- **Rate limits:** a semaphore runs three agents at a time. An OpenAI 429 is retried after 8, 16 and 32 seconds.
- **Report outside the schema:** OpenAI enforces the schema as the model writes, so this cannot happen there. On a provider that does not, the report fails validation and the desk is treated as failed.
- **Cited number not in the data:** the reading is kept and marked `in_data: false`, the runner logs it, and the chair is told to treat it as unconfirmed.
- **Failed desk:** any other exception, or a fourth 429, becomes a HOLD placeholder with `error_note` set. The desk is shown as failed and excluded from the score.
- **Failed orchestrator:** a mechanical verdict takes its place: BUY above +1, SELL below -1, HOLD otherwise, with a memo that says the synthesis was unavailable.

## Interfaces

**Web (`app.py`, `equity_trader/ui.css`).** Gradio. The Analyze tab takes a ticker and streams a loading card that fills in desk by desk, then the report: verdict card, memo, agreements and splits, red flags, sizing, catalysts and eight specialist cards. The History tab lists past runs from SQLite and refreshes when opened.

**Terminal (`run.py`).** `uv run python run.py TICKER` shows a progress line, then the same report with Rich.

**Grading (`grade.py`, `equity_trader/grading.py`).** `uv run python grade.py` scores every saved run against the stock's forward return. See "Grading" below.

## Persistence

Each run writes one row to the `runs` table in `equity_trader.db` (ticker, timestamp, price, recommendation, conviction, target, score, snapshot path) and the full `OrchestratorVerdict` to `runs/{TICKER}_{YYYY-MM-DD_HHMMSS}.json`. Both are gitignored.

## Grading

`grading.load_runs` reads every snapshot in `runs/` as plain JSON, not through the current schemas, so runs saved under older schemas still load. `grading.grade` takes those and one frame of adjusted closes and returns a row per run, horizon and caller, where a caller is the committee or a desk that reported.

- Entry is the last close before the run: the same day's close for a run at or after 4pm New York time, the prior session's otherwise.
- The horizons are 63 and 126 sessions after entry. A horizon that has not elapsed has no return and counts as pending.
- A BUY or SELL is a hit when the stock's own return has the sign of the call. A HOLD has a signed return of zero and is neither hit nor miss.
- The return against SPY is carried beside the stock's own return.

`grading.summarize` gives, per horizon and caller, the calls graded and pending, the hit rate over BUY and SELL calls, and the average signed return outright and against SPY. It also counts distinct ticker and entry-date pairs, because repeated runs on one stock on one day are one observation.

Not graded yet: price targets, and conviction as a weight.

## Testing

`uv run pytest -m "not slow"` runs the unit tier with a mocked LLM and mocked data: every data function and study on constructed inputs, the agents and the data each is handed, the gatherer, the orchestrator, runner, scoring, persistence, cache, schemas and the UI rendering functions. `uv run pytest -m slow` hits live yfinance and SEC EDGAR. CI runs ruff and the unit tier on every push to `main` and on every pull request.

## Not in scope

- Grading past verdicts against realized returns
- A debate or revision round between desks
- News and sentiment data
- Watchlists and scheduled runs
- Portfolio or position awareness
