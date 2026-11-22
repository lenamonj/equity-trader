<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" width="100%" alt="EquityTrader. A multi-strategy investment committee: eight specialist desks, one ticker, one verdict.">
</picture>

<br><br>

<img src="https://img.shields.io/badge/python-3.11+-B08D57?style=flat-square&labelColor=0F1B33" alt="Python 3.11+">
<img src="https://img.shields.io/badge/agents-OpenAI%20Agents%20SDK-B08D57?style=flat-square&labelColor=0F1B33" alt="OpenAI Agents SDK">
<img src="https://img.shields.io/badge/interface-Gradio%20%2B%20Rich-B08D57?style=flat-square&labelColor=0F1B33" alt="Gradio and Rich">
<img src="https://img.shields.io/badge/managed%20by-uv-B08D57?style=flat-square&labelColor=0F1B33" alt="Managed by uv">
<img src="https://img.shields.io/badge/license-MIT-B08D57?style=flat-square&labelColor=0F1B33" alt="MIT license">
<img src="https://img.shields.io/github/actions/workflow/status/lenamonj/equity-trader/ci.yml?branch=main&label=tests&style=flat-square&labelColor=0F1B33" alt="Test status">

<br><br>

**Eight specialist agents, each modeled on the desk of a storied investment house, analyze one ticker independently.**<br>
**A committee chair turns their verdicts into a single Buy, Hold or Sell for the next 3 to 6 months.**<br>
The premise is diversity in thought process: desks that read different evidence make different mistakes, so the committee's call is harder to fool than any one view.

<br>

<a href="#quick-start">Quick start</a> &nbsp;&middot;&nbsp;
<a href="#the-committee">The committee</a> &nbsp;&middot;&nbsp;
<a href="#a-run-live">A run, live</a> &nbsp;&middot;&nbsp;
<a href="#how-the-desks-report">Structured reports</a> &nbsp;&middot;&nbsp;
<a href="#the-report">The report</a> &nbsp;&middot;&nbsp;
<a href="#architecture">Architecture</a> &nbsp;&middot;&nbsp;
<a href="#how-the-verdict-is-computed">Scoring</a>

</div>

<br>

<img src="docs/screenshots/verdict.png" width="100%" alt="The verdict card for NVDA: recommendation, conviction, current price, 6-month target and committee score.">

<br>

> [!IMPORTANT]
> **Every ticker run makes paid LLM calls.** An `OPENAI_API_KEY` with funded credit is required. One run is nine model calls, eight `gpt-6-luna` specialists and one `gpt-6.1-sol` orchestrator, billed per token by OpenAI. Cost depends on how much data each desk is handed. Models are set per agent in `equity_trader/config.py`, which can also route any agent to Gemini, Groq or DeepSeek.

<br>

## The committee

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/committee-dark.svg">
  <img src="docs/assets/committee-light.svg" width="100%" alt="The eight desks and their weights: J.P. Morgan (Fundamental) 25%, Bridgewater Associates (All-Weather Macro) 18%, Goldman Sachs (Technical) 12%, Citadel (Quantitative) 12%, Jane Street (ETF and Basket Flow) 10%, D. E. Shaw (Options-Derived) 10%, Renaissance Technologies (Pattern Recognition) 8%, Two Sigma (Backtest Sanity) 5%.">
</picture>

Each desk has its own analysis framework, its own data and its own voice. All eight return the same structured verdict: a recommendation, a conviction from 1 to 10, thesis bullets, risks, and the data points cited. The three desks whose discipline prices the stock (fundamental, technical, options) also give a 6-month price target.

<br>

## What each desk sees

Every desk starts from the same context: today's date and price, the company's sector, next earnings date and short interest, a dashboard of about fifty macro readings with their direction, the series that drive the company's sector, and the company's recent events with the price reaction to each. Then each works with data of its own.

All of it is computed in code and handed over with dates and sample sizes, so the model interprets numbers instead of producing them.

| Desk | Its own data |
|---|---|
| J.P. Morgan | SEC statements with ratios, the earnings release and guidance, consensus and revisions, analyst actions, insider trades, recent 8-Ks |
| Bridgewater Associates | How the stock moves with rates, credit, the dollar and oil, its record in past market stress, cross-asset and sector boards |
| Goldman Sachs | Trend on three timeframes, support and resistance, volume, relative strength, the last 120 daily bars |
| Citadel | Factor scorecard ranked against industry peers, estimate revisions, style performance, risk statistics |
| Jane Street | ETF weights and fund size, sector rotation and breadth, relative strength, index event dates, liquidity |
| D. E. Shaw | Implied volatility across expiries, skew, positioning, implied against realized, earnings pricing, index volatility |
| Renaissance Technologies | Full-history seasonality, earnings event study, what followed past shocks and days like today, gaps, volatility regime |
| Two Sigma | Eleven long and short rules at one, three and six month holds, with significance, stability and a cross-check on the market |

<br>

## A run, live

Each desk's call appears the moment it reports, so a run that takes a little over two minutes is never a blank spinner. The terminal report shows the same progress.

<img src="docs/screenshots/progress.png" width="100%" alt="The loading card mid-run: desks that have reported show their call, the rest are still working.">

<br>

## The report

The chair writes a portfolio manager memo, names where the desks agree and where they split, flags what could break the call, and suggests a position size.

<img src="docs/screenshots/memo.png" width="100%" alt="The investment memo with agreements, splits, red flags and position sizing.">

Below the memo, code prints what has already happened and what is scheduled. Each recent 8-K and heavy-volume session is shown with that session's price reaction and the press headlines around it. The catalyst calendar is fetched live from market data, so the dates are real.

<img src="docs/screenshots/events.png" width="100%" alt="Recent events with the opening gap, close, volume and move against the sector ETF for each, press headlines with their sources, and the catalyst calendar.">

Every desk's reasoning stays on the page. Nothing is averaged away.

<img src="docs/screenshots/specialists.png" width="100%" alt="Eight specialist verdict cards, each with a recommendation, conviction, thesis bullets and risks.">

<br>

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img src="docs/assets/architecture-light.svg" width="100%" alt="A ticker goes to the gatherer, which fetches every desk's data before any desk starts. Eight specialist agents are each handed their share and answer in one model call, three at a time. The orchestrator combines a deterministic score, an LLM memo and live catalysts into one verdict, saved to SQLite and JSON.">
</picture>

- **Independent specialists.** Each desk is a standalone OpenAI Agents SDK agent that answers in its own report schema. No desk sees another's work. Desks share context, never conclusions.
- **The model interprets; code computes.** Ratios, indicators, event studies and backtests are calculated in code. No desk does arithmetic or recalls a fact from memory.
- **A gatherer before the desks.** One component fetches everything the desks will read, one request at a time, before any desk starts. It is code, not a model: what to fetch is fixed by the data each desk declares. A run makes about 60 FRED requests against a limit of 120 a minute, and the macro dashboard is kept for the day.
- **One model call per desk.** A desk has no tools. Its data arrives in its request, it reads what bears on its discipline, and it answers once. No desk can reach a data provider.
- **Rate-limit aware fan-out.** A semaphore runs three agents at a time, with exponential backoff on `RateLimitError`.
- **No silent gaps.** A request Yahoo refuses is retried once after a pause and then raises. The desk is told that section is unavailable, so a failed fetch never reads as empty data.
- **Temporal grounding.** The orchestrator is handed today's date, the current price, and a real catalyst calendar. Any catalyst dated outside the horizon is dropped in code.
- **Events, not just numbers.** The fundamental desk reads the company's five most recent 8-Ks in full, so a CEO departure, an acquisition or a new guarantee is weighed in the call before it shows up in any statement.
- **Events joined to price in code.** Each recent 8-K and each heaviest-volume session is joined to that session's opening gap, return, volume and move against the sector ETF, with press headlines. Every desk and the chair read it, the chair must address the largest, and the report prints it from code, so a material event is on the page whatever the memo says.
- **Dated inputs, honest statistics.** Every reading carries its observation date. Every statistical claim carries its sample size and the baseline it has to beat, so a stale number or a fake edge is visible to the chair.
- **Graceful degradation.** A failed desk is excluded and its weight is redistributed. Only the runner can mark a desk as failed. If the orchestrator call fails, a mechanical aggregation takes its place.

<br>

## How the desks report

Every desk answers the chair in a fixed JSON schema, and the model provider holds it to that schema while it writes. Most of a report is a list of evidence, one typed reading per item:

```json
{"metric": "RSI, 14-day", "value": 16.3, "unit": "index",
 "as_of": "2026-11-17", "source": "technicals", "signal": "bearish", "in_data": true}
```

A short thesis, the risks and the call follow the evidence. The chair receives all eight reports as one JSON document and compares them reading by reading.

| What is controlled | How |
|---|---|
| **Shape** | The schema is enforced as the model generates, so a report that does not fit cannot be produced: the call is BUY, HOLD or SELL, conviction is 1 to 10, every reading has a number, a unit, a date and a source. |
| **Sources** | Each desk's schema lists only the data sections that desk was handed. It cannot cite anything else. |
| **Order** | Evidence comes first and the call comes last, and a model writes in that order. The call follows from the evidence instead of being justified after the fact. |
| **Ownership** | A model writes only what needs judgment. Which desk it is, the ticker, the price, the score, the target and the catalysts are added in code. The schemas have no such fields, so no model can get them wrong. |
| **Grounding** | Code looks for every number a desk cites in the data section it names and marks it `in_data`. The chair is told to treat a reading that was not found as unconfirmed. |

This is what keeps a chain of models honest. A number a desk cites that is not in its data is flagged before the chair reads it. A format cannot drift from run to run, because there is only one format. The score, the price and the target cannot be garbled in a hand-off, because code carries them and no model writes them. In the run shown above the desks cited 81 readings, and all 81 were found in the data they were handed.

The check has limits. It confirms that a number is in the data, not that the desk read it correctly, and a round number such as 16 is easier to match than 16.3.

<br>

## How the verdict is computed

The committee score is a weighted sum, computed in code:

```
score = sum( sign(recommendation) * conviction * weight ) / 100
```

`sign` is +1 for BUY, 0 for HOLD and -1 for SELL. Conviction runs 1 to 10 and weights sum to 100, so the score ranges from -10 to +10.

The weights are fixed in code. The orchestrator cannot change them. It reads the score and all eight reports, then writes a 150 to 300 word memo and makes the final call. That call is a judgment and may depart from the score, but a call against the score's sign has to come with a written rationale.

The orchestrator's schema holds only its judgment. The score, the conviction-weighted price target, the run timestamp, the current price, the desks' reports, the sourced catalysts and the recent events are added in code afterwards, so the model never writes them and cannot alter them. A call against the score's sign with no rationale is flagged on the report.

<br>

## Quick start

[uv](https://docs.astral.sh/uv/) is the only prerequisite. It installs Python for you.

```bash
git clone https://github.com/lenamonj/equity-trader.git
cd equity-trader
cp .env.template .env      # then fill in the three keys below
uv sync
```

```bash
uv run python run.py NVDA   # terminal report
uv run python app.py        # web UI at http://127.0.0.1:7860
uv run gradio app.py        # web UI with hot reload
```

| Key | Source | Cost |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) | Paid, per token |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Free |
| `SEC_EDGAR_USER_AGENT_EMAIL` | Your own email | Free. The SEC requires it for EDGAR access |
| `ICE_HISTORY_FILE` | A local parquet of older ICE BofA spread observations | Optional. FRED serves three years of these series; with the file, spreads are ranked against ten |

<details>
<summary>Installing uv</summary>

<br>

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Other routes are in the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/).

</details>

<br>

## Data sources

All market data is free. No paid feeds.

| Source | Used for |
|---|---|
| SEC EDGAR | Financial statements as filed, earnings releases and guidance, the text of recent 8-Ks |
| yfinance | Prices, valuation multiples, analyst estimates and rating actions, insider transactions, options chains, ETF holdings, earnings and dividend dates |
| FRED | About fifty macro series on rates, inflation, growth, labor, credit and liquidity, sector series, the data release calendar |
| Google News RSS | Press headlines around each recent event, attributed to their source |

Companies that file no US-GAAP statements with the SEC, such as IFRS filers, fall back to yfinance statements, and the desk is told which source it received.

Free does not mean unrestricted. yfinance reads Yahoo Finance, which yfinance itself describes as intended for personal use only, and the Google News feed states that it is for personal, non-commercial use. Read both providers' terms before using this project for anything else.

<br>

## Testing

```bash
uv run pytest -m "not slow"   # unit tests: mocked LLM, mocked data
uv run pytest -m slow         # integration: live yfinance and SEC EDGAR
uv run pytest                 # everything
```

The mocked tier covers every data function and study on constructed inputs, all eight agents and the data each is handed, the gatherer, the orchestrator, the runner, the scoring math, persistence, the cache and the schemas.

<br>

## Project layout

<details>
<summary>Expand the tree</summary>

```
equity-trader/
├── app.py                      # Gradio UI entry point
├── run.py                      # CLI entry point
├── pyproject.toml              # uv project + dependencies
├── .env.template               # required environment variables
├── equity_trader/
│   ├── config.py               # model routing + weights
│   ├── schemas.py              # what a desk and the chair must answer in
│   ├── exceptions.py
│   ├── scoring.py              # deterministic weighted score
│   ├── orchestrator.py         # LLM synthesizer + mechanical fallback
│   ├── gatherer.py             # fetches the data, briefs each desk, checks citations
│   ├── runner.py               # gatherer, then async fan-out, then the chair
│   ├── persistence.py          # SQLite + JSON snapshots
│   ├── ui.css                  # web UI stylesheet
│   ├── data/
│   │   ├── cache.py            # per-run ContextVar cache
│   │   ├── edgar_tools.py      # SEC statements, earnings releases, 8-Ks
│   │   ├── event_tools.py      # 8-Ks and heavy-volume days joined to price and headlines
│   │   ├── fred_tools.py       # macro dashboard, sector series, releases
│   │   ├── yfinance_tools.py   # raw market data access
│   │   ├── company_tools.py    # snapshot, estimates, insiders, peers, factors
│   │   ├── technical_tools.py  # trend, levels, volume, relative strength
│   │   ├── pattern_tools.py    # seasonality, event and forward-return studies
│   │   ├── options_tools.py    # volatility surface, skew, positioning
│   │   ├── market_tools.py     # cross-asset, sectors, ETFs, stress history
│   │   └── calendar_tools.py   # real upcoming catalysts
│   └── agents/
│       ├── base.py             # agent factory + shared rules
│       ├── shared_data.py      # context every desk is given
│       ├── jpm_fundamental.py
│       ├── bridgewater_macro.py
│       ├── gs_technical.py
│       ├── citadel_quant.py
│       ├── jane_street_etf.py
│       ├── de_shaw_options.py
│       ├── renaissance_pattern.py
│       └── two_sigma_backtest.py
├── tests/                      # unit + integration tests
├── .github/workflows/ci.yml    # lint + unit tests on every push
└── docs/
    ├── design.md               # system design
    ├── assets/                 # README plates + their generator
    └── screenshots/
```

</details>

The full design is in [docs/design.md](docs/design.md).

<br>

## Disclaimer

EquityTrader is a research and engineering project for personal, non-commercial use. Its output is generated by language models, can be wrong, and is not investment advice.

The code is MIT licensed. The data is not: the Yahoo Finance and Google News data it reads are offered by those providers for personal, non-commercial use only.

The desks are stylized personas. This project is not affiliated with, endorsed by, or representative of J.P. Morgan, Bridgewater Associates, Goldman Sachs, Citadel, Jane Street, D. E. Shaw, Renaissance Technologies or Two Sigma. Those names belong to their respective owners.

<br>

<div align="center">

Built with [Claude Code](https://claude.com/claude-code), the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) and [Gradio](https://gradio.app/).

[MIT](LICENSE). Copyright (c) 2026 Jeff Lenamon.

</div>
