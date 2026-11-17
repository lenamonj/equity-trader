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
**A committee chair turns their verdicts into a single Buy, Hold or Sell for the next 3 to 6 months.**

<br>

<a href="#quick-start">Quick start</a> &nbsp;&middot;&nbsp;
<a href="#the-committee">The committee</a> &nbsp;&middot;&nbsp;
<a href="#a-run-live">A run, live</a> &nbsp;&middot;&nbsp;
<a href="#the-report">The report</a> &nbsp;&middot;&nbsp;
<a href="#architecture">Architecture</a> &nbsp;&middot;&nbsp;
<a href="#how-the-verdict-is-computed">Scoring</a>

</div>

<br>

<img src="docs/screenshots/verdict.png" width="100%" alt="The verdict card for NVDA: recommendation, conviction, current price, 6-month target and committee score.">

<br>

> [!IMPORTANT]
> **Every ticker run makes paid LLM calls.** An `OPENAI_API_KEY` with funded credit is required. One run is eight `gpt-6-luna` specialist runs plus one `gpt-6.1-sol` orchestrator call, billed per token by OpenAI. Cost depends on prompt length and how many tools each agent calls. Models are set per agent in `equity_trader/config.py`, which can also route any agent to Gemini, Groq or DeepSeek.

<br>

## The committee

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/committee-dark.svg">
  <img src="docs/assets/committee-light.svg" width="100%" alt="The eight desks and their weights. The same data is in the table below.">
</picture>

Each desk has its own analysis framework, its own tools and its own voice. All eight return the same structured verdict: a recommendation, a conviction from 1 to 10, an optional 6-month price target, thesis bullets, risks, and the data points cited.

<details>
<summary>The committee as a table</summary>

<br>

| Desk | Discipline | Weight |
|---|---|---:|
| J.P. Morgan | Fundamental | 25% |
| Bridgewater Associates | All-Weather Macro | 18% |
| Goldman Sachs | Technical | 12% |
| Citadel | Quantitative | 12% |
| Jane Street | ETF & Basket Flow | 10% |
| D. E. Shaw | Options-Derived | 10% |
| Renaissance Technologies | Pattern Recognition | 8% |
| Two Sigma | Backtest Sanity | 5% |

</details>

<br>

## A run, live

Each desk's call appears the moment it reports, so a run that takes a minute or two is never a blank spinner. The terminal report shows the same progress.

<img src="docs/screenshots/progress.png" width="100%" alt="The loading card mid-run: desks that have reported show their call, the rest are still working.">

<br>

## The report

The chair writes a portfolio manager memo, names where the desks agree and where they split, flags what could break the call, and suggests a position size. The catalyst calendar is fetched live from market data, so the dates are real.

<img src="docs/screenshots/memo.png" width="100%" alt="The investment memo with agreements, splits, red flags, position sizing, and the catalyst calendar.">

Every desk's reasoning stays on the page. Nothing is averaged away.

<img src="docs/screenshots/specialists.png" width="100%" alt="Eight specialist verdict cards, each with a recommendation, conviction, thesis bullets and risks.">

<br>

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img src="docs/assets/architecture-light.svg" width="100%" alt="A ticker fans out to eight specialist agents, three at a time, sharing one data cache. The orchestrator combines a deterministic score, an LLM memo and live catalysts into one verdict, saved to SQLite and JSON.">
</picture>

- **Independent specialists.** Each desk is a standalone OpenAI Agents SDK agent with `AgentVerdict` as its output type. No desk sees another's work.
- **Rate-limit aware fan-out.** A semaphore runs three agents at a time, with exponential backoff on `RateLimitError`.
- **One fetch per run.** Data tools are memoized in a per-run `ContextVar` cache, so eight agents share fetches instead of hammering the APIs.
- **Temporal grounding.** The orchestrator is handed today's date, the current price, and a real catalyst calendar. Any catalyst dated outside the horizon is dropped in code.
- **Dated inputs.** Every reading a desk sees carries its observation date, and the backtest reports an unconditional baseline next to each rule, so a stale number or a fake edge is visible to the chair.
- **Graceful degradation.** A failed desk is excluded and its weight is redistributed. Only the runner can mark a desk as failed. If the orchestrator call fails, a mechanical aggregation takes its place.

<br>

## How the verdict is computed

The committee score is a weighted sum, computed in code:

```
score = sum( sign(recommendation) * conviction * weight ) / 100
```

`sign` is +1 for BUY, 0 for HOLD and -1 for SELL. Conviction runs 1 to 10 and weights sum to 100, so the score ranges from -10 to +10.

The weights are fixed in code. The orchestrator cannot change them. It reads the score and all eight verdicts, then writes a 150 to 300 word memo and makes the final call. That call is a judgment and may depart from the score, but a call against the score's sign has to come with a written rationale.

After the LLM call, the score, the conviction-weighted price target, the run timestamp, the current price, the agent verdicts and the sourced catalysts are stamped over whatever the model returned. The model cannot alter them. A call against the score's sign with no rationale is flagged on the report.

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
| yfinance | Price history, annual and quarterly statements, valuation multiples, options chains, analyst targets, peer sets, earnings and dividend calendar |
| SEC EDGAR | Filing index: latest 10-K and 10-Q dates, recent 8-Ks with item codes |
| FRED | Macro series, each with its observation date: DGS10, DGS2, T10Y2Y, CPI, UNRATE, FEDFUNDS, NFCI, VIX |

<br>

## Testing

```bash
uv run pytest -m "not slow"   # unit tests: mocked LLM, mocked data
uv run pytest -m slow         # integration: live yfinance, EDGAR and FRED
uv run pytest                 # everything
```

The mocked tier covers all eight agents, the orchestrator, the runner, the scoring math, persistence, the cache and the schemas.

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
│   ├── schemas.py              # Pydantic: AgentVerdict, OrchestratorVerdict
│   ├── exceptions.py
│   ├── scoring.py              # deterministic weighted score
│   ├── orchestrator.py         # LLM synthesizer + mechanical fallback
│   ├── runner.py               # async fan-out with rate-limit handling
│   ├── persistence.py          # SQLite + JSON snapshots
│   ├── ui.css                  # web UI stylesheet
│   ├── data/
│   │   ├── cache.py            # per-run ContextVar cache
│   │   ├── yfinance_tools.py
│   │   ├── edgar_tools.py
│   │   ├── fred_tools.py
│   │   └── calendar_tools.py   # real upcoming catalysts
│   └── agents/
│       ├── base.py             # agent factory + shared rules
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

EquityTrader is a research and engineering project. Its output is generated by language models, can be wrong, and is not investment advice.

The desks are stylized personas. This project is not affiliated with, endorsed by, or representative of J.P. Morgan, Bridgewater Associates, Goldman Sachs, Citadel, Jane Street, D. E. Shaw, Renaissance Technologies or Two Sigma. Those names belong to their respective owners.

<br>

<div align="center">

Built with [Claude Code](https://claude.com/claude-code), the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) and [Gradio](https://gradio.app/).

[MIT](LICENSE). Copyright (c) 2026 Jeff Lenamon.

</div>
