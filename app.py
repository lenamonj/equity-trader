"""EquityTrader Gradio UI.

Modern-fintech design language: warm paper background, Fraunces display +
Geist body + Geist Mono numerals, navy/gold accents, dominant verdict card,
specialist grid as supporting matter.

Two launch modes:
  uv run python app.py     # production: single process, no hot reload
  uv run gradio app.py     # development: hot reload on .py changes
"""
import asyncio
import html
import logging
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # Must load before importing equity_trader.runner which triggers agent imports

from equity_trader.config import AGENT_NAMES, AGENT_PERSONAS
from equity_trader.exceptions import CommitteeFailedError, DataFetchError
from equity_trader.persistence import list_runs
from equity_trader.runner import analyze_ticker
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict, RecentEvent

log = logging.getLogger(__name__)

_AGENT_PERSONAS = AGENT_PERSONAS
_AGENT_ORDER = AGENT_NAMES


_CSS_PATH = Path(__file__).parent / "equity_trader" / "ui.css"


_BRAND_HEADER = """
<header class="brand-bar">
  <div class="brand">
    <div class="brand-mark">EQ</div>
    <div>
      <div class="brand-name">EquityTrader</div>
      <div class="brand-sub">Multi-Strategy Committee . 3-6 Month Horizon</div>
    </div>
  </div>
  <div class="brand-clock">
    <span class="brand-clock-dot"></span>Live
  </div>
</header>
"""


# ============================================================
# Helpers
# ============================================================

def _e(s) -> str:
    """HTML-escape any value. Treat None as empty string."""
    if s is None:
        return ""
    return html.escape(str(s))


def _fmt_usd(x) -> str:
    if x is None:
        return "-"
    return f"${x:,.2f}"


def _fmt_reading(value: float, unit: str) -> str:
    """A cited reading with its unit, large dollar amounts shortened."""
    if unit == "usd":
        for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= size:
                return f"${value / size:,.1f}{suffix}"
        return f"${value:,.2f}"
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "fraction":
        return f"{value * 100:,.2f}%"
    return f"{value:,.2f}" + ("%" if unit == "percent" else "")


def _fmt_delta(target, current) -> str:
    if target is None or current is None or current == 0:
        return ""
    pct = (target / current - 1.0) * 100.0
    arrow = "&#9650;" if pct >= 0 else "&#9660;"
    klass = "delta-up" if pct >= 0 else "delta-down"
    return f'<div class="metric-delta {klass}">{arrow} {pct:+.2f}% implied</div>'


def _target_note(v: OrchestratorVerdict) -> str:
    """How many desks the target rests on: it is an average of those that gave one."""
    if v.price_target_6mo is None:
        return '<div class="metric-note">No desk gave a target</div>'
    return (f'<div class="metric-note">from {v.price_target_desks} of '
            f"{len(v.agent_verdicts)} desks</div>")


def _rec_pill(rec: str, size: str = "lg") -> str:
    klass_map = {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}
    klass = klass_map.get(rec, "rec-hold")
    size_klass = f"rec-pill-{size}"
    return f'<span class="rec-pill {size_klass} {klass}">{_e(rec)}</span>'


def _conviction_track(conviction: int, width: int = 240) -> str:
    pct = max(0, min(10, conviction)) * 10
    return (
        f'<div class="conv-track" style="width: {width}px">'
        f'<div class="conv-fill" style="width: {pct}%"></div>'
        f'</div>'
    )


def _mini_bar(conviction: int) -> str:
    pct = max(0, min(10, conviction)) * 10
    return f'<div class="mini-bar"><div style="width: {pct}%"></div></div>'


def _score_bar(score: float) -> str:
    clamped = max(-10.0, min(10.0, score))
    position = 50.0 + clamped * 5.0
    if score > 0.2:
        marker_klass = "score-marker-pos"
    elif score < -0.2:
        marker_klass = "score-marker-neg"
    else:
        marker_klass = "score-marker-zero"
    return (
        f'<div class="score-bar">'
        f'<div class="score-track"></div>'
        f'<div class="score-center"></div>'
        f'<div class="score-marker {marker_klass}" style="left: {position}%"></div>'
        f'</div>'
        f'<div class="score-scale"><span>-10</span><span>0</span><span>+10</span></div>'
    )


# ============================================================
# Section renderers
# ============================================================

def _welcome_html() -> str:
    roster_items = "".join(
        f'<div class="roster-item">{_e(house)}'
        f'<span class="roster-discipline">{_e(discipline)}</span></div>'
        for house, discipline in _AGENT_PERSONAS.values()
    )
    return f"""
    <div class="welcome-card">
      <div class="welcome-eyebrow">A Multi-Strategy Investment Committee</div>
      <h1 class="welcome-title">Seven specialists.<br>One ticker. One verdict.</h1>
      <p class="welcome-body">EquityTrader convenes seven specialist analysts modeled
      after the desks of the world's most respected investment houses, then
      synthesizes their structured verdicts into a single Buy, Hold, or Sell
      recommendation over a 3-6 month horizon.</p>
      <div class="welcome-roster">{roster_items}</div>
    </div>
    """


def _loading_html(ticker: str, reported: dict[str, AgentVerdict] | None = None,
                  gathered: bool = True) -> str:
    """Loading card. Desks that have reported show their call."""
    reported = reported or {}
    chips = []
    for name, (house, _) in _AGENT_PERSONAS.items():
        av = reported.get(name)
        if av is None:
            chips.append(f"<span>{_e(house)}</span>")
        elif av.error_note:
            chips.append(f'<span class="done failed">{_e(house)} . failed</span>')
        else:
            chips.append(f'<span class="done {av.recommendation.lower()}">'
                         f"{_e(house)} . {av.recommendation}</span>")
    roster = "".join(chips)
    if len(reported) == len(_AGENT_PERSONAS):
        stage = "All desks reported . the chair is writing the memo"
    elif reported:
        stage = f"{len(reported)} of {len(_AGENT_PERSONAS)} desks reported"
    elif gathered:
        stage = "Seven specialists at work, three at a time"
    else:
        stage = "The gatherer is collecting the data"
    return f"""
    <div class="loading-card">
      <div class="loading-mark"></div>
      <div class="loading-title">Convening the committee on {_e(ticker.upper())}</div>
      <div class="loading-sub">{_e(stage)}</div>
      <div class="loading-roster">{roster}</div>
    </div>
    """


def _error_html(ticker: str, message: str) -> str:
    return f"""
    <div class="loading-card error-card">
      <div class="loading-title">The committee could not convene on {_e(ticker)}</div>
      <div class="error-message">{_e(message)}</div>
    </div>
    """


def _verdict_html(v: OrchestratorVerdict) -> str:
    try:
        ts = v.run_timestamp.strftime("%B %d, %Y %H:%M UTC")
    except Exception:
        ts = str(v.run_timestamp)

    return f"""
    <div class="verdict-card">
      <div class="verdict-top">
        <div>
          <div class="ticker-symbol">{_e(v.ticker)}</div>
          <div class="ticker-meta">As of {_e(ts)}</div>
        </div>
        <div class="rec-block">
          {_rec_pill(v.final_recommendation, size="lg")}
          <div class="conviction-line">
            <span class="conv-num">{v.conviction}</span>
            <span class="conv-denom">/ 10</span>
            <span class="conv-label">Conviction</span>
          </div>
          {_conviction_track(v.conviction)}
        </div>
      </div>
      <div class="verdict-metrics">
        <div class="metric">
          <div class="metric-label">Current</div>
          <div class="metric-value">{_fmt_usd(v.current_price)}</div>
        </div>
        <div class="metric">
          <div class="metric-label">6-Month Target</div>
          <div class="metric-value">{_fmt_usd(v.price_target_6mo)}</div>
          {_fmt_delta(v.price_target_6mo, v.current_price)}
          {_target_note(v)}
        </div>
        <div class="metric metric-score">
          <div class="metric-label">Committee Score</div>
          <div class="metric-value">{v.weighted_score:+.2f}</div>
          {_score_bar(v.weighted_score)}
        </div>
      </div>
    </div>
    """


def _synthesis_html(v: OrchestratorVerdict) -> str:
    sidebar = []
    sidebar.append(
        f'<div class="sb-block">'
        f'<div class="sb-label">Position Sizing</div>'
        f'<div class="sb-value">{_e(v.position_sizing_suggestion)}</div>'
        f'</div>'
    )
    if v.stop_loss_level is not None:
        sidebar.append(
            f'<div class="sb-block">'
            f'<div class="sb-label">Stop Loss</div>'
            f'<div class="sb-value sb-value-mono">{_fmt_usd(v.stop_loss_level)}</div>'
            f'<div class="sb-sub">Technical-anchored</div>'
            f'</div>'
        )
    if v.dominant_drivers:
        drivers_text = ", ".join(_AGENT_PERSONAS.get(d, (d, ""))[0]
                                  for d in v.dominant_drivers[:3])
        sidebar.append(
            f'<div class="sb-block">'
            f'<div class="sb-label">Dominant Drivers</div>'
            f'<div class="sb-value">{_e(drivers_text)}</div>'
            f'</div>'
        )
    if v.score_departure_rationale:
        sidebar.append(
            f'<div class="sb-block">'
            f'<div class="sb-label">Departure From Score</div>'
            f'<div class="sb-note">{_e(v.score_departure_rationale)}</div>'
            f'</div>'
        )
    sidebar_html = "".join(sidebar)

    def _block(title, items, klass=""):
        if not items:
            return ""
        lis = "".join(f"<li>{_e(i)}</li>" for i in items)
        return (
            f'<div class="signal-block {klass}">'
            f'<div class="signal-title">{title}</div>'
            f'<ul>{lis}</ul></div>'
        )

    signals = (
        _block("Where they agree", v.key_agreements)
        + _block("Where they split", v.key_disagreements)
        + _block("Red flags", v.red_flags, klass="signal-warn")
    )

    return f"""
    <div class="synthesis-card">
      <div class="section-title">Investment Memo</div>
      <div class="synthesis-grid">
        <div>
          <div class="synthesis-text">{_e(v.synthesis)}</div>
          <div class="signals-row">{signals}</div>
        </div>
        <div class="synthesis-sidebar">{sidebar_html}</div>
      </div>
    </div>
    """


def _catalyst_html(v: OrchestratorVerdict) -> str:
    if not v.catalyst_calendar:
        return ""
    events = sorted(v.catalyst_calendar, key=lambda c: c.date)
    rows = ""
    for c in events:
        impact_klass = {
            "HIGH":   "impact-high",
            "MEDIUM": "impact-medium",
            "LOW":    "impact-low",
        }.get(c.expected_impact, "impact-low")
        notes_html = f'<div class="event-notes">{_e(c.notes)}</div>' if c.notes else ""
        rows += (
            f'<div class="timeline-event">'
            f'  <div class="event-date">'
            f'    <div class="event-month">{c.date.strftime("%b").upper()}</div>'
            f'    <div class="event-day">{c.date.day:02d}</div>'
            f'  </div>'
            f'  <div>'
            f'    <div class="event-name">{_e(c.event)}</div>'
            f'    {notes_html}'
            f'  </div>'
            f'  <div class="event-impact {impact_klass}">{_e(c.expected_impact)}</div>'
            f'</div>'
        )
    return f"""
    <div class="timeline-card">
      <div class="section-title">3-6 Month Catalyst Calendar</div>
      <div class="timeline">{rows}</div>
    </div>
    """


def _fmt_move(x: float | None) -> str:
    """A session return as a signed, colored percentage."""
    if x is None:
        return "-"
    klass = "delta-up" if x >= 0 else "delta-down"
    return f'<b class="{klass}">{x * 100:+.1f}%</b>'


def _event_html(event: RecentEvent, sector_etf: str | None) -> str:
    # A filing no session has traded on yet is dated by the filing.
    when = event.session or event.filings[0].filing_date
    filed = ", ".join(f.filing_date.isoformat() for f in event.filings)
    notes = f"8-K filed {filed}" if filed else ""
    if event.session is None:
        reaction = "No session has traded since this filing"
    else:
        versus = (f"<span>vs {_e(sector_etf)} {_fmt_move(event.return_vs_sector_etf)}</span>"
                  if sector_etf else "")
        reaction = (f"<span>Gap {_fmt_move(event.opening_gap)}</span>"
                    f"<span>Close {_fmt_move(event.close_to_close_return)}</span>"
                    f"<span>Volume <b>{event.volume_vs_50d_avg:.1f}x</b> 50-day avg</span>"
                    f"{versus}")
    if event.headlines is None:
        headlines = '<div class="event-notes">Headlines unavailable</div>'
    else:
        headlines = '<ul class="headline-list">' + "".join(
            f"<li>{_e(h.title)}<span>{_e(h.source)} . {h.published.strftime('%b %d')}</span></li>"
            for h in event.headlines) + "</ul>"
    tags = "".join(
        f'<div class="event-impact {klass}">{label}</div>'
        for shown, label, klass in ((event.material, "MATERIAL", "impact-high"),
                                    (event.unexplained, "UNEXPLAINED", "impact-medium"))
        if shown)
    return (
        f'<div class="timeline-event recent-event">'
        f'  <div class="event-date">'
        f'    <div class="event-month">{when.strftime("%b").upper()}</div>'
        f'    <div class="event-day">{when.day:02d}</div>'
        f'  </div>'
        f'  <div>'
        f'    <div class="event-name">{_e(event.name)}</div>'
        f'    <div class="event-notes">{_e(notes)}</div>'
        f'    <div class="event-reaction">{reaction}</div>'
        f'    {headlines}'
        f'  </div>'
        f'  <div class="event-tags">{tags}</div>'
        f'</div>'
    )


def _events_html(v: OrchestratorVerdict) -> str:
    """Recent filings and heavy-volume sessions with their price reaction. All
    of it is computed in code, so it is on the page whatever the memo says."""
    if v.recent_events is None:
        body = '<div class="event-notes">Recent events could not be gathered for this run.</div>'
    else:
        rows = "".join(_event_html(event, v.recent_events.sector_etf)
                       for event in v.recent_events.events)
        body = (f'<div class="timeline">{rows}</div>'
                '<div class="evidence-note">Reactions are computed in code from SEC filings '
                "and price bars, largest move first. Headlines are press reports from Google "
                "News and are not verified.</div>")
    return f"""
    <div class="timeline-card">
      <div class="section-title">Recent Events and Price Reaction</div>
      {body}
    </div>
    """


def _agent_card_html(name: str, av: AgentVerdict | None) -> str:
    house, discipline = _AGENT_PERSONAS.get(name, (name, ""))

    if av is None:
        return f"""
        <div class="agent-card agent-card-empty">
          <div class="agent-card-header">
            <div>
              <div class="agent-house">{_e(house)}</div>
              <div class="agent-discipline">{_e(discipline)}</div>
            </div>
          </div>
          <div class="agent-placeholder">Awaiting analysis</div>
        </div>
        """

    if av.error_note:
        return f"""
        <div class="agent-card agent-card-error">
          <div class="agent-card-header">
            <div>
              <div class="agent-house">{_e(house)}</div>
              <div class="agent-discipline">{_e(discipline)}</div>
            </div>
            {_rec_pill("HOLD", size="sm")}
          </div>
          <div class="agent-error">
            <div class="error-label">Unavailable</div>
            <div>{_e(av.error_note[:140])}</div>
          </div>
        </div>
        """

    target_html = ""
    if av.price_target_6mo is not None:
        target_html = (
            f'<div style="text-align: right">'
            f'  <div class="agent-target-label">6mo Target</div>'
            f'  <div class="agent-target">{_fmt_usd(av.price_target_6mo)}</div>'
            f'</div>'
        )

    thesis_html = ""
    if av.thesis:
        lis = "".join(f"<li>{_e(t)}</li>" for t in av.thesis[:3])
        thesis_html = f'<ul class="thesis-list">{lis}</ul>'

    evidence_html = ""
    if av.evidence:
        rows = "".join(
            f'<div class="evidence-row"><span class="sig sig-{item.signal}"></span>'
            f'<span class="evidence-metric">{_e(item.metric)}</span>'
            f'<span class="evidence-value">{_e(_fmt_reading(item.value, item.unit))}</span>'
            f"</div>"
            for item in av.evidence[:4])
        found = sum(item.in_data for item in av.evidence)
        evidence_html = (
            f'<div class="evidence-block">'
            f'  <div class="evidence-label">Evidence</div>{rows}'
            f'  <div class="evidence-note">{len(av.evidence)} readings cited . '
            f"{found} found in the data</div>"
            f"</div>")

    risks_html = ""
    if av.risks:
        lis = "".join(f"<li>{_e(r)}</li>" for r in av.risks[:2])
        risks_html = (
            f'<div class="risks-block">'
            f'  <div class="risks-label">Risk</div>'
            f'  <ul class="risks-list">{lis}</ul>'
            f'</div>'
        )

    return f"""
    <div class="agent-card">
      <div class="agent-card-header">
        <div>
          <div class="agent-house">{_e(house)}</div>
          <div class="agent-discipline">{_e(discipline)}</div>
        </div>
        {_rec_pill(av.recommendation, size="sm")}
      </div>
      <div class="agent-metrics">
        <div class="agent-conv">
          <span class="conv-num-sm">{av.conviction}</span>
          <span class="conv-denom-sm">/10</span>
        </div>
        {target_html}
      </div>
      {_mini_bar(av.conviction)}
      {evidence_html}
      {thesis_html}
      {risks_html}
    </div>
    """


def _render_report(v: OrchestratorVerdict) -> str:
    by_agent = {av.agent: av for av in v.agent_verdicts}
    cards = "".join(
        _agent_card_html(name, by_agent.get(name)) for name in _AGENT_ORDER
    )
    return (
        _verdict_html(v)
        + _synthesis_html(v)
        + _events_html(v)
        + _catalyst_html(v)
        + '<div class="section-title section-title-grid">Specialist Verdicts</div>'
        + f'<div class="agent-grid">{cards}</div>'
    )


# ============================================================
# Gradio callbacks
# ============================================================

async def analyze(ticker: str):
    """Stream the loading card as desks report, then the finished report."""
    if not ticker or not ticker.strip():
        yield _welcome_html()
        return
    ticker = ticker.strip().upper()
    reported: dict[str, AgentVerdict] = {}
    # A desk's verdict as it reports, or None when the gatherer has finished.
    updates: asyncio.Queue[AgentVerdict | None] = asyncio.Queue()
    task = asyncio.create_task(analyze_ticker(
        ticker, on_verdict=updates.put_nowait,
        on_gathered=lambda report: updates.put_nowait(None)))
    yield _loading_html(ticker, gathered=False)
    while not task.done():
        try:
            av = await asyncio.wait_for(updates.get(), timeout=0.5)
        except TimeoutError:
            continue
        if av is not None:
            reported[av.agent] = av
        yield _loading_html(ticker, reported)
    try:
        yield _render_report(task.result())
    except (DataFetchError, CommitteeFailedError) as exc:
        yield _error_html(ticker, str(exc))
    except Exception as exc:
        log.exception("Run failed for %s", ticker)
        yield _error_html(ticker, f"{type(exc).__name__}: {exc}")


def history_table():
    rows = list_runs(limit=50)
    return [
        [
            r["ticker"],
            r["run_timestamp"][:16].replace("T", " "),
            r["final_recommendation"],
            r["conviction"],
            f"{r['weighted_score']:+.2f}",
            f"${r['price_target_6mo']:,.2f}" if r["price_target_6mo"] else "-",
        ]
        for r in rows
    ]


# ============================================================
# Layout
# ============================================================

_THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
)

with gr.Blocks(title="EquityTrader", fill_width=True) as demo:
    gr.HTML(_BRAND_HEADER)

    with gr.Tab("Analyze"):
        with gr.Row(elem_classes="input-row", equal_height=True):
            ticker = gr.Textbox(
                label="Ticker",
                placeholder="e.g. AVGO, NVDA, JPM",
                scale=5,
                show_label=False,
                elem_classes="ticker-input",
                lines=1,
                max_lines=1,
            )
            run_btn = gr.Button(
                "Convene Committee",
                variant="primary",
                scale=1,
                elem_classes="run-btn",
            )

        with gr.Row(elem_classes="report"):
            report = gr.HTML(_welcome_html())

        # The loading card is the progress display; hide Gradio's own.
        run_btn.click(analyze, inputs=ticker, outputs=report, show_progress="hidden")
        ticker.submit(analyze, inputs=ticker, outputs=report, show_progress="hidden")

    with gr.Tab("History") as history_tab:
        with gr.Row(elem_classes="history-wrap"):
            history = gr.Dataframe(
                value=history_table,
                headers=["Ticker", "Run (UTC)", "Call", "Conv.", "Score", "Target"],
                elem_classes="history-table",
                interactive=False,
                wrap=False,
            )
    history_tab.select(history_table, outputs=history)


if __name__ == "__main__":
    demo.launch(theme=_THEME, css_paths=[_CSS_PATH])
