from dotenv import load_dotenv

load_dotenv()  # Must load before importing equity_trader.runner, which builds the agents

import argparse
import asyncio
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from equity_trader.config import AGENT_NAMES, AGENT_PERSONAS
from equity_trader.exceptions import DataFetchError
from equity_trader.runner import analyze_ticker
from equity_trader.schemas import OrchestratorVerdict

# A redirected stdout on Windows defaults to cp1252, which cannot encode the
# minus signs and arrows that appear in model output.
sys.stdout.reconfigure(encoding="utf-8")
console = Console()

_REC_STYLE = {"BUY": "bold green", "HOLD": "bold yellow", "SELL": "bold red"}


def _usd(x: float | None) -> str:
    return f"${x:,.2f}" if x else "-"


def _bullets(title: str, items: list[str], style: str = "bold") -> None:
    if items:
        console.print(f"\n[{style}]{title}[/]")
        for item in items:
            console.print(f"  - {item}")


def _pct(x: float | None) -> str:
    return f"{x * 100:+.1f}%" if x is not None else "-"


def _events(v: OrchestratorVerdict) -> None:
    """Recent filings and heavy-volume sessions with their price reaction."""
    if v.recent_events is None:
        console.print("\n[bold]Recent events[/]\n  Recent events could not be gathered "
                      "for this run.")
        return
    table = Table(title="Recent events and price reaction", title_justify="left")
    for column in ("Session", "Event", "Gap", "Close", "Volume",
                   f"vs {v.recent_events.sector_etf or 'sector'}", "Flags"):
        table.add_column(column)
    for e in v.recent_events.events:
        flags = ", ".join(label for shown, label in ((e.material, "MATERIAL"),
                                                     (e.unexplained, "UNEXPLAINED")) if shown)
        if e.session is None:
            # A filing no session has traded on yet is dated by the filing.
            table.add_row(e.filings[0].filing_date.isoformat(), escape(e.name),
                          "no session yet", "-", "-", "-", flags)
        else:
            table.add_row(e.session.isoformat(), escape(e.name), _pct(e.opening_gap),
                          _pct(e.close_to_close_return), f"{e.volume_vs_50d_avg:.1f}x",
                          _pct(e.return_vs_sector_etf), flags)
    console.print(table)

    # Headlines for the sessions that need explaining. They are outside text,
    # so they are printed as text, never as markup.
    flagged = [e for e in v.recent_events.events
               if (e.material or e.unexplained) and e.headlines]
    if flagged:
        console.print("\n[bold]Headlines[/] (press reports, not verified)")
    for e in flagged:
        console.print(f"  {e.session.isoformat()}")
        for h in e.headlines:
            console.print(f"    - {escape(h.title)} ({escape(h.source)}, "
                          f"{h.published.isoformat()})", highlight=False)


def render(v: OrchestratorVerdict) -> None:
    rec = f"[{_REC_STYLE[v.final_recommendation]}]{v.final_recommendation}[/]"
    summary = Table.grid(padding=(0, 3))
    summary.add_row("Current price", _usd(v.current_price))
    rests_on = (f" (from {v.price_target_desks} of {len(v.agent_verdicts)} desks)"
                if v.price_target_6mo else "")
    summary.add_row("6-month target", _usd(v.price_target_6mo) + rests_on)
    summary.add_row("Committee score", f"{v.weighted_score:+.2f}  (range -10 to +10)")
    summary.add_row("Position sizing", v.position_sizing_suggestion)
    if v.stop_loss_level:
        summary.add_row("Stop loss", _usd(v.stop_loss_level))
    console.print(Panel(summary, title=f"{v.ticker}  |  {rec}  |  conviction {v.conviction}/10",
                        title_align="left"))

    console.print(Panel(v.synthesis, title="Investment memo", title_align="left"))
    if v.score_departure_rationale:
        console.print(Panel(v.score_departure_rationale, title="Departure from score",
                            title_align="left"))

    desks = Table(title="Specialist verdicts", title_justify="left")
    for column in ("Desk", "Discipline", "Call", "Conviction", "Target", "Weight"):
        desks.add_column(column)
    by_agent = {a.agent: a for a in v.agent_verdicts}
    for name in AGENT_NAMES:
        a = by_agent[name]
        house, discipline = AGENT_PERSONAS[name]
        if a.error_note:
            desks.add_row(house, discipline, "[dim]failed[/]", "-", "-", "-")
        else:
            desks.add_row(house, discipline,
                          f"[{_REC_STYLE[a.recommendation]}]{a.recommendation}[/]",
                          f"{a.conviction}/10", _usd(a.price_target_6mo),
                          f"{v.weights_used.get(name, 0):.0f}%")
    console.print(desks)

    _bullets("Where they agree", v.key_agreements)
    _bullets("Where they split", v.key_disagreements)
    _bullets("Red flags", v.red_flags, style="bold red")
    console.print()
    _events(v)
    _bullets("Catalysts", [f"{c.date.isoformat()}  {c.event}  ({c.expected_impact})"
                           for c in v.catalyst_calendar])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convene the committee on one ticker.")
    parser.add_argument("ticker")
    ticker = parser.parse_args().ticker.upper()

    reported = 0
    with console.status(f"Convening the committee on {ticker}: gathering data") as status:
        def on_gathered(_) -> None:
            status.update(f"Convening the committee on {ticker}: seven desks at work")

        def on_verdict(_) -> None:
            nonlocal reported
            reported += 1
            stage = ("the chair is writing the memo" if reported == len(AGENT_NAMES)
                     else f"{reported} of {len(AGENT_NAMES)} desks reported")
            status.update(f"Convening the committee on {ticker}: {stage}")

        try:
            v = asyncio.run(analyze_ticker(ticker, on_verdict=on_verdict,
                                           on_gathered=on_gathered))
        except DataFetchError as exc:
            console.print(f"[bold red]{exc}[/]")
            sys.exit(1)
    render(v)


if __name__ == "__main__":
    main()
