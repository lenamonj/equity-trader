from dotenv import load_dotenv

load_dotenv()  # Must load before importing equity_trader.runner, which builds the agents

import argparse
import asyncio
import sys

from rich.console import Console
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


def render(v: OrchestratorVerdict) -> None:
    rec = f"[{_REC_STYLE[v.final_recommendation]}]{v.final_recommendation}[/]"
    summary = Table.grid(padding=(0, 3))
    summary.add_row("Current price", _usd(v.current_price))
    summary.add_row("6-month target", _usd(v.price_target_6mo))
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
    _bullets("Catalysts", [f"{c.date.isoformat()}  {c.event}  ({c.expected_impact})"
                           for c in v.catalyst_calendar])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convene the committee on one ticker.")
    parser.add_argument("ticker")
    ticker = parser.parse_args().ticker.upper()

    reported = 0
    with console.status(f"Convening the committee on {ticker}: gathering data") as status:
        def on_gathered(_) -> None:
            status.update(f"Convening the committee on {ticker}: eight desks at work")

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
