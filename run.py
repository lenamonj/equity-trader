from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
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
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    args = parser.parse_args()
    with console.status(f"Analyzing {args.ticker.upper()}..."):
        v = asyncio.run(analyze_ticker(args.ticker.upper()))
    render(v)


if __name__ == "__main__":
    main()
