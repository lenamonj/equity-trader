"""EquityTrader Gradio UI.

Two launch modes:
  uv run python app.py     # production: single process, no hot reload
  uv run gradio app.py     # development: hot reload on .py changes
"""
import asyncio
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # Must load before importing equity_trader.runner which triggers agent module imports

from equity_trader.runner import analyze_ticker
from equity_trader.persistence import list_runs
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict


def _agent_md(v: AgentVerdict) -> str:
    target = f"${v.price_target_6mo:,.2f}" if v.price_target_6mo else "-"
    err = f"\n\n_Error: {v.error_note}_" if v.error_note else ""
    thesis = "\n".join(f"- {t}" for t in v.thesis)
    risks = "\n".join(f"- {r}" for r in v.risks)
    return (
        f"### {v.agent}\n"
        f"**{v.recommendation}**  -  Conviction {v.conviction}/10  -  Target {target}\n\n"
        f"**Thesis:**\n{thesis}\n\n"
        f"**Risks:**\n{risks}{err}"
    )


def _orch_md(v: OrchestratorVerdict) -> str:
    target = f"${v.price_target_6mo:,.2f}" if v.price_target_6mo else "-"
    stop = f"${v.stop_loss_level:,.2f}" if v.stop_loss_level else "-"
    agreements = "\n".join(f"- {a}" for a in v.key_agreements) or "_(none)_"
    disagree = "\n".join(f"- {d}" for d in v.key_disagreements) or "_(none)_"
    drivers = ", ".join(v.dominant_drivers) or "_(none)_"
    red = "\n".join(f"- {r}" for r in v.red_flags) or "_(none)_"
    catalysts = "\n".join(
        f"- **{c.date}** {c.event} ({c.expected_impact})" for c in v.catalyst_calendar
    ) or "_(none)_"
    return (
        f"# {v.ticker}  -  {v.final_recommendation}  -  Conviction {v.conviction}/10\n"
        f"**Current:** ${v.current_price:,.2f}  -  **6mo Target:** {target}  -  "
        f"**Weighted score:** {v.weighted_score:.2f}\n\n"
        f"## Synthesis\n{v.synthesis}\n\n"
        f"**Position sizing:** {v.position_sizing_suggestion}  -  **Stop:** {stop}\n\n"
        f"**Dominant drivers:** {drivers}\n\n"
        f"### Agreements\n{agreements}\n\n"
        f"### Disagreements\n{disagree}\n\n"
        f"### Red flags\n{red}\n\n"
        f"### Catalysts\n{catalysts}"
    )


async def analyze(ticker: str):
    if not ticker:
        return "Enter a ticker.", *(["" for _ in range(7)])
    v = await analyze_ticker(ticker.upper())
    by_agent = {av.agent: av for av in v.agent_verdicts}
    cards = [_agent_md(by_agent[n]) if n in by_agent else ""
             for n in ["jpm_fundamental", "bridgewater_macro", "gs_technical",
                      "citadel_quant", "renaissance_pattern", "de_shaw_options",
                      "two_sigma_backtest"]]
    return _orch_md(v), *cards


def history_table():
    rows = list_runs(limit=50)
    return [[r["ticker"], r["run_timestamp"], r["final_recommendation"],
             r["conviction"], r["weighted_score"]] for r in rows]


with gr.Blocks(title="EquityTrader") as demo:
    gr.Markdown("# EquityTrader")
    with gr.Tab("Analyze"):
        with gr.Row():
            ticker = gr.Textbox(label="Ticker", placeholder="NVDA", scale=4)
            run_btn = gr.Button("Run", variant="primary", scale=1)
        with gr.Row():
            jpm = gr.Markdown()
            bw = gr.Markdown()
            gst = gr.Markdown()
        with gr.Row():
            cq = gr.Markdown()
            rp = gr.Markdown()
            dso = gr.Markdown()
        with gr.Row():
            tsb = gr.Markdown()
        gr.Markdown("---")
        orch_out = gr.Markdown()
        run_btn.click(analyze, inputs=ticker,
                      outputs=[orch_out, jpm, bw, gst, cq, rp, dso, tsb])

    with gr.Tab("History"):
        gr.Dataframe(value=history_table,
                     headers=["Ticker", "Timestamp", "Rec", "Conviction", "Score"],
                     every=10)


if __name__ == "__main__":
    demo.launch()
