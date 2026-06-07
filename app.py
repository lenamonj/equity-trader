"""EquityTrader Gradio UI.

Modern-fintech design language: warm paper background, Fraunces display +
Geist body + Geist Mono numerals, navy/gold accents, dominant verdict card,
specialist grid as supporting matter.

Two launch modes:
  uv run python app.py     # production: single process, no hot reload
  uv run gradio app.py     # development: hot reload on .py changes
"""
import html
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # Must load before importing equity_trader.runner which triggers agent imports

from equity_trader.persistence import list_runs
from equity_trader.runner import analyze_ticker
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict


# Each specialist's house + discipline. Rendered as eyebrow text on the agent cards.
_AGENT_PERSONAS: dict[str, tuple[str, str]] = {
    "jpm_fundamental":      ("J.P. Morgan",              "Fundamental"),
    "bridgewater_macro":    ("Bridgewater Associates",   "All-Weather Macro"),
    "gs_technical":         ("Goldman Sachs",            "Technical"),
    "citadel_quant":        ("Citadel",                  "Quantitative"),
    "renaissance_pattern":  ("Renaissance Technologies", "Pattern Recognition"),
    "de_shaw_options":      ("D. E. Shaw",               "Options-Derived"),
    "two_sigma_backtest":   ("Two Sigma",                "Backtest Sanity"),
}

# Default weights order (heaviest first) so the grid reads top-to-bottom by importance.
_AGENT_ORDER = [
    "jpm_fundamental", "bridgewater_macro", "gs_technical", "citadel_quant",
    "renaissance_pattern", "de_shaw_options", "two_sigma_backtest",
]


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

/* ============================================================
   Tokens
   ============================================================ */
:root {
  /* Paper palette */
  --paper:       #FBFAF7;
  --paper-warm:  #F4F1EA;
  --card:        #FFFFFF;
  --border:      #EAE5DC;
  --border-mid:  #D8D1C4;
  --border-strong: #C5BBA8;

  /* Ink (warm-neutral) */
  --ink-1:       #14110D;
  --ink-2:       #2E2A24;
  --ink-3:       #57514A;
  --ink-4:       #8A8378;
  --ink-5:       #B8B0A2;

  /* Brand */
  --navy:        #0F1B33;
  --navy-soft:   #233252;
  --gold:        #B08D57;
  --gold-soft:   #D9C094;

  /* Signal */
  --buy:         #14693D;
  --buy-bg:      rgba(20, 105, 61, 0.07);
  --buy-bd:      rgba(20, 105, 61, 0.22);
  --hold:        #8B6914;
  --hold-bg:     rgba(139, 105, 20, 0.07);
  --hold-bd:     rgba(139, 105, 20, 0.22);
  --sell:        #A21F1F;
  --sell-bg:     rgba(162, 31, 31, 0.07);
  --sell-bd:     rgba(162, 31, 31, 0.22);

  /* Type */
  --font-display: 'Fraunces', 'Times New Roman', Georgia, serif;
  --font-body:    'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:    'Geist Mono', 'JetBrains Mono', ui-monospace, monospace;

  /* Geometry */
  --radius-sm:   2px;
  --radius:      4px;
  --radius-lg:   6px;
}

/* ============================================================
   Reset Gradio defaults
   ============================================================ */
.gradio-container,
.gradio-container * {
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.gradio-container {
  background: var(--paper) !important;
  color: var(--ink-1) !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}
.gradio-container .prose { color: var(--ink-2); }
footer, .show-api, .built-with, .footer { display: none !important; }
.tabs > .tab-nav { border-bottom: 1px solid var(--border) !important; padding: 0 32px !important; }
.tabs > .tab-nav button {
  font-family: var(--font-body) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--ink-4) !important;
  padding: 18px 0 !important;
  margin-right: 32px !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: color 0.2s ease, border-color 0.2s ease !important;
}
.tabs > .tab-nav button:hover { color: var(--ink-2) !important; }
.tabs > .tab-nav button.selected {
  color: var(--navy) !important;
  border-bottom-color: var(--gold) !important;
}

/* ============================================================
   Brand bar
   ============================================================ */
.brand-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 28px 32px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--paper);
}
.brand {
  display: flex;
  align-items: center;
  gap: 18px;
}
.brand-mark {
  width: 44px;
  height: 44px;
  border: 1.5px solid var(--navy);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 18px;
  letter-spacing: -0.02em;
  color: var(--navy);
  background: var(--card);
  position: relative;
}
.brand-mark::after {
  content: '';
  position: absolute;
  inset: 3px;
  border: 0.5px solid var(--navy);
  pointer-events: none;
}
.brand-name {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 22px;
  color: var(--navy);
  letter-spacing: -0.025em;
  line-height: 1;
}
.brand-sub {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-top: 6px;
}
.brand-clock {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-4);
}
.brand-clock-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--buy);
  margin-right: 8px;
  vertical-align: 1px;
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.4; transform: scale(0.85); }
}

/* ============================================================
   Input bar
   ============================================================ */
.input-row { padding: 24px 32px 8px !important; gap: 12px !important; align-items: stretch !important; }

/* Reset Gradio's default wrapper chrome around the textbox */
.ticker-input,
.ticker-input > .block,
.ticker-input > .form,
.ticker-input .wrap,
.ticker-input .container,
.ticker-input > div {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  border-radius: 0 !important;
  min-height: 0 !important;
}

.ticker-input textarea,
.ticker-input input,
.ticker-input input[type="text"] {
  font-family: var(--font-mono) !important;
  font-size: 18px !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  background: var(--card) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--radius) !important;
  padding: 0 20px !important;
  color: var(--ink-1) !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
  height: 56px !important;
  width: 100% !important;
  display: block !important;
  resize: none !important;
  line-height: 56px !important;
  overflow: hidden !important;
}
.ticker-input textarea::placeholder,
.ticker-input input::placeholder {
  color: var(--ink-5) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-style: italic !important;
}
.ticker-input textarea:focus,
.ticker-input input:focus {
  border-color: var(--navy) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(15, 27, 51, 0.06) !important;
}
.ticker-input label,
.ticker-input span[data-testid="block-info"] { display: none !important; }

.run-btn {
  background: var(--navy) !important;
  color: var(--paper) !important;
  font-family: var(--font-body) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  padding: 0 28px !important;
  height: 56px !important;
  border-radius: var(--radius) !important;
  border: none !important;
  transition: background 0.2s ease, transform 0.1s ease !important;
}
.run-btn:hover { background: var(--navy-soft) !important; }
.run-btn:active { transform: translateY(1px) !important; }

/* ============================================================
   Report container
   ============================================================ */
.report { padding: 8px 32px 48px !important; }
.report > * { animation: rise 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
.report > *:nth-child(2) { animation-delay: 0.08s; }
.report > *:nth-child(3) { animation-delay: 0.16s; }
.report > *:nth-child(4) { animation-delay: 0.24s; }
@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ============================================================
   Verdict card (the hero)
   ============================================================ */
.verdict-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 40px 48px 36px;
  margin-top: 16px;
  position: relative;
  overflow: hidden;
}
.verdict-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--gold) 0%, var(--gold-soft) 30%, transparent 70%);
}

.verdict-top {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 48px;
  align-items: end;
  padding-bottom: 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}
.ticker-symbol {
  font-family: var(--font-display);
  font-size: 76px;
  font-weight: 500;
  letter-spacing: -0.045em;
  line-height: 0.95;
  color: var(--navy);
  font-feature-settings: 'ss01' on, 'cv11' on;
}
.ticker-meta {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-top: 12px;
}

.rec-block {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 14px;
  min-width: 260px;
}
.rec-pill {
  display: inline-block;
  font-family: var(--font-mono);
  font-weight: 500;
  letter-spacing: 0.18em;
  border: 1px solid;
  border-radius: var(--radius-sm);
  background: var(--card);
}
.rec-pill-lg { font-size: 14px; padding: 10px 22px; }
.rec-pill-sm { font-size: 10px; padding: 4px 10px; letter-spacing: 0.16em; }
.rec-buy  { color: var(--buy);  background: var(--buy-bg);  border-color: var(--buy-bd); }
.rec-hold { color: var(--hold); background: var(--hold-bg); border-color: var(--hold-bd); }
.rec-sell { color: var(--sell); background: var(--sell-bg); border-color: var(--sell-bd); }

.conviction-line {
  display: flex;
  align-items: baseline;
  gap: 6px;
}
.conv-num {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 32px;
  color: var(--ink-1);
  letter-spacing: -0.025em;
  line-height: 1;
}
.conv-denom {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--ink-4);
}
.conv-label {
  font-family: var(--font-body);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-left: 10px;
}
.conv-track {
  width: 240px;
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
  position: relative;
}
.conv-fill {
  height: 100%;
  background: var(--ink-1);
  border-radius: 2px;
  transform-origin: left;
  animation: fill-bar 0.9s cubic-bezier(0.16, 1, 0.3, 1) both;
  animation-delay: 0.3s;
}
@keyframes fill-bar {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

.verdict-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr 1.4fr;
  gap: 48px;
}
.metric {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.metric-label {
  font-family: var(--font-body);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-4);
}
.metric-value {
  font-family: var(--font-mono);
  font-size: 32px;
  font-weight: 500;
  color: var(--ink-1);
  letter-spacing: -0.02em;
  line-height: 1.05;
  font-feature-settings: 'tnum' on, 'lnum' on;
}
.metric-delta {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.04em;
  margin-top: 2px;
}
.delta-up   { color: var(--buy); }
.delta-down { color: var(--sell); }

.score-bar {
  position: relative;
  height: 16px;
  margin: 6px 0 2px;
}
.score-track {
  position: absolute;
  top: 7px;
  left: 0; right: 0;
  height: 2px;
  background: var(--border);
}
.score-center {
  position: absolute;
  top: 0; bottom: 0;
  left: 50%;
  width: 1px;
  background: var(--ink-5);
}
.score-marker {
  position: absolute;
  top: 2px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transform: translateX(-50%);
  border: 2px solid var(--card);
}
.score-marker-pos { background: var(--buy); box-shadow: 0 0 0 1px var(--buy-bd); }
.score-marker-neg { background: var(--sell); box-shadow: 0 0 0 1px var(--sell-bd); }
.score-marker-zero { background: var(--ink-4); box-shadow: 0 0 0 1px var(--ink-5); }

.score-scale {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.1em;
  color: var(--ink-5);
  margin-top: 4px;
}

/* ============================================================
   Synthesis + sidebar
   ============================================================ */
.synthesis-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 36px 40px;
  margin-top: 20px;
}
.synthesis-grid {
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 48px;
}

.section-title {
  font-family: var(--font-body);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

.synthesis-text {
  font-family: var(--font-display);
  font-size: 17px;
  line-height: 1.65;
  color: var(--ink-2);
  font-weight: 400;
  letter-spacing: -0.003em;
  font-feature-settings: 'ss01' on;
  margin-bottom: 32px;
}

.signals-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 24px;
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
}
.signal-block .signal-title {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-bottom: 10px;
}
.signal-block ul {
  margin: 0;
  padding: 0;
  list-style: none;
}
.signal-block li {
  font-family: var(--font-body);
  font-size: 13px;
  line-height: 1.55;
  color: var(--ink-2);
  padding: 6px 0 6px 14px;
  position: relative;
}
.signal-block li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 14px;
  width: 6px;
  height: 1px;
  background: var(--ink-4);
}
.signal-warn .signal-title { color: var(--sell); }
.signal-warn li::before { background: var(--sell); }

.synthesis-sidebar {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding-left: 36px;
  border-left: 1px solid var(--border);
}
.sb-block .sb-label {
  font-family: var(--font-body);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-bottom: 6px;
}
.sb-block .sb-value {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 500;
  color: var(--ink-1);
  letter-spacing: -0.01em;
  line-height: 1.3;
}
.sb-block .sb-value-mono {
  font-family: var(--font-mono);
  font-size: 22px;
  font-weight: 500;
  letter-spacing: -0.02em;
}
.sb-block .sb-sub {
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--ink-4);
  margin-top: 2px;
}

/* ============================================================
   Catalyst timeline
   ============================================================ */
.timeline-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 32px 40px;
  margin-top: 20px;
}
.timeline { margin-top: 4px; }
.timeline-event {
  display: grid;
  grid-template-columns: 72px 1fr auto;
  gap: 32px;
  align-items: center;
  padding: 18px 0;
  border-bottom: 1px dashed var(--border);
  transition: background 0.15s ease;
}
.timeline-event:last-child { border-bottom: none; }
.timeline-event:hover { background: var(--paper); }
.event-date {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  border-left: 2px solid var(--gold);
  padding-left: 14px;
}
.event-month {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  color: var(--gold);
}
.event-day {
  font-family: var(--font-display);
  font-size: 28px;
  font-weight: 500;
  color: var(--ink-1);
  letter-spacing: -0.03em;
  line-height: 1;
  margin-top: 2px;
}
.event-name {
  font-family: var(--font-display);
  font-size: 17px;
  font-weight: 500;
  color: var(--ink-1);
  letter-spacing: -0.01em;
}
.event-notes {
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--ink-4);
  margin-top: 4px;
  letter-spacing: 0;
}
.event-impact {
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.16em;
  padding: 4px 10px;
  border-radius: var(--radius-sm);
}
.impact-high   { color: var(--navy); background: rgba(15, 27, 51, 0.08); }
.impact-medium { color: var(--ink-3); background: var(--paper-warm); }
.impact-low    { color: var(--ink-4); background: var(--paper-warm); }

/* ============================================================
   Agent grid
   ============================================================ */
.section-title-grid {
  margin: 36px 0 20px;
  font-size: 11px;
}
.agent-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}
.agent-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 22px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 280px;
  transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
}
.agent-card:hover {
  border-color: var(--border-strong);
  transform: translateY(-2px);
  box-shadow: 0 12px 28px -8px rgba(15, 27, 51, 0.08);
}
.agent-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 10px;
}
.agent-house {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--navy);
  letter-spacing: -0.012em;
  line-height: 1.15;
}
.agent-discipline {
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-top: 3px;
}
.agent-metrics {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding-top: 4px;
}
.agent-conv {
  display: flex;
  align-items: baseline;
  gap: 2px;
}
.agent-conv .conv-num-sm {
  font-family: var(--font-display);
  font-size: 28px;
  font-weight: 600;
  color: var(--ink-1);
  letter-spacing: -0.03em;
  line-height: 1;
}
.agent-conv .conv-denom-sm {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-4);
}
.agent-target {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 500;
  color: var(--ink-2);
  letter-spacing: -0.01em;
  font-feature-settings: 'tnum' on;
}
.agent-target-label {
  font-family: var(--font-body);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-bottom: 2px;
}
.mini-bar {
  height: 2px;
  background: var(--border);
  position: relative;
  overflow: hidden;
}
.mini-bar > div {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  background: var(--ink-2);
  transform-origin: left;
  animation: fill-bar 0.9s cubic-bezier(0.16, 1, 0.3, 1) both;
  animation-delay: 0.2s;
}
.agent-card .thesis-list,
.agent-card .risks-list {
  margin: 0;
  padding: 0;
  list-style: none;
}
.agent-card .thesis-list li,
.agent-card .risks-list li {
  font-family: var(--font-body);
  font-size: 12px;
  line-height: 1.45;
  color: var(--ink-2);
  padding: 4px 0 4px 12px;
  position: relative;
}
.agent-card .thesis-list li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 11px;
  width: 5px;
  height: 1px;
  background: var(--ink-3);
}
.agent-card .risks-list li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 11px;
  width: 5px;
  height: 1px;
  background: var(--sell);
}
.risks-block {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
}
.risks-label {
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--sell);
  margin-bottom: 4px;
}

.agent-card-error,
.agent-card-empty {
  opacity: 0.65;
}
.agent-card-error .agent-error {
  margin-top: 8px;
  padding: 10px 12px;
  background: var(--sell-bg);
  border: 1px solid var(--sell-bd);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--sell);
  line-height: 1.4;
}
.agent-card-error .error-label {
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.agent-placeholder {
  font-family: var(--font-display);
  font-style: italic;
  font-size: 14px;
  color: var(--ink-4);
  margin-top: 8px;
}

/* ============================================================
   Welcome state
   ============================================================ */
.welcome-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 64px 56px 56px;
  margin-top: 16px;
  position: relative;
  overflow: hidden;
}
.welcome-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--gold) 0%, var(--gold-soft) 30%, transparent 70%);
}
.welcome-eyebrow {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 18px;
}
.welcome-title {
  font-family: var(--font-display);
  font-size: 44px;
  font-weight: 500;
  color: var(--navy);
  letter-spacing: -0.025em;
  line-height: 1.08;
  margin: 0 0 24px;
  max-width: 720px;
}
.welcome-body {
  font-family: var(--font-display);
  font-size: 17px;
  line-height: 1.65;
  color: var(--ink-3);
  font-weight: 400;
  max-width: 640px;
  margin: 0 0 32px;
}
.welcome-roster {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px 32px;
  padding-top: 32px;
  border-top: 1px solid var(--border);
}
.roster-item {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
  position: relative;
  padding-left: 14px;
}
.roster-item::before {
  content: '';
  position: absolute;
  left: 0;
  top: 7px;
  width: 6px;
  height: 1px;
  background: var(--gold);
}
.roster-item .roster-discipline {
  display: block;
  font-family: var(--font-body);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  color: var(--ink-4);
  margin-top: 3px;
}

/* ============================================================
   Loading state
   ============================================================ */
.loading-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 64px 48px;
  margin-top: 16px;
  text-align: center;
}
.loading-mark {
  width: 56px;
  height: 56px;
  border: 1.5px solid var(--navy);
  border-top-color: transparent;
  border-radius: 50%;
  margin: 0 auto 24px;
  animation: spin 1.1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-title {
  font-family: var(--font-display);
  font-size: 28px;
  font-weight: 500;
  color: var(--navy);
  letter-spacing: -0.02em;
  margin-bottom: 8px;
}
.loading-sub {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  margin-bottom: 36px;
}
.loading-roster {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px 14px;
  max-width: 720px;
  margin: 0 auto;
}
.loading-roster span {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-4);
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--paper);
  animation: shimmer 1.6s ease-in-out infinite;
}
.loading-roster span:nth-child(2) { animation-delay: 0.1s; }
.loading-roster span:nth-child(3) { animation-delay: 0.2s; }
.loading-roster span:nth-child(4) { animation-delay: 0.3s; }
.loading-roster span:nth-child(5) { animation-delay: 0.4s; }
.loading-roster span:nth-child(6) { animation-delay: 0.5s; }
.loading-roster span:nth-child(7) { animation-delay: 0.6s; }
@keyframes shimmer {
  0%, 100% { opacity: 0.5; }
  50%      { opacity: 1; color: var(--navy); border-color: var(--gold); }
}

/* ============================================================
   History table
   ============================================================ */
.history-wrap { padding: 24px 32px 48px !important; }
.history-table table {
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  border-collapse: collapse !important;
  width: 100%;
  background: var(--card) !important;
}
.history-table th {
  font-family: var(--font-body) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: var(--ink-4) !important;
  background: var(--paper-warm) !important;
  padding: 14px 16px !important;
  text-align: left !important;
  border-bottom: 1px solid var(--border) !important;
}
.history-table td {
  padding: 12px 16px !important;
  border-bottom: 1px solid var(--border) !important;
  color: var(--ink-2) !important;
  font-feature-settings: 'tnum' on !important;
}
.history-table tr:hover td { background: var(--paper) !important; }
"""


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


def _fmt_delta(target, current) -> str:
    if target is None or current is None or current == 0:
        return ""
    pct = (target / current - 1.0) * 100.0
    arrow = "&#9650;" if pct >= 0 else "&#9660;"
    klass = "delta-up" if pct >= 0 else "delta-down"
    return f'<div class="metric-delta {klass}">{arrow} {pct:+.2f}% implied</div>'


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


def _loading_html(ticker: str) -> str:
    roster = "".join(
        f"<span>{_e(house)}</span>" for house, _ in _AGENT_PERSONAS.values()
    )
    return f"""
    <div class="loading-card">
      <div class="loading-mark"></div>
      <div class="loading-title">Convening the committee on {_e(ticker.upper())}</div>
      <div class="loading-sub">Seven specialists working in parallel . ~20s</div>
      <div class="loading-roster">{roster}</div>
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
    if v.weight_overrides_rationale:
        sidebar.append(
            f'<div class="sb-block">'
            f'<div class="sb-label">Weight Override</div>'
            f'<div class="sb-value" style="font-size:14px">{_e(v.weight_overrides_rationale)}</div>'
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
        + _catalyst_html(v)
        + '<div class="section-title section-title-grid">Specialist Verdicts</div>'
        + f'<div class="agent-grid">{cards}</div>'
    )


# ============================================================
# Gradio callbacks
# ============================================================

def _show_loading(ticker: str) -> str:
    if not ticker or not ticker.strip():
        return _welcome_html()
    return _loading_html(ticker.strip())


async def analyze(ticker: str) -> str:
    if not ticker or not ticker.strip():
        return _welcome_html()
    v = await analyze_ticker(ticker.strip().upper())
    return _render_report(v)


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

with gr.Blocks(
    title="EquityTrader",
    theme=gr.themes.Base(
        font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
    ),
    css=_CSS,
    fill_width=True,
) as demo:
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

        # Show loading state instantly on click, then run the real analysis.
        run_btn.click(_show_loading, inputs=ticker, outputs=report).then(
            analyze, inputs=ticker, outputs=report
        )
        ticker.submit(_show_loading, inputs=ticker, outputs=report).then(
            analyze, inputs=ticker, outputs=report
        )

    with gr.Tab("History"):
        with gr.Row(elem_classes="history-wrap"):
            gr.Dataframe(
                value=history_table,
                headers=["Ticker", "Timestamp", "Rec", "Conviction", "Score", "Target"],
                every=10,
                elem_classes="history-table",
                interactive=False,
                wrap=False,
            )


if __name__ == "__main__":
    demo.launch()
