"""The web UI's HTML builders and its streaming callback."""
from datetime import UTC, date, datetime

import pytest

import app
from equity_trader.config import AGENT_NAMES, AGENT_WEIGHTS
from equity_trader.exceptions import TickerNotFoundError
from equity_trader.schemas import AgentVerdict, CatalystEvent, OrchestratorVerdict


def _desk(name, rec="HOLD", err=None):
    return AgentVerdict(agent=name, ticker="NVDA", recommendation=rec, conviction=6,
                        price_target_6mo=210.0, thesis=["<b>thesis</b>"], risks=["risk"],
                        data_cited=["cite"], error_note=err)


def _verdict():
    return OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime(2026, 11, 17, tzinfo=UTC),
        current_price=190.0, final_recommendation="BUY", conviction=7,
        price_target_6mo=210.0, weighted_score=3.5, weights_used=AGENT_WEIGHTS,
        score_departure_rationale=None,
        synthesis="First paragraph of the memo.\n\nSecond paragraph of the memo, long enough.",
        key_agreements=["agree"], key_disagreements=["split"],
        dominant_drivers=["jpm_fundamental"], red_flags=["flag"],
        position_sizing_suggestion="starter (~1%)", stop_loss_level=182.0,
        catalyst_calendar=[CatalystEvent(event="Earnings release", date=date(2026, 12, 9),
                                         expected_impact="HIGH")],
        agent_verdicts=[_desk(n) for n in AGENT_NAMES],
    )


def test_report_renders_every_desk_and_escapes_model_text():
    html = app._render_report(_verdict())
    for house, _ in app._AGENT_PERSONAS.values():
        assert house in html
    assert "Earnings release" in html
    assert "<b>thesis</b>" not in html
    assert "&lt;b&gt;thesis&lt;/b&gt;" in html


def test_loading_card_shows_calls_of_desks_that_reported():
    html = app._loading_html("NVDA", {"jpm_fundamental": _desk("jpm_fundamental", "BUY"),
                                      "gs_technical": _desk("gs_technical", err="boom")})
    assert "2 of 8 desks reported" in html
    assert "J.P. Morgan . BUY" in html
    assert "Goldman Sachs . failed" in html


@pytest.mark.asyncio
async def test_analyze_streams_progress_then_the_report(monkeypatch):
    async def fake_run(ticker, on_verdict=None, on_gathered=None):
        on_gathered([])
        for name in AGENT_NAMES:
            on_verdict(_desk(name))
        return _verdict()

    monkeypatch.setattr(app, "analyze_ticker", fake_run)
    pages = [page async for page in app.analyze(" nvda ")]
    assert "Convening the committee on NVDA" in pages[0]
    assert "The gatherer is collecting the data" in pages[0]
    assert "Eight specialists at work" in pages[1]
    assert "verdict-card" in pages[-1]


@pytest.mark.asyncio
async def test_analyze_shows_an_error_card_for_an_unknown_ticker(monkeypatch):
    async def fake_run(ticker, on_verdict=None, on_gathered=None):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(app, "analyze_ticker", fake_run)
    pages = [page async for page in app.analyze("zzzz")]
    assert "could not convene on ZZZZ" in pages[-1]
    assert "Ticker not found: ZZZZ" in pages[-1]
