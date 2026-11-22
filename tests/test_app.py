"""The web UI's HTML builders and its streaming callback."""
from datetime import UTC, date, datetime

import pytest

import app
from equity_trader.config import AGENT_NAMES, AGENT_WEIGHTS
from equity_trader.exceptions import TickerNotFoundError
from equity_trader.schemas import (
    AgentVerdict,
    CatalystEvent,
    CheckedEvidence,
    OrchestratorVerdict,
    RecentEvents,
)


def _reading(metric, value, unit, signal, in_data=True):
    return CheckedEvidence(metric=metric, value=value, unit=unit, as_of=date(2026, 11, 17),
                           source="fundamentals", signal=signal, in_data=in_data)


def _desk(name, rec="HOLD", err=None):
    return AgentVerdict(
        agent=name, ticker="NVDA", recommendation=rec, conviction=6, price_target_6mo=210.0,
        evidence=[_reading("Revenue, <i>Q2</i>", 96_221_000_000, "usd", "bullish"),
                  _reading("Gross margin", 75.0, "percent", "bullish"),
                  _reading("Return, 3 months", -0.2901, "fraction", "bearish"),
                  _reading("Forward P/E", 9.69, "ratio", "neutral", in_data=False)],
        thesis=["<b>thesis</b>"], risks=["risk"], error_note=err)


def test_a_desk_card_shows_its_evidence_and_how_much_was_found_in_the_data():
    html = app._agent_card_html("jpm_fundamental", _desk("jpm_fundamental"))
    assert "Revenue, &lt;i&gt;Q2&lt;/i&gt;" in html and "$96.2B" in html
    assert "75.00%" in html and "-29.01%" in html and ">9.69<" in html
    assert 'class="sig sig-bullish"' in html and 'class="sig sig-neutral"' in html
    assert "4 readings cited . 3 found in the data" in html


def _event(rank, session, **changes):
    return {"rank": rank, "session": session, "filings": [], "heavy_volume": True,
            "unexplained": True, "opening_gap": -0.0169, "close_to_close_return": -0.1401,
            "volume_vs_50d_avg": 3.51, "sector_etf_return": -0.0727,
            "return_vs_sector_etf": -0.0674, "move": 0.1401, "material": True,
            "headlines": [], **changes}


def _events():
    return RecentEvents(as_of="2026-11-20", sector_etf="XLK", notes="n", events=[
        _event(1, "2026-11-20", headlines=None),
        _event(2, "2026-11-12", unexplained=False, opening_gap=-0.1308,
               close_to_close_return=-0.0434, volume_vs_50d_avg=5.21,
               return_vs_sector_etf=-0.04, move=0.1308,
               filings=[{"filing_date": "2026-11-12", "excerpt": "Item 5.02. Resigned.",
                         "items": ["5.02 Departure or appointment of directors or officers"]}],
               headlines=[{"title": "Acme <b>chief</b> resigns", "source": "Reuters",
                           "published": "2026-11-12"}]),
        _event(3, None, heavy_volume=False, unexplained=False, opening_gap=None,
               close_to_close_return=None, volume_vs_50d_avg=None, sector_etf_return=None,
               return_vs_sector_etf=None, move=None, material=False,
               filings=[{"filing_date": "2026-11-20", "excerpt": None,
                         "items": ["8.01 Other events"]}]),
    ])


def _verdict():
    return OrchestratorVerdict(
        recent_events=_events(), price_target_desks=3,
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
    assert "Recent Events" in html and "5.02 Departure" in html
    assert "<b>thesis</b>" not in html
    assert "&lt;b&gt;thesis&lt;/b&gt;" in html


def test_loading_card_shows_calls_of_desks_that_reported():
    html = app._loading_html("NVDA", {"jpm_fundamental": _desk("jpm_fundamental", "BUY"),
                                      "gs_technical": _desk("gs_technical", err="boom")})
    assert "2 of 7 desks reported" in html
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
    assert "Seven specialists at work" in pages[1]
    assert "verdict-card" in pages[-1]


@pytest.mark.asyncio
async def test_analyze_shows_an_error_card_for_an_unknown_ticker(monkeypatch):
    async def fake_run(ticker, on_verdict=None, on_gathered=None):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(app, "analyze_ticker", fake_run)
    pages = [page async for page in app.analyze("zzzz")]
    assert "could not convene on ZZZZ" in pages[-1]
    assert "Ticker not found: ZZZZ" in pages[-1]


def test_recent_events_show_each_reaction_as_code_computed_it():
    html = app._events_html(_verdict())
    resignation = html[html.index("5.02 Departure"):html.index("8.01 Other events")]
    assert "8-K filed 2026-11-12" in resignation
    for reading in ("-13.1%", "-4.3%", "5.2x", "vs XLK", "-4.0%"):
        assert reading in resignation, reading
    assert "MATERIAL" in resignation and "UNEXPLAINED" not in resignation
    # A headline is shown with its source, and its text is escaped.
    assert "Acme &lt;b&gt;chief&lt;/b&gt; resigns" in resignation and "Reuters" in resignation


def test_recent_events_mark_what_is_unexplained_or_missing():
    html = app._events_html(_verdict())
    first = html[:html.index("5.02 Departure")]
    assert "Heavy volume, no 8-K filed" in first and "UNEXPLAINED" in first
    assert "-14.0%" in first and "Headlines unavailable" in first
    assert "No session has traded since this filing" in html[html.index("8.01 Other events"):]


def test_a_report_says_so_when_recent_events_could_not_be_gathered():
    html = app._events_html(_verdict().model_copy(update={"recent_events": None}))
    assert "Recent events could not be gathered for this run" in html


def test_the_verdict_card_says_how_many_desks_the_target_rests_on():
    assert "from 3 of 7 desks" in app._verdict_html(_verdict())
    none = _verdict().model_copy(update={"price_target_6mo": None, "price_target_desks": 0})
    html = app._verdict_html(none)
    assert "No desk gave a target" in html and "of 7 desks" not in html
