"""The terminal report."""
from rich.console import Console

import run
from equity_trader.schemas import Headline
from tests.test_app import _events, _verdict


def _printed(monkeypatch, verdict) -> str:
    console = Console(record=True, width=200)
    monkeypatch.setattr(run, "console", console)
    run.render(verdict)
    return console.export_text()


def test_the_terminal_report_lists_recent_events_with_their_reaction(monkeypatch):
    text = _printed(monkeypatch, _verdict())
    resignation = next(line for line in text.splitlines() if "5.02 Departure" in line)
    for reading in ("2026-11-12", "-13.1%", "-4.3%", "5.2x", "-4.0%", "MATERIAL"):
        assert reading in resignation, reading
    assert "UNEXPLAINED" not in resignation
    unexplained = next(line for line in text.splitlines() if "Heavy volume, no 8-K" in line)
    assert "2026-11-20" in unexplained and "-14.0%" in unexplained
    assert "MATERIAL, UNEXPLAINED" in unexplained
    assert "vs XLK" in text


def test_the_terminal_report_attributes_headlines_and_prints_them_as_text(monkeypatch):
    # A headline is outside text: brackets in it must not be read as markup.
    events = _events().model_copy(update={"events": [
        _events().events[1].model_copy(update={"headlines": [
            Headline(title="Acme falls [/red] as chief resigns", source="Reuters",
                     published="2026-11-12")]})]})
    text = _printed(monkeypatch, _verdict().model_copy(update={"recent_events": events}))
    assert "Acme falls [/red] as chief resigns (Reuters, 2026-11-12)" in text
    assert "press reports, not verified" in text


def test_the_terminal_report_dates_a_filing_no_session_has_traded_on(monkeypatch):
    text = _printed(monkeypatch, _verdict())
    pending = next(line for line in text.splitlines() if "8.01 Other events" in line)
    assert "2026-11-20" in pending and "no session yet" in pending


def test_the_terminal_report_says_so_when_events_could_not_be_gathered(monkeypatch):
    text = _printed(monkeypatch, _verdict().model_copy(update={"recent_events": None}))
    assert "Recent events could not be gathered for this run" in text

