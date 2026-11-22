"""The terminal report."""
import asyncio
import sys

import pytest
from rich.console import Console

import run
from equity_trader.exceptions import CommitteeFailedError, TickerNotFoundError
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


def test_the_terminal_report_says_how_many_desks_the_target_rests_on(monkeypatch):
    text = _printed(monkeypatch, _verdict())
    target = next(line for line in text.splitlines() if "6-month target" in line)
    assert "$210.00 (from 3 of 7 desks)" in target


# The event loops the faked committee ran in.
loops: set = set()


def _batch(monkeypatch, tickers, unknown=(), failed=()):
    """Run the command on several tickers with the committee faked."""
    seen = []
    loops.clear()

    async def committee(ticker, on_verdict=None, on_gathered=None):
        seen.append(ticker)
        loops.add(asyncio.get_running_loop())
        if ticker in unknown:
            raise TickerNotFoundError(ticker)
        if ticker in failed:
            raise CommitteeFailedError(ticker, "RuntimeError: boom")
        return _verdict().model_copy(update={"ticker": ticker})

    console = Console(record=True, width=200)
    monkeypatch.setattr(run, "console", console)
    monkeypatch.setattr(run, "analyze_ticker", committee)
    monkeypatch.setattr(sys, "argv", ["run.py", *tickers])
    return seen, console


def test_several_tickers_run_in_sequence_and_end_with_a_summary(monkeypatch):
    seen, console = _batch(monkeypatch, ["mu", "ORCL"])
    run.main()
    assert seen == ["MU", "ORCL"]
    summary = console.export_text().split("Committee calls")[1]
    for ticker in ("MU", "ORCL"):
        row = next(line for line in summary.splitlines() if ticker in line)
        for cell in ("BUY", "7/10", "+3.50", "$190.00", "$210.00"):
            assert cell in row, cell


def test_a_ticker_that_cannot_be_fetched_does_not_stop_the_batch(monkeypatch):
    seen, console = _batch(monkeypatch, ["MU", "ZZZZ", "ORCL"], unknown={"ZZZZ"})
    with pytest.raises(SystemExit) as stopped:
        run.main()
    assert stopped.value.code == 1
    assert seen == ["MU", "ZZZZ", "ORCL"]
    text = console.export_text()
    assert "Ticker not found: ZZZZ" in text
    summary = text.split("Committee calls")[1]
    assert "MU" in summary and "ORCL" in summary
    assert "Not run: ZZZZ" in summary


def test_one_ticker_prints_no_summary(monkeypatch):
    seen, console = _batch(monkeypatch, ["MU"])
    run.main()
    assert seen == ["MU"] and "Committee calls" not in console.export_text()


def test_a_batch_runs_in_one_event_loop(monkeypatch):
    # The model clients hold connections that belong to the loop they were
    # opened in. A loop per ticker left every second ticker with dead ones.
    seen, _ = _batch(monkeypatch, ["MU", "ORCL", "TGT"])
    run.main()
    assert seen == ["MU", "ORCL", "TGT"] and len(loops) == 1


def test_a_ticker_on_which_every_desk_failed_is_reported_and_skipped(monkeypatch):
    seen, console = _batch(monkeypatch, ["MU", "TGT", "ORCL"], failed={"TGT"})
    with pytest.raises(SystemExit) as stopped:
        run.main()
    assert stopped.value.code == 1 and seen == ["MU", "TGT", "ORCL"]
    text = console.export_text()
    assert "Every desk failed on TGT" in text and "Not run: TGT" in text
