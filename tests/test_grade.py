"""The grading report in the terminal."""
from rich.console import Console

import grade
from equity_trader import grading
from tests.test_grading import closes, run


def _printed(monkeypatch, runs) -> str:
    console = Console(record=True, width=200)
    monkeypatch.setattr(grade, "console", console)
    grade.render(grading.grade(runs, closes()))
    return console.export_text()


RUNS = [run("2026-02-10T21:30:00Z", "BUY", [("jpm_fundamental", "SELL", 8, None)]),
        run("2026-02-10T22:30:00Z", "BUY", [("jpm_fundamental", "BUY", 8, None)]),
        run("2026-08-03T21:30:00Z", "SELL", [("jpm_fundamental", "SELL", 5, None)])]


def test_the_report_gives_each_callers_record_at_each_horizon(monkeypatch):
    text = _printed(monkeypatch, RUNS)
    three, six = text.split("6 months")[0], text.split("6 months")[1]
    committee = next(line for line in three.splitlines() if "Committee" in line)
    # Two graded runs on one ticker-day, one pending; both BUY calls were right.
    for cell in ("2", "1", "100%", "+6.5%", "+3.3%"):
        assert cell in committee.split(), cell
    morgan = next(line for line in three.splitlines() if "J.P. Morgan" in line)
    assert "50%" in morgan.split() and "+0.0%" in morgan.split()
    assert "+13.4%" in next(line for line in six.splitlines() if "Committee" in line)


def test_the_report_lists_the_committees_calls_run_by_run(monkeypatch):
    text = _printed(monkeypatch, RUNS)
    calls = text.split("Committee calls")[1]
    first = next(line for line in calls.splitlines() if "2026-02-10" in line)
    assert "ACME" in first and "BUY" in first and "+6.5%" in first and "+13.4%" in first
    pending = next(line for line in calls.splitlines() if "2026-08-03" in line)
    assert "SELL" in pending and "pending" in pending


def test_the_report_warns_that_runs_on_one_day_are_one_observation(monkeypatch):
    assert "one observation" in _printed(monkeypatch, RUNS)


def test_a_retired_desk_is_labelled_in_the_report(monkeypatch):
    runs = [run("2026-02-10T21:30:00Z", "BUY", [("jane_street_etf", "SELL", 8, None)])]
    assert "Jane Street (retired)" in _printed(monkeypatch, runs)
