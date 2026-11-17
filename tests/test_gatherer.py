import inspect

from equity_trader import gatherer
from equity_trader.agents.base import TOOL_FUNCTIONS


def test_gatherer_knows_every_tool_the_desks_hold():
    names = gatherer.tool_names()
    assert len(names) == len(set(names)) == 33
    assert set(names) == set(TOOL_FUNCTIONS)
    assert names[0] == "tool_get_company_snapshot"  # shared context comes first


def test_no_desk_tool_takes_an_argument_a_model_chooses():
    # A tool with a free argument could not be gathered ahead of time, and a
    # desk calling it would reach a data provider in the middle of a run.
    for name in gatherer.tool_names():
        assert list(inspect.signature(TOOL_FUNCTIONS[name]).parameters) in ([], ["ticker"]), name


def test_gather_calls_each_tool_once_in_order_and_records_failures(monkeypatch):
    calls = []

    def tool_a(ticker: str):
        calls.append(("a", ticker))

    def tool_b():
        calls.append(("b",))

    def tool_c(ticker: str):
        calls.append(("c", ticker))
        raise ValueError("Yahoo refused")

    fakes = {"tool_a": tool_a, "tool_b": tool_b, "tool_c": tool_c}
    monkeypatch.setattr(gatherer, "TOOL_FUNCTIONS", fakes)
    monkeypatch.setattr(gatherer, "tool_names", lambda: list(fakes))

    report = gatherer.gather("NVDA")

    assert calls == [("a", "NVDA"), ("b",), ("c", "NVDA")]
    assert [(row["tool"], row["error"]) for row in report] == [
        ("tool_a", None), ("tool_b", None), ("tool_c", "ValueError: Yahoo refused")]
