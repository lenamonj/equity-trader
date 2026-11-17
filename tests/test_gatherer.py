import inspect

from equity_trader import gatherer
from equity_trader.agents.base import TOOL_FUNCTIONS


def test_gatherer_knows_every_tool_the_desks_hold():
    names = gatherer.tool_names()
    assert len(names) == len(set(names)) == 35
    assert set(names) == set(TOOL_FUNCTIONS)
    assert names[0] == "tool_get_company_snapshot"  # shared context comes first


def test_only_the_two_fred_lookups_are_left_to_the_desks():
    left = [name for name in gatherer.tool_names()
            if list(inspect.signature(TOOL_FUNCTIONS[name]).parameters) not in ([], ["ticker"])]
    assert sorted(left) == ["tool_get_fred_series", "tool_search_fred"]


def test_gather_calls_each_tool_once_in_order_and_records_failures(monkeypatch):
    calls = []

    def tool_a(ticker: str):
        calls.append(("a", ticker))

    def tool_b():
        calls.append(("b",))

    def tool_c(series_id: str):
        calls.append(("c", series_id))

    def tool_d(ticker: str):
        calls.append(("d", ticker))
        raise ValueError("Yahoo refused")

    fakes = {"tool_a": tool_a, "tool_b": tool_b, "tool_c": tool_c, "tool_d": tool_d}
    monkeypatch.setattr(gatherer, "TOOL_FUNCTIONS", fakes)
    monkeypatch.setattr(gatherer, "tool_names", lambda: list(fakes))

    report = gatherer.gather("NVDA")

    assert calls == [("a", "NVDA"), ("b",), ("d", "NVDA")]  # the lookup is not gathered
    assert [(row["tool"], row["error"]) for row in report] == [
        ("tool_a", None), ("tool_b", None), ("tool_d", "ValueError: Yahoo refused")]
