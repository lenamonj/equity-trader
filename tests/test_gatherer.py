import json

import pytest

from equity_trader import gatherer
from equity_trader.schemas import Evidence


def company_snapshot(ticker: str) -> dict:
    """Today's date and
    the current price."""
    return {"ticker": ticker, "price": 153.38, "as_of": None}


def macro_dashboard() -> dict:
    """Rates and inflation."""
    return {"rates": [{"series": "DGS10", "value": 5.15}]}


def options_surface(ticker: str) -> dict:
    """Implied volatility by expiry."""
    raise ValueError("Yahoo refused")


SECTIONS = [company_snapshot, macro_dashboard, options_surface]


def test_gather_runs_each_section_once_in_order_and_records_failures():
    gathered = gatherer.gather("NVDA", SECTIONS)
    assert list(gathered) == ["company_snapshot", "macro_dashboard", "options_surface"]
    assert gathered["company_snapshot"]["data"] == {"ticker": "NVDA", "price": 153.38,
                                                    "as_of": None}
    assert gathered["macro_dashboard"]["error"] is None
    assert gathered["options_surface"]["data"] is None
    assert gathered["options_surface"]["error"] == "ValueError: Yahoo refused"


def test_brief_hands_a_desk_its_sections_and_nothing_else():
    gathered = gatherer.gather("NVDA", SECTIONS)
    text = gatherer.brief("NVDA", [company_snapshot, options_surface], gathered)
    lines = text.split("\n")
    assert lines[0] == "Analyze NVDA for a 3-6 month horizon."
    at = lines.index("## company_snapshot")
    assert lines[at + 1] == "Today's date and the current price."
    assert json.loads(lines[at + 2]) == {"ticker": "NVDA", "price": 153.38, "as_of": None}
    assert lines[-3:] == ["## options_surface", "Implied volatility by expiry.",
                          "UNAVAILABLE: ValueError: Yahoo refused"]
    assert "macro_dashboard" not in text


SECTION = {"atm_iv": 0.5385, "revenue": 96221000000, "return_3m": -0.2906, "trades": 58,
           "nothing": None, "flag": True,
           "text": "Revenue is expected to be $108.0 billion, plus or minus 2%.",
           "rows": [{"close": 153.82}, {"close": 1234.5}]}


@pytest.mark.parametrize("cited", [
    0.5385,     # as it appears
    53.85,      # a fraction given as a percentage
    96.221,     # billions
    96221,      # millions
    -29.06,     # a signed return as a percentage
    29.06,      # the same decline, stated without its sign
    58,         # a count
    108.0,      # a number in the text of a filing
    153.82,     # inside a nested row
    153.8,      # rounded to one decimal
])
def test_a_number_copied_from_the_section_is_found(cited):
    assert gatherer.in_data(cited, SECTION)


@pytest.mark.parametrize("cited", [61.2, 0.75, 987.65, 150.0])
def test_a_number_that_is_not_in_the_section_is_not_found(cited):
    assert not gatherer.in_data(cited, SECTION)


def test_nothing_is_found_in_a_section_that_failed_to_fetch():
    assert not gatherer.in_data(5.0, None)


def test_check_looks_in_the_section_each_reading_names():
    gathered = {"options_surface": {"data": SECTION, "error": None},
                "company_snapshot": {"data": {"price": 153.82}, "error": None}}

    def reading(value, source):
        return Evidence(metric="m", value=value, unit="other", as_of="2026-11-17",
                        source=source, signal="neutral")

    checked = gatherer.check([reading(53.85, "options_surface"),
                              reading(53.85, "company_snapshot")], gathered)
    assert [item.in_data for item in checked] == [True, False]
