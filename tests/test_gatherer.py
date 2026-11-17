import json

from equity_trader import gatherer


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
