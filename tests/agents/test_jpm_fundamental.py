from equity_trader.agents import jpm_fundamental as mod
from equity_trader.exceptions import DataFetchError


def test_fundamentals_come_from_the_sec_with_market_valuation(monkeypatch):
    monkeypatch.setattr(mod.edg, "get_financial_statements",
                        lambda t: {"source": "SEC EDGAR XBRL company facts", "annual": []})
    monkeypatch.setattr(mod.yft, "get_valuation", lambda t: {"trailing_pe": 30.0})
    out = mod.get_fundamentals("NVDA")
    assert out["source"] == "SEC EDGAR XBRL company facts"
    assert out["valuation"] == {"trailing_pe": 30.0}


def test_fundamentals_fall_back_to_yfinance_and_say_so(monkeypatch):
    def no_filings(ticker):
        raise DataFetchError("No US-GAAP annual statements on EDGAR for TSM")

    monkeypatch.setattr(mod.edg, "get_financial_statements", no_filings)
    monkeypatch.setattr(mod.yft, "get_fundamentals", lambda t: {"income_stmt": {}})
    out = mod.get_fundamentals("TSM")
    # The desk is told why it did not get the SEC's statements.
    assert out["source"] == ("yfinance, because the SEC statements could not be used: "
                             "No US-GAAP annual statements on EDGAR for TSM")
    assert "income_stmt" in out
