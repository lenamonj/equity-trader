import pytest
import respx
import httpx
from equity_trader.data import edgar_tools as et


_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}


@respx.mock
def test_ticker_to_cik():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON)
    )
    cik = et._ticker_to_cik("AAPL")
    assert cik == "0000320193"


@respx.mock
def test_get_recent_8k_summaries():
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON)
    )
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={
            "filings": {"recent": {
                "form": ["8-K", "10-Q", "8-K"],
                "filingDate": ["2026-05-01", "2026-04-15", "2026-03-10"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "accessionNumber": ["1-1", "2-2", "3-3"],
                "items": ["2.02", "", "8.01"],
            }}
        })
    )
    out = et.get_recent_8k_summaries("AAPL", n=2)
    assert len(out) == 2
    assert out[0]["form"] == "8-K"
