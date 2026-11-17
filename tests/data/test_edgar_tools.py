import httpx
import pytest
import respx

from equity_trader.data import edgar_tools as et
from equity_trader.data.cache import reset_run_cache
from equity_trader.exceptions import DataFetchError

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


def _fact(val, end, form, filed, start=None):
    row = {"val": val, "end": end, "form": form, "filed": filed}
    if start:
        row["start"] = start
    return row


# Fiscal 2025 = calendar 2025. Net income is filed per quarter and for the
# year; operating cash flow only year to date, as in real 10-Qs.
_FACTS = {"facts": {"us-gaap": {
    "NetIncomeLoss": {"units": {"USD": [
        _fact(900, "2024-12-31", "10-K", "2025-02-01", "2024-01-01"),
        _fact(100, "2025-03-31", "10-Q", "2025-05-01", "2025-01-01"),
        _fact(110, "2025-06-30", "10-Q", "2025-08-01", "2025-04-01"),
        _fact(210, "2025-06-30", "10-Q", "2025-08-01", "2025-01-01"),
        _fact(120, "2025-09-30", "10-Q", "2025-11-01", "2025-07-01"),
        _fact(330, "2025-09-30", "10-Q", "2025-11-01", "2025-01-01"),
        _fact(480, "2025-12-31", "10-K", "2026-02-01", "2025-01-01"),
        # Restated a year later, and repeated in a proxy statement.
        _fact(105, "2025-03-31", "10-Q", "2026-05-01", "2025-01-01"),
        _fact(999, "2025-12-31", "DEF 14A", "2026-04-01", "2025-01-01"),
    ]}},
    "EarningsPerShareDiluted": {"units": {"USD/shares": [
        _fact(1.2, "2025-09-30", "10-Q", "2025-11-01", "2025-07-01"),
        _fact(4.8, "2025-12-31", "10-K", "2026-02-01", "2025-01-01"),
    ]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
        _fact(400, "2025-09-30", "10-Q", "2025-11-01", "2025-01-01"),
        _fact(600, "2025-12-31", "10-K", "2026-02-01", "2025-01-01"),
    ]}},
    "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
        _fact(40, "2025-09-30", "10-Q", "2025-11-01", "2025-01-01"),
        _fact(70, "2025-12-31", "10-K", "2026-02-01", "2025-01-01"),
    ]}},
    # An older revenue tag the company stopped filing, and its current one.
    "SalesRevenueNet": {"units": {"USD": [
        _fact(5000, "2019-12-31", "10-K", "2020-02-01", "2019-01-01"),
    ]}},
    "Revenues": {"units": {"USD": [
        _fact(2000, "2025-12-31", "10-K", "2026-02-01", "2025-01-01"),
    ]}},
    "StockholdersEquity": {"units": {"USD": [
        _fact(3000, "2025-12-31", "10-K", "2026-02-01"),
    ]}},
}}}


@respx.mock
def test_financial_statements_as_filed():
    reset_run_cache()
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON))
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json=_FACTS))
    out = et.get_financial_statements("AAPL")

    assert out["xbrl_tags"]["revenue"] == "Revenues"
    year = out["annual"][0]
    assert (year["period_end"], year["form"], year["filed"]) == (
        "2025-12-31", "10-K", "2026-02-01")
    assert year["net_income"] == 480  # the proxy statement's figure is ignored
    assert year["revenue"] == 2000
    assert year["free_cash_flow"] == 530
    assert year["stockholders_equity"] == 3000

    q4, q3, q2, q1 = out["quarterly"][:4]
    assert q4["net_income"] == 150            # year less nine months
    assert q4["operating_cash_flow"] == 200   # year-to-date difference
    assert q4["eps_diluted"] is None          # never derived
    assert q3["net_income"] == 120 and q3["eps_diluted"] == 1.2
    assert q2["net_income"] == 110
    assert q1["net_income"] == 105            # restated value
    assert q1["filed"] == "2025-05-01"        # first reported


@respx.mock
def test_financial_statements_raise_without_us_gaap():
    reset_run_cache()
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=_TICKERS_JSON))
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={"facts": {"ifrs-full": {}}}))
    with pytest.raises(DataFetchError):
        et.get_financial_statements("AAPL")


@pytest.mark.slow
def test_financial_statements_live():
    reset_run_cache()
    out = et.get_financial_statements("AAPL")
    assert len(out["annual"]) == 4 and len(out["quarterly"]) == 5
    assert out["annual"][0]["revenue"] > 0
