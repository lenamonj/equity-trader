import os
import time
import httpx

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import TickerNotFoundError, DataFetchError

_BASE = "https://www.sec.gov"
_DATA = "https://data.sec.gov"
_TICKERS_URL = f"{_BASE}/files/company_tickers.json"
_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.1  # ~10 req/sec ceiling per SEC


def _user_agent() -> str:
    email = os.environ.get("SEC_EDGAR_USER_AGENT_EMAIL", "anon@example.com")
    return f"EquityTrader {email}"


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": _user_agent()}, timeout=15.0)


def _throttle() -> None:
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL["t"] = time.monotonic()


@cached_for_run
def _ticker_to_cik(ticker: str) -> str:
    _throttle()
    with _client() as c:
        r = c.get(_TICKERS_URL)
        r.raise_for_status()
        data = r.json()
    upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == upper:
            return f"{int(entry['cik_str']):010d}"
    raise TickerNotFoundError(ticker)


def _submissions(cik: str) -> dict:
    _throttle()
    with _client() as c:
        r = c.get(f"{_DATA}/submissions/CIK{cik}.json")
        r.raise_for_status()
        return r.json()


@cached_for_run
def get_recent_8k_summaries(ticker: str, n: int = 5) -> list[dict]:
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    out: list[dict] = []
    for i, form in enumerate(recent["form"]):
        if form != "8-K":
            continue
        out.append({
            "form": form,
            "filing_date": recent["filingDate"][i],
            "accession_no": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            "items": recent.get("items", [""] * len(recent["form"]))[i],
        })
        if len(out) >= n:
            break
    return out


def _latest_form(ticker: str, form: str) -> dict:
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    for i, f in enumerate(recent["form"]):
        if f == form:
            return {
                "form": f,
                "filing_date": recent["filingDate"][i],
                "accession_no": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "cik": cik,
            }
    raise DataFetchError(f"No {form} found for {ticker}")


@cached_for_run
def get_latest_10k(ticker: str) -> dict:
    return _latest_form(ticker, "10-K")


@cached_for_run
def get_latest_10q(ticker: str) -> dict:
    return _latest_form(ticker, "10-Q")


@cached_for_run
def get_filing_text(accession_no: str, cik: str, primary_document: str) -> str:
    _throttle()
    acc_clean = accession_no.replace("-", "")
    url = f"{_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_document}"
    with _client() as c:
        r = c.get(url)
        r.raise_for_status()
        return r.text
