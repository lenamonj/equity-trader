import os
import time
from datetime import date

import httpx

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import DataFetchError, TickerNotFoundError

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


# Line item -> candidate US-GAAP tags. Companies differ in which tag they
# file, so the candidate reported most recently is the one used.
_FLOW_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "RevenuesNetOfInterestExpense"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "ExtraordinaryItemsNoncontrollingInterest"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
}
_INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "stockholders_equity": ["StockholdersEquity"],
}
# A year-to-date figure for these cannot be differenced into a quarter.
_AS_FILED_ONLY = {"eps_diluted", "diluted_shares"}
_PERIODIC_FORMS = ("10-K", "10-Q", "20-F", "40-F")
_STATEMENT_NOTES = (
    "Quarterly figures are single quarters. Where filings give only "
    "year-to-date figures (cash flow items, fourth quarters), the quarter is "
    "the difference between consecutive year-to-date figures. Per-share "
    "figures appear only as filed. `filed` is the date the period was first "
    "reported; values reflect any later restatement."
)


def _company_facts(cik: str) -> dict:
    _throttle()
    try:
        with httpx.Client(headers={"User-Agent": _user_agent()}, timeout=30.0) as c:
            r = c.get(f"{_DATA}/api/xbrl/companyfacts/CIK{cik}.json")
            r.raise_for_status()
            return r.json()["facts"]
    except httpx.HTTPError as exc:
        raise DataFetchError(f"EDGAR company facts unavailable for CIK {cik}: {exc}") from exc


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _is_quarter(start: str, end: str) -> bool:
    return 80 <= _days(start, end) <= 100


def _is_year(start: str, end: str) -> bool:
    return 350 <= _days(start, end) <= 380


def _series(gaap: dict, tags: list[str]) -> tuple[str | None, dict]:
    """Facts for whichever candidate tag was reported most recently.

    Returns the tag and {(start, end): (value, first_filed, form)}. `start` is
    None for balance sheet items. The latest filing's value wins.
    """
    best: tuple[str, str, list[dict]] | None = None
    for tag in tags:
        rows = [r for unit_rows in gaap.get(tag, {}).get("units", {}).values()
                for r in unit_rows if r["form"].startswith(_PERIODIC_FORMS)]
        if rows:
            latest = max(r["end"] for r in rows)
            if best is None or latest > best[0]:
                best = (latest, tag, rows)
    if best is None:
        return None, {}
    periods: dict = {}
    for r in sorted(best[2], key=lambda r: r["filed"]):
        key = (r.get("start"), r["end"])
        first = periods[key][1:] if key in periods else (r["filed"], r["form"])
        periods[key] = (r["val"], *first)
    return best[1], periods


def _annual(periods: dict, end: str) -> float | None:
    return next((v[0] for (s, e), v in periods.items()
                 if e == end and s and _is_year(s, e)), None)


def _quarterly(periods: dict, end: str, derive: bool) -> float | None:
    """The single quarter ending `end`, as filed or from year-to-date figures."""
    ending = {s: v[0] for (s, e), v in periods.items() if e == end and s}
    for start, value in ending.items():
        if _is_quarter(start, end):
            return value
    if not ending or not derive:
        return None
    start = min(ending)  # the year-to-date fact
    prior_ends = [e for (s, e) in periods if s == start and e < end]
    if not prior_ends or not _is_quarter(max(prior_ends), end):
        return None
    return ending[start] - periods[(start, max(prior_ends))][0]


@cached_for_run
def get_financial_statements(ticker: str, years: int = 4, quarters: int = 5) -> dict:
    """Core income statement, cash flow and balance sheet lines, as filed.

    Sourced from the SEC's XBRL company facts. Raises DataFetchError when the
    company files no US-GAAP annual statements (for example IFRS filers).
    """
    cik = _ticker_to_cik(ticker)
    gaap = _company_facts(cik).get("us-gaap", {})
    tags: dict[str, str] = {}
    flows: dict[str, dict] = {}
    instants: dict[str, dict] = {}
    for target, table in ((flows, _FLOW_TAGS), (instants, _INSTANT_TAGS)):
        for name, candidates in table.items():
            tag, periods = _series(gaap, candidates)
            if tag:
                tags[name] = tag
                target[name] = periods

    anchor = flows.get("net_income") or flows.get("revenue") or {}
    annual_ends = sorted({e for (s, e) in anchor if s and _is_year(s, e)}, reverse=True)
    if not annual_ends:
        raise DataFetchError(f"No US-GAAP annual statements on EDGAR for {ticker}")
    quarter_ends = sorted({e for (s, e) in anchor if s}, reverse=True)

    def row(end: str, annual: bool) -> dict:
        filed, form = min(v[1:] for (s, e), v in anchor.items() if e == end and s)
        out: dict = {"period_end": end, "form": form, "filed": filed}
        for name, periods in flows.items():
            out[name] = (_annual(periods, end) if annual
                         else _quarterly(periods, end, name not in _AS_FILED_ONLY))
        for name, periods in instants.items():
            out[name] = periods.get((None, end), (None,))[0]
        if out.get("operating_cash_flow") is not None and out.get("capex") is not None:
            out["free_cash_flow"] = out["operating_cash_flow"] - out["capex"]
        return out

    return {
        "source": "SEC EDGAR XBRL company facts",
        "xbrl_tags": tags,
        "annual": [row(e, True) for e in annual_ends[:years]],
        "quarterly": [row(e, False) for e in quarter_ends[:quarters]],
        "notes": _STATEMENT_NOTES,
    }
