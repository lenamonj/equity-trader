import html
import logging
import os
import re
import time
from datetime import date

import httpx

from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import num
from equity_trader.exceptions import DataFetchError, TickerNotFoundError

log = logging.getLogger(__name__)

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


# Form 8-K item codes, so an event's meaning never rests on a model's memory.
_8K_ITEMS = {
    "1.01": "Entry into a material definitive agreement",
    "1.02": "Termination of a material definitive agreement",
    "1.03": "Bankruptcy or receivership",
    "1.05": "Material cybersecurity incident",
    "2.01": "Completion of acquisition or disposition of assets",
    "2.02": "Results of operations and financial condition",
    "2.03": "Creation of a direct financial obligation",
    "2.04": "Triggering event that accelerates a financial obligation",
    "2.05": "Costs associated with exit or disposal activities",
    "2.06": "Material impairment",
    "3.01": "Notice of delisting or failure to satisfy a listing rule",
    "3.02": "Unregistered sale of equity securities",
    "3.03": "Material modification to rights of security holders",
    "4.01": "Change in certifying accountant",
    "4.02": "Non-reliance on previously issued financial statements",
    "5.01": "Change in control",
    "5.02": "Departure or appointment of directors or officers",
    "5.03": "Amendment to articles or bylaws; change in fiscal year",
    "5.07": "Submission of matters to a vote of security holders",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
    "9.01": "Financial statements and exhibits",
}
_8K_TEXT_LIMIT = 6000  # characters; 8-K bodies are short, exhibits are not included


def _text(document: str) -> str:
    """An HTML filing as plain text."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", document))).strip()


def _8k_body(document: str) -> str:
    """The item text of an 8-K: after the cover page, before the signature."""
    text = _text(document)
    start = re.search(r"Item\s+\d\.\d\d", text)
    if start:
        text = text[start.start():]
    return text.split("SIGNATURE")[0].strip()[:_8K_TEXT_LIMIT]


def _8k_text(cik: str, accession_no: str, primary_document: str) -> str | None:
    _throttle()
    url = (f"{_BASE}/Archives/edgar/data/{int(cik)}/"
           f"{accession_no.replace('-', '')}/{primary_document}")
    try:
        with _client() as c:
            r = c.get(url)
            r.raise_for_status()
            return _8k_body(r.text)
    except httpx.HTTPError as exc:
        log.warning("8-K text unavailable (%s): %s", url, exc)
        return None


@cached_for_run
def get_recent_8ks(ticker: str, n: int = 5) -> list[dict]:
    """The most recent 8-Ks: filing date, named items and the filing's text."""
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    items = recent.get("items", [""] * len(recent["form"]))
    out: list[dict] = []
    for i, form in enumerate(recent["form"]):
        if form != "8-K":
            continue
        codes = [c.strip() for c in items[i].split(",") if c.strip()]
        out.append({
            "filing_date": recent["filingDate"][i],
            "items": [f"{c} {_8K_ITEMS.get(c, 'Other')}" for c in codes],
            "text": _8k_text(cik, recent["accessionNumber"][i],
                             recent["primaryDocument"][i]),
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
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "selling_general_and_admin": ["SellingGeneralAndAdministrativeExpense"],
    "stock_based_compensation": ["ShareBasedCompensation"],
    "depreciation_and_amortization": ["DepreciationDepletionAndAmortization",
                                      "DepreciationAndAmortization",
                                      "DepreciationAmortizationAndAccretionNet"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating",
                         "InterestExpenseDebt"],
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
    "reported; values reflect any later restatement. quarterly[0] is the latest "
    "reported quarter and quarters_ago counts back from it; use `period` and "
    "`period_end` when citing a figure. A null means the company does not report "
    "that line under a tag this tool reads."
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

    Returns the tag and {(start, end): (value, first_filed, form, fiscal_year,
    fiscal_period)}. `start` is None for balance sheet items. The latest
    filing's value wins; the filing details are those of the first report.
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
        first = (periods[key][1:] if key in periods
                 else (r["filed"], r["form"], r.get("fy"), r.get("fp")))
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


def _label(fiscal_year, fiscal_period, annual: bool) -> str | None:
    if not fiscal_year:
        return None
    if annual:
        return f"FY{fiscal_year}"
    quarter = fiscal_period if fiscal_period in ("Q1", "Q2", "Q3") else "Q4"
    return f"{quarter} FY{fiscal_year}"


def _div(a, b) -> float | None:
    return num(a / b) if a is not None and b else None


@cached_for_run
def get_financial_statements(ticker: str, years: int = 5, quarters: int = 6) -> dict:
    """Income statement, cash flow and balance sheet lines as filed, with ratios.

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

    def value(name: str, end: str, annual: bool) -> float | None:
        periods = flows.get(name)
        if periods is None:
            return None
        return (_annual(periods, end) if annual
                else _quarterly(periods, end, name not in _AS_FILED_ONLY))

    def growth(name: str, ends: list[str], i: int, lag: int, annual: bool) -> float | None:
        if i + lag >= len(ends):
            return None
        now, then = value(name, ends[i], annual), value(name, ends[i + lag], annual)
        return num(now / then - 1.0) if now is not None and then and then > 0 else None

    def row(ends: list[str], i: int, annual: bool) -> dict:
        end = ends[i]
        filed, form, fiscal_year, fiscal_period = min(
            (v[1:] for (s, e), v in anchor.items() if e == end and s), key=lambda v: v[0])
        out: dict = {"period": _label(fiscal_year, fiscal_period, annual),
                     "years_ago" if annual else "quarters_ago": i,
                     "period_end": end, "form": form, "filed": filed}
        for name in flows:
            out[name] = value(name, end, annual)
        for name, periods in instants.items():
            out[name] = periods.get((None, end), (None,))[0]
        if out.get("operating_cash_flow") is not None and out.get("capex") is not None:
            out["free_cash_flow"] = out["operating_cash_flow"] - out["capex"]

        revenue, cash_flow = out.get("revenue"), out.get("free_cash_flow")
        lag = 1 if annual else 4  # the same period a year earlier
        ratios = {
            "gross_margin": _div(out.get("gross_profit"), revenue),
            "operating_margin": _div(out.get("operating_income"), revenue),
            "net_margin": _div(out.get("net_income"), revenue),
            "free_cash_flow_margin": _div(cash_flow, revenue),
            "free_cash_flow_to_net_income": _div(cash_flow, out.get("net_income")),
            "revenue_growth_yoy": growth("revenue", ends, i, lag, annual),
            "net_income_growth_yoy": growth("net_income", ends, i, lag, annual),
            "diluted_shares_change_yoy": growth("diluted_shares", ends, i, lag, annual),
            "interest_coverage": _div(out.get("operating_income"), out.get("interest_expense")),
            "current_ratio": _div(out.get("current_assets"), out.get("current_liabilities")),
            "debt_to_equity": _div(out.get("long_term_debt"), out.get("stockholders_equity")),
        }
        if annual:
            ratios["return_on_equity"] = _div(out.get("net_income"),
                                              out.get("stockholders_equity"))
        else:
            ratios["revenue_growth_qoq"] = growth("revenue", ends, i, 1, annual)
        out["ratios"] = ratios
        return out

    def cagr(n: int) -> float | None:
        if len(annual_ends) <= n:
            return None
        now, then = value("revenue", annual_ends[0], True), value("revenue", annual_ends[n], True)
        return num((now / then) ** (1 / n) - 1.0) if now and then and then > 0 else None

    def trailing(name: str) -> float | None:
        values = [value(name, end, False) for end in quarter_ends[:4]]
        return sum(values) if len(values) == 4 and None not in values else None

    cash_flow, capex = trailing("operating_cash_flow"), trailing("capex")
    return {
        "source": "SEC EDGAR XBRL company facts",
        "xbrl_tags": tags,
        "summary": {
            "revenue_cagr_3y": cagr(3),
            "revenue_cagr_5y": cagr(5),
            "trailing_12_months": {
                "through": quarter_ends[0],
                "revenue": trailing("revenue"),
                "net_income": trailing("net_income"),
                "free_cash_flow": (cash_flow - capex
                                   if cash_flow is not None and capex is not None else None)},
        },
        "annual": [row(annual_ends, i, True) for i in range(min(years, len(annual_ends)))],
        "quarterly": [row(quarter_ends, i, False)
                      for i in range(min(quarters, len(quarter_ends)))],
        "notes": _STATEMENT_NOTES,
    }


_OUTLOOK = re.compile(r"\b(Business Outlook|Financial Outlook|Outlook|Guidance)\b")


@cached_for_run
def get_earnings_release(ticker: str) -> dict:
    """The company's latest earnings release and commentary, as furnished to the SEC.

    Each exhibit gives its opening and, where the document has one, the outlook
    section, which is where guidance is stated.
    """
    cik = _ticker_to_cik(ticker)
    recent = _submissions(cik)["filings"]["recent"]
    items = recent.get("items", [""] * len(recent["form"]))
    found = next((i for i, form in enumerate(recent["form"])
                  if form == "8-K" and "2.02" in items[i]), None)
    if found is None:
        raise DataFetchError(f"No earnings release on EDGAR for {ticker}")
    folder = (f"{_BASE}/Archives/edgar/data/{int(cik)}/"
              f"{recent['accessionNumber'][found].replace('-', '')}")
    exhibits = []
    try:
        with _client() as c:
            _throttle()
            listing = c.get(f"{folder}/index.json")
            listing.raise_for_status()
            names = [f["name"] for f in listing.json()["directory"]["item"]
                     if f["name"].endswith(".htm")
                     and f["name"] != recent["primaryDocument"][found]
                     and not re.fullmatch(r"R\d+\.htm", f["name"])
                     and "index" not in f["name"]]
            for name in names[:2]:
                _throttle()
                page = c.get(f"{folder}/{name}")
                page.raise_for_status()
                text = _text(page.text)
                outlook = _OUTLOOK.search(text, 200)
                exhibits.append({
                    "document": name,
                    "opening": text[:3000],
                    "outlook": text[outlook.start():outlook.start() + 2500] if outlook else None})
    except httpx.HTTPError as exc:
        raise DataFetchError(f"Earnings release unavailable for {ticker}: {exc}") from exc
    return {"source": "SEC EDGAR 8-K exhibits", "filing_date": recent["filingDate"][found],
            "exhibits": exhibits}
