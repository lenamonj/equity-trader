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


@cached_for_run
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
    """The most recent 8-Ks: filing date, the time the SEC accepted the filing
    (UTC), named items and the filing's text."""
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
            "accepted": recent["acceptanceDateTime"][i],
            "items": [f"{c} {_8K_ITEMS.get(c, 'Other')}" for c in codes],
            "text": _8k_text(cik, recent["accessionNumber"][i],
                             recent["primaryDocument"][i]),
        })
        if len(out) >= n:
            break
    return out


# Line item -> candidate US-GAAP tags. Companies differ in which tag they
# file, and some move from one to another, so the candidate with the most
# recent full-year figure is the one used.
_FLOW_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "RevenuesNetOfInterestExpense"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "ExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                      "MinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    # Income attributable to the company. These two differ only by preferred
    # dividends, and companies move between them: Caterpillar left
    # NetIncomeLoss in 2011, Occidental in 2024, and Freeport took it up in
    # 2026. `_net_income` reads each period from the first that reports it.
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    # Part of net income, shown so a one-off gain on a sale is not read as earnings.
    "discontinued_operations": ["IncomeLossFromDiscontinuedOperationsNetOfTax"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
                       "PaymentsOfOrdinaryDividends"],
    "research_and_development": ["ResearchAndDevelopmentExpense",
                                 "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "selling_general_and_admin": ["SellingGeneralAndAdministrativeExpense"],
    "stock_based_compensation": ["ShareBasedCompensation",
                                 "AllocatedShareBasedCompensationExpense"],
    "depreciation_and_amortization": ["DepreciationDepletionAndAmortization",
                                      "DepreciationAndAmortization",
                                      "DepreciationAmortizationAndAccretionNet"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating",
                         "InterestExpenseDebt", "InterestAndDebtExpense"],
}
_INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    # Oracle, Target and Home Depot carry their debt under the last three.
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent",
                       "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesAndLoans",
                       "LongTermNotesPayable"],
    "stockholders_equity": ["StockholdersEquity"],
}
# A year-to-date figure for these cannot be differenced into a quarter.
_AS_FILED_ONLY = {"eps_diluted", "diluted_shares"}
_PERIODIC_FORMS = ("10-K", "10-Q", "20-F", "40-F")
# The latest fiscal year is at most about fifteen months old on the day
# before the next annual report is filed. Statements older than this are not
# the company's current ones, whatever the reason.
_STALE_DAYS = 500
# Consolidated profit, used only when no attributable figure is current.
_PROFIT_TAG = "ProfitLoss"
# What net income means when it is not filed under NetIncomeLoss.
_NET_INCOME_NOTES = {
    _PROFIT_TAG: (" net_income here is consolidated profit, before the share of "
                  "noncontrolling interests is taken out."),
    "NetIncomeLossAvailableToCommonStockholdersBasic": (
        " Where NetIncomeLoss is not filed for a period, net_income is income available "
        "to common stockholders, after preferred dividends."),
}
# The facts feed can stop while the company keeps filing. A latest report this
# many days past the last period in the facts means a quarter is missing.
_REPORT_GAP_DAYS = 45
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


def _today() -> date:
    return date.today()


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
    """Facts for whichever candidate tag has the most recent full-year figure,
    then the most recent period of any length. Ties go to the earlier candidate.

    Returns the tag and {(start, end): (value, first_filed, form, fiscal_year,
    fiscal_period)}. `start` is None for balance sheet items. The latest
    filing's value wins; the filing details are those of the first report.
    """
    best: tuple[tuple[str, str], str, list[dict]] | None = None
    for tag in tags:
        rows = [r for unit_rows in gaap.get(tag, {}).get("units", {}).values()
                for r in unit_rows if r["form"].startswith(_PERIODIC_FORMS)]
        if rows:
            # A tag a company has just moved to has quarters and no year yet;
            # one it has left has years and no recent quarter.
            years = [r["end"] for r in rows if r.get("start") and _is_year(r["start"], r["end"])]
            latest = (max(years, default=""), max(r["end"] for r in rows))
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


def _latest_year(periods: dict) -> str:
    return max((e for (s, e) in periods if s and _is_year(s, e)), default="")


def _net_income(gaap: dict) -> tuple[str | None, dict]:
    """Net income attributable to the company, with the tags it came from.

    Each period is read from the first attributable tag that reports it, so
    the series follows a company from one tag to the other. Consolidated
    profit, which includes minority owners' share, is used only when it has a
    more recent full year than anything attributable.
    """
    periods: dict = {}
    used: list[str] = []
    for tag in _FLOW_TAGS["net_income"]:
        _, found = _series(gaap, [tag])
        new = {period: fact for period, fact in found.items() if period not in periods}
        if new:
            periods.update(new)
            used.append(tag)
    profit_tag, profit = _series(gaap, [_PROFIT_TAG])
    if profit_tag and _latest_year(profit) > _latest_year(periods):
        return profit_tag, profit
    return "+".join(used) or None, periods


def _latest_report(cik: str) -> str | None:
    """The period end of the most recent annual or quarterly report filed."""
    recent = _submissions(cik)["filings"]["recent"]
    return max((covering for form, covering in zip(recent["form"], recent["reportDate"],
                                                    strict=True)
                if form in _PERIODIC_FORMS and covering), default=None)


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
    company files no US-GAAP annual statements (for example IFRS filers), when
    the latest fiscal year in the facts is long past, or when the facts stop
    short of the company's latest report.
    """
    cik = _ticker_to_cik(ticker)
    gaap = _company_facts(cik).get("us-gaap", {})
    tags: dict[str, str] = {}
    flows: dict[str, dict] = {}
    instants: dict[str, dict] = {}
    for target, table in ((flows, _FLOW_TAGS), (instants, _INSTANT_TAGS)):
        for name, candidates in table.items():
            tag, periods = (_net_income(gaap) if name == "net_income"
                            else _series(gaap, candidates))
            if tag:
                tags[name] = tag
                target[name] = periods

    anchor = flows.get("net_income") or flows.get("revenue") or {}
    annual_ends = sorted({e for (s, e) in anchor if s and _is_year(s, e)}, reverse=True)
    if not annual_ends:
        raise DataFetchError(f"No US-GAAP annual statements on EDGAR for {ticker}")
    if (_today() - date.fromisoformat(annual_ends[0])).days > _STALE_DAYS:
        raise DataFetchError(
            f"US-GAAP statements on EDGAR for {ticker} end {annual_ends[0]}")
    quarter_ends = sorted({e for (s, e) in anchor if s}, reverse=True)
    reported = _latest_report(cik)
    if reported and _days(quarter_ends[0], reported) > _REPORT_GAP_DAYS:
        raise DataFetchError(f"EDGAR facts for {ticker} end {quarter_ends[0]}, but its "
                             f"latest report covers {reported}")
    # Gross profit is derived where it is not filed, or was last filed for an
    # earlier year than cost of revenue.
    if "cost_of_revenue" in flows and (_latest_year(flows.get("gross_profit", {}))
                                       < _latest_year(flows["cost_of_revenue"])):
        flows.pop("gross_profit", None)
        tags["gross_profit"] = "derived: revenue less cost_of_revenue"

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
        if "gross_profit" not in flows and "cost_of_revenue" in flows:
            revenue, cost = out.get("revenue"), out["cost_of_revenue"]
            out["gross_profit"] = (revenue - cost
                                   if revenue is not None and cost is not None else None)
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
        "notes": _STATEMENT_NOTES + "".join(
            note for tag, note in _NET_INCOME_NOTES.items()
            if tag in tags.get("net_income", "").split("+")),
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
