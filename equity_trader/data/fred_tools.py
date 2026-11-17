import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import httpx
import pandas as pd
from fredapi import Fred

from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import iso, num
from equity_trader.exceptions import DataFetchError

log = logging.getLogger(__name__)


def _client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    return Fred(api_key=key)


@cached_for_run
def get_series(series_id: str, start: str | None = None) -> pd.Series:
    try:
        s = _client().get_series(series_id, observation_start=start)
    except Exception as exc:
        raise DataFetchError(f"FRED fetch failed for {series_id}: {exc}") from exc
    if s is None or len(s) == 0:
        raise DataFetchError(f"FRED series empty: {series_id}")
    return s



# ---------------------------------------------------------------------------
# Macro dashboard
# ---------------------------------------------------------------------------

_HISTORY_YEARS = 11  # enough for a ten-year percentile on a year-over-year series

# (series id, label, kind). A "level" series is read as published. A "yoy"
# series is an index or a total, read as its year-over-year percent change.
DASHBOARD: dict[str, list[tuple[str, str, str]]] = {
    "rates": [
        ("DGS3MO", "3-month Treasury yield", "level"),
        ("DGS2", "2-year Treasury yield", "level"),
        ("DGS10", "10-year Treasury yield", "level"),
        ("DGS30", "30-year Treasury yield", "level"),
        ("T10Y2Y", "10-year minus 2-year", "level"),
        ("T10Y3M", "10-year minus 3-month", "level"),
        ("DFF", "Effective federal funds rate", "level"),
        ("DFII10", "10-year real yield (TIPS)", "level"),
        ("T10YIE", "10-year breakeven inflation", "level"),
        ("T5YIFR", "5-year, 5-year forward inflation expectation", "level"),
        ("MORTGAGE30US", "30-year mortgage rate", "level"),
    ],
    "inflation": [
        ("CPIAUCSL", "CPI", "yoy"),
        ("CPILFESL", "Core CPI", "yoy"),
        ("PCEPILFE", "Core PCE price index", "yoy"),
        ("PPIFIS", "Producer prices, final demand", "yoy"),
        ("CES0500000003", "Average hourly earnings", "yoy"),
        ("MICH", "Michigan 1-year inflation expectation", "level"),
    ],
    "growth": [
        ("GDPC1", "Real GDP", "yoy"),
        ("GDPNOW", "Atlanta Fed GDPNow, current quarter annualized", "level"),
        ("INDPRO", "Industrial production", "yoy"),
        ("TCU", "Capacity utilization", "level"),
        ("RSAFS", "Retail sales", "yoy"),
        ("NEWORDER", "Core capital goods orders", "yoy"),
        ("HOUST", "Housing starts, thousands annualized", "level"),
        ("UMCSENT", "Michigan consumer sentiment", "level"),
    ],
    "labor": [
        ("UNRATE", "Unemployment rate", "level"),
        ("PAYEMS", "Nonfarm payrolls", "yoy"),
        ("ICSA", "Initial jobless claims", "level"),
        ("JTSJOL", "Job openings, thousands", "level"),
        ("SAHMREALTIME", "Sahm rule recession indicator", "level"),
    ],
    "credit": [
        ("BAMLH0A0HYM2", "High yield spread (OAS)", "level"),
        ("BAMLC0A0CM", "Investment grade spread (OAS)", "level"),
        ("BAMLC0A4CBBB", "BBB spread (OAS)", "level"),
        ("BAMLH0A3HYC", "CCC and lower spread (OAS)", "level"),
        ("DRTSCILM", "Banks tightening C&I loan standards, net percent", "level"),
        ("DRCCLACBS", "Credit card delinquency rate", "level"),
    ],
    "liquidity": [
        ("WALCL", "Federal Reserve balance sheet", "yoy"),
        ("RRPONTSYD", "Overnight reverse repo, $ billions", "level"),
        ("WTREGEN", "Treasury General Account, $ millions", "level"),
        ("M2SL", "M2 money supply", "yoy"),
        ("NFCI", "Chicago Fed financial conditions (above zero is tight)", "level"),
        ("STLFSI4", "St. Louis Fed financial stress", "level"),
    ],
    "markets": [
        ("VIXCLS", "VIX", "level"),
        ("DTWEXBGS", "Broad trade-weighted dollar", "level"),
        ("DCOILWTICO", "WTI crude oil", "level"),
        ("DHHNGSP", "Henry Hub natural gas", "level"),
        ("PCOPPUSDM", "Copper, $ per metric ton", "level"),
    ],
    "recession": [
        ("RECPROUSM156N", "Smoothed recession probability, percent", "level"),
    ],
}

# yfinance sector name -> series that drive that sector.
SECTOR_SERIES: dict[str, list[tuple[str, str, str]]] = {
    "Technology": [
        ("IPG3344S", "Semiconductor and electronic component production", "yoy"),
        ("CAPUTLG3344S", "Semiconductor capacity utilization", "level"),
        ("PCU33443344", "Semiconductor producer prices", "yoy"),
        ("A679RC1Q027SBEA", "Investment in information processing equipment and software",
         "yoy"),
        ("NEWORDER", "Core capital goods orders", "yoy"),
    ],
    "Financial Services": [
        ("T10Y2Y", "10-year minus 2-year", "level"),
        ("DRTSCILM", "Banks tightening C&I loan standards, net percent", "level"),
        ("DRCCLACBS", "Credit card delinquency rate", "level"),
        ("DRSFRMACBS", "Residential mortgage delinquency rate", "level"),
        ("BUSLOANS", "Commercial and industrial loans", "yoy"),
        ("TOTBKCR", "Bank credit", "yoy"),
    ],
    "Energy": [
        ("DCOILWTICO", "WTI crude oil", "level"),
        ("DCOILBRENTEU", "Brent crude oil", "level"),
        ("DHHNGSP", "Henry Hub natural gas", "level"),
        ("IPG211S", "Oil and gas extraction production", "yoy"),
        ("GASREGW", "Regular gasoline, $ per gallon", "level"),
    ],
    "Consumer Cyclical": [
        ("RSAFS", "Retail sales", "yoy"),
        ("UMCSENT", "Michigan consumer sentiment", "level"),
        ("TOTALSA", "Vehicle sales, millions annualized", "level"),
        ("TOTALSL", "Consumer credit", "yoy"),
        ("DSPIC96", "Real disposable personal income", "yoy"),
        ("IPG3361T3S", "Motor vehicle and parts production", "yoy"),
    ],
    "Consumer Defensive": [
        ("CPIUFDSL", "Food prices (CPI)", "yoy"),
        ("RSAFS", "Retail sales", "yoy"),
        ("DSPIC96", "Real disposable personal income", "yoy"),
        ("UMCSENT", "Michigan consumer sentiment", "level"),
    ],
    "Healthcare": [
        ("CPIMEDSL", "Medical care prices (CPI)", "yoy"),
        ("CES6562000001", "Health care and social assistance employment", "yoy"),
    ],
    "Industrials": [
        ("INDPRO", "Industrial production", "yoy"),
        ("TCU", "Capacity utilization", "level"),
        ("NEWORDER", "Core capital goods orders", "yoy"),
        ("DGORDER", "Durable goods orders", "yoy"),
        ("AMTMNO", "Manufacturers' new orders", "yoy"),
        ("TSIFRGHT", "Freight transportation services index", "yoy"),
        ("ISRATIO", "Business inventories to sales ratio", "level"),
    ],
    "Basic Materials": [
        ("PCOPPUSDM", "Copper, $ per metric ton", "level"),
        ("WPU10", "Metals producer prices", "yoy"),
        ("IPG325S", "Chemical production", "yoy"),
        ("INDPRO", "Industrial production", "yoy"),
    ],
    "Real Estate": [
        ("MORTGAGE30US", "30-year mortgage rate", "level"),
        ("DGS10", "10-year Treasury yield", "level"),
        ("HOUST", "Housing starts, thousands annualized", "level"),
        ("PERMIT", "Building permits, thousands annualized", "level"),
        ("MSPUS", "Median home sale price", "yoy"),
        ("TTLCONS", "Construction spending", "yoy"),
    ],
    "Utilities": [
        ("DGS10", "10-year Treasury yield", "level"),
        ("DHHNGSP", "Henry Hub natural gas", "level"),
        ("IPG2211S", "Electric power generation", "yoy"),
        ("CUSR0000SEHF01", "Electricity prices (CPI)", "yoy"),
    ],
    "Communication Services": [
        ("USINFO", "Information sector employment", "yoy"),
        ("A679RC1Q027SBEA", "Investment in information processing equipment and software",
         "yoy"),
        ("UMCSENT", "Michigan consumer sentiment", "level"),
    ],
}

# FRED release ids for the scheduled data that moves markets.
_RELEASES = {10: "CPI", 50: "Jobs report", 53: "GDP", 54: "Personal income and PCE inflation"}
_FRED_API = "https://api.stlouisfed.org/fred"


def _history_start() -> str:
    return (date.today() - timedelta(days=366 * _HISTORY_YEARS)).isoformat()


def _fetch_many(series_ids: list[str]) -> dict[str, pd.Series | None]:
    """Fetch several series at once. A series that fails is logged and None."""
    client, start = _client(), _history_start()

    def one(series_id: str) -> pd.Series | None:
        try:
            return client.get_series(series_id, observation_start=start)
        except Exception as exc:
            log.warning("FRED series %s unavailable: %s", series_id, exc)
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(zip(series_ids, pool.map(one, series_ids), strict=True))


def _reading(series_id: str, label: str, kind: str, series: pd.Series | None) -> dict:
    """One dashboard row: the value now, three months and a year ago, and where
    it sits in its own ten-year range."""
    row = {"series": series_id, "label": label + (" (YoY %)" if kind == "yoy" else "")}
    raw = series.dropna() if series is not None else pd.Series(dtype=float)
    view = raw
    if kind == "yoy" and len(raw):
        # Month-end values, interpolated so a month that was never published
        # (CPI for October 2025) does not distort the reading a year later.
        monthly = raw.resample("ME").last().interpolate()
        view = (monthly.pct_change(12) * 100.0).dropna()
    if view.empty:
        return {**row, "value": None, "as_of": None}
    last = view.index[-1]
    window = view[view.index >= last - pd.DateOffset(years=10)]
    return {
        **row,
        "value": num(view.iloc[-1], 3),
        "as_of": iso(raw.index[-1]),
        "three_months_ago": num(view.asof(last - pd.DateOffset(months=3)), 3),
        "year_ago": num(view.asof(last - pd.DateOffset(years=1)), 3),
        "pctile": round(float((window <= view.iloc[-1]).mean() * 100.0)),
        "pctile_years": round((last - window.index[0]).days / 365.25, 1),
    }


def _readings(specs: list[tuple[str, str, str]]) -> list[dict]:
    fetched = _fetch_many([series_id for series_id, _, _ in specs])
    return [_reading(series_id, label, kind, fetched[series_id])
            for series_id, label, kind in specs]


@cached_for_run
def get_release_calendar(horizon_days: int = 180) -> list[dict]:
    """Scheduled CPI, jobs, GDP and PCE releases inside the horizon.

    FRED carries each agency's published schedule, which usually runs to the
    end of the calendar year, so the list can stop short of the horizon.
    """
    today = date.today()
    cutoff = today + timedelta(days=horizon_days)
    out: list[dict] = []
    try:
        with httpx.Client(timeout=20.0) as client:
            for release_id, name in _RELEASES.items():
                r = client.get(f"{_FRED_API}/release/dates", params={
                    "release_id": release_id, "api_key": os.environ.get("FRED_API_KEY"),
                    "file_type": "json", "include_release_dates_with_no_data": "true",
                    "sort_order": "desc", "limit": 12})
                r.raise_for_status()
                out += [{"date": d["date"], "release": name}
                        for d in r.json()["release_dates"]
                        if today <= date.fromisoformat(d["date"]) <= cutoff]
    except httpx.HTTPError as exc:
        log.warning("FRED release calendar unavailable: %s", exc)
    return sorted(out, key=lambda r: r["date"])


# The dashboard is about fifty FRED requests and its readings do not change
# within a day, so a complete one is kept for the day and reused across runs.
_DASHBOARD_BY_DAY: dict[str, dict] = {}


def get_macro_dashboard() -> dict:
    """The macro backdrop: rates, inflation, growth, labor, credit, liquidity,
    markets and recession risk, each reading with its direction and history."""
    today = date.today().isoformat()
    if today not in _DASHBOARD_BY_DAY:
        dashboard = _build_dashboard(today)
        complete = all(row["value"] is not None
                       for rows in dashboard["groups"].values() for row in rows)
        if not complete:
            return dashboard  # a failed series is retried on the next call
        _DASHBOARD_BY_DAY.clear()
        _DASHBOARD_BY_DAY[today] = dashboard
    return _DASHBOARD_BY_DAY[today]


def _build_dashboard(today: str) -> dict:
    flat = [spec for specs in DASHBOARD.values() for spec in specs]
    rows = {row["series"]: row for row in _readings(flat)}
    return {
        "as_of": today,
        "how_to_read": ("value is the latest reading (YoY % where the label says so). "
                        "three_months_ago and year_ago show direction. pctile is where "
                        "the value sits in its own history over pctile_years years."),
        "groups": {group: [rows[series_id] for series_id, _, _ in specs]
                   for group, specs in DASHBOARD.items()},
        "upcoming_releases": get_release_calendar(),
    }


@cached_for_run
def get_sector_macro(sector: str | None) -> dict:
    """Macro series specific to a sector, read the same way as the dashboard."""
    specs = SECTOR_SERIES.get(sector or "", [])
    return {"sector": sector, "readings": _readings(specs) if specs else []}


@cached_for_run
def get_fred_series(series_id: str) -> dict:
    """Any FRED series by id: what it is, where it stands, and its last 12 readings."""
    series_id = series_id.strip().upper()
    series = get_series(series_id, start=_history_start()).dropna()
    try:
        meta = _client().get_series_info(series_id)
    except Exception as exc:
        raise DataFetchError(f"FRED series info failed for {series_id}: {exc}") from exc
    return {
        **_reading(series_id, meta["title"], "level", series),
        "units": meta["units_short"],
        "frequency": meta["frequency_short"],
        "recent": [{"date": iso(when), "value": num(value, 3)}
                   for when, value in series.tail(12).items()],
    }


@cached_for_run
def search_fred(query: str) -> list[dict]:
    """The most-used FRED series matching a keyword search."""
    try:
        found = _client().search(query, limit=8, order_by="popularity", sort_order="desc")
    except Exception as exc:
        raise DataFetchError(f"FRED search failed for {query!r}: {exc}") from exc
    if found is None:
        return []
    return [{"series": series_id, "title": row["title"], "frequency": row["frequency_short"],
             "units": row["units_short"], "last_observation": iso(row["observation_end"])}
            for series_id, row in found.iterrows()]
