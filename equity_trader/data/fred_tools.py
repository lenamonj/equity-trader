import logging
import os
import pandas as pd
from fredapi import Fred

from equity_trader.data.cache import cached_for_run
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


def _latest(series_id: str) -> tuple[float | None, str | None]:
    """Latest observation and its date, or (None, None) if unavailable."""
    try:
        s = get_series(series_id).dropna()
        return float(s.iloc[-1]), s.index[-1].date().isoformat()
    except Exception as exc:
        log.warning("FRED series %s unavailable: %s", series_id, exc)
        return None, None


def _last_or_none(series_id: str) -> float | None:
    return _latest(series_id)[0]


def _cpi_yoy() -> tuple[float | None, str | None]:
    try:
        # Interpolate so a missing month (October 2025 was never published)
        # does not blank the year-over-year reading twelve months later.
        cpi = get_series("CPIAUCSL").interpolate()
        return (float((cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0),
                cpi.index[-1].date().isoformat())
    except Exception as exc:
        log.warning("CPI year-over-year unavailable: %s", exc)
        return None, None


@cached_for_run
def get_macro_snapshot() -> dict:
    readings = {
        "dgs10": _latest("DGS10"),
        "dgs2": _latest("DGS2"),
        "t10y2y": _latest("T10Y2Y"),
        "cpi_yoy": _cpi_yoy(),
        "unrate": _latest("UNRATE"),
        "fedfunds": _latest("FEDFUNDS"),
        "nfci": _latest("NFCI"),
        "vix": _latest("VIXCLS"),
    }
    snapshot: dict = {k: value for k, (value, _) in readings.items()}
    snapshot["as_of"] = {k: as_of for k, (_, as_of) in readings.items()}
    return snapshot


# Sector -> macro series mapping for the Bridgewater agent
_SECTOR_SERIES = {
    "Financials": ["DGS10", "T10Y2Y", "DRCCLACBS"],
    "Technology": ["UMCSENT", "INDPRO"],
    "Energy": ["DCOILWTICO", "DHHNGSP"],
    "Consumer Cyclical": ["UMCSENT", "RSXFS"],
    "Industrials": ["INDPRO", "ISRATIO"],
    "Healthcare": ["CPIAUCSL"],
    "Utilities": ["DGS10"],
    "Real Estate": ["MORTGAGE30US", "DGS10"],
}


@cached_for_run
def get_sector_macro(sector: str) -> dict:
    series_ids = _SECTOR_SERIES.get(sector, [])
    return {sid.lower(): _last_or_none(sid) for sid in series_ids}
