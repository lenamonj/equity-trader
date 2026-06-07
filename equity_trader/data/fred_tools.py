import os
import pandas as pd
from fredapi import Fred

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import DataFetchError


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


def _last_or_none(series_id: str) -> float | None:
    try:
        s = get_series(series_id)
        v = s.dropna().iloc[-1]
        return float(v)
    except Exception:
        return None


@cached_for_run
def get_macro_snapshot() -> dict:
    cpi = get_series("CPIAUCSL")
    cpi_yoy = None
    try:
        cpi_yoy = float((cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0)
    except Exception:
        pass
    return {
        "dgs10": _last_or_none("DGS10"),
        "dgs2": _last_or_none("DGS2"),
        "t10y2y": _last_or_none("T10Y2Y"),
        "cpi_yoy": cpi_yoy,
        "unrate": _last_or_none("UNRATE"),
        "fedfunds": _last_or_none("FEDFUNDS"),
        "nfci": _last_or_none("NFCI"),
        "vix": _last_or_none("VIXCLS"),
    }


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
