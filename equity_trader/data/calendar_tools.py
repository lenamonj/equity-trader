"""Real upcoming catalysts for a ticker, from authoritative sources.

The orchestrator's LLM has no temporal grounding, so it will hallucinate dates
if asked to fill in `catalyst_calendar` from training-data knowledge. This
module gives the orchestrator real, sourced upcoming events to anchor its
output against.
"""
from datetime import date, timedelta
from typing import Iterable

import yfinance as yf

from equity_trader.data.cache import cached_for_run


def _within(d: date, today: date, horizon_days: int) -> bool:
    return today <= d <= today + timedelta(days=horizon_days)


def _coerce_dates(value) -> Iterable[date]:
    """yfinance's calendar exposes single dates or lists; normalize both."""
    if value is None:
        return ()
    if isinstance(value, date):
        return (value,)
    if isinstance(value, list):
        return tuple(v for v in value if isinstance(v, date))
    return ()


@cached_for_run
def get_real_catalysts(ticker: str, today: date,
                        horizon_days: int = 180) -> list[dict]:
    """Return events whose date is known and falls within the horizon.

    Each entry: {event, date (ISO), expected_impact, notes}.
    Currently sources next earnings, next dividend, and ex-dividend dates from
    yfinance's calendar endpoint. Extend with FOMC/CPI dates from FRED if
    macro events become relevant downstream.
    """
    out: list[dict] = []
    try:
        cal = yf.Ticker(ticker).calendar or {}
    except Exception:
        return out

    for ed in _coerce_dates(cal.get("Earnings Date")):
        if _within(ed, today, horizon_days):
            out.append({
                "event": "Earnings release",
                "date": ed.isoformat(),
                "expected_impact": "HIGH",
                "notes": "Source: yfinance calendar",
            })

    for dd in _coerce_dates(cal.get("Dividend Date")):
        if _within(dd, today, horizon_days):
            out.append({
                "event": "Dividend payment",
                "date": dd.isoformat(),
                "expected_impact": "LOW",
                "notes": "Source: yfinance calendar",
            })

    for xd in _coerce_dates(cal.get("Ex-Dividend Date")):
        if _within(xd, today, horizon_days):
            out.append({
                "event": "Ex-dividend date",
                "date": xd.isoformat(),
                "expected_impact": "LOW",
                "notes": "Source: yfinance calendar",
            })

    return sorted(out, key=lambda c: c["date"])
