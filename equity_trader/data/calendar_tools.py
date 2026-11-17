"""Real upcoming catalysts for a ticker, from authoritative sources.

The orchestrator's LLM has no temporal grounding, so it will hallucinate dates
if asked to fill in `catalyst_calendar` from training-data knowledge. This
module gives the orchestrator real, sourced upcoming events to anchor its
output against.
"""
import logging
from collections.abc import Iterable
from datetime import date, timedelta

import yfinance as yf

from equity_trader.data import fred_tools
from equity_trader.data.cache import cached_for_run

log = logging.getLogger(__name__)


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
def _calendar(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).calendar or {}
    except Exception as exc:
        log.warning("Calendar unavailable for %s: %s", ticker, exc)
        return {}


def next_earnings_date(ticker: str, today: date) -> str | None:
    upcoming = [d for d in _coerce_dates(_calendar(ticker).get("Earnings Date")) if d >= today]
    return min(upcoming).isoformat() if upcoming else None


@cached_for_run
def get_real_catalysts(ticker: str, today: date,
                        horizon_days: int = 180) -> list[dict]:
    """Return events whose date is known and falls within the horizon.

    Each entry: {event, date (ISO), expected_impact, notes}. Earnings and
    dividend dates come from yfinance's calendar; scheduled CPI, jobs, GDP and
    PCE releases come from FRED's release calendar.
    """
    out: list[dict] = []
    cal = _calendar(ticker)

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

    out += [{"event": f"{release['release']} release", "date": release["date"],
             "expected_impact": "MEDIUM", "notes": "Source: FRED release calendar"}
            for release in fred_tools.get_release_calendar(horizon_days)]
    return sorted(out, key=lambda c: c["date"])
