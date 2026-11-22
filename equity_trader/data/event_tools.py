"""Recent events and the price reaction to each, joined in code.

A filing says what happened and the price bars say what the market made of
it. No desk reads both, so the join is done here and handed to every desk and
to the chair.
"""
import logging
import re
import time
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
import pandas as pd

from equity_trader.data import edgar_tools, technical_tools
from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import iso, num

log = logging.getLogger(__name__)

_MARKET_TIME = "America/New_York"
_CLOSE_HOUR = 16
_MATERIAL_MOVE = 0.10  # a session move this large must be addressed by the chair
_EXCERPT = 500         # characters of a filing's text
_NEWS_URL = "https://news.google.com/rss/search"
_NEWS_PAUSE = 0.5      # seconds between requests to the feed
_HEADLINES = 6
_TITLE_LIMIT = 200     # characters
_NOTES = (
    "One entry per session, largest move first. A filing accepted after the "
    "4pm close is read on the next session. opening_gap is the open against "
    "the prior close, volume_vs_50d_avg is that session's volume over its "
    "50-day average, and return_vs_sector_etf is the close-to-close return "
    "less the sector ETF's. move is the larger of the opening gap and the "
    "close-to-close return, in absolute terms; material marks a move of 10% "
    "or more. unexplained marks a heavy-volume session with no 8-K. "
    "Filings are the company's own statements. Headlines are press claims, "
    "not facts: attribute each to its source. Null headlines mean the feed "
    "could not be read; a null reaction means no session has traded since."
)

_LEGAL_SUFFIX = re.compile(
    r"(\.com)?[\s,&]*\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|"
    r"holdings|group|llc|n\.v|s\.a|ag|se)\.?$", re.IGNORECASE)


def company_name(legal_name: str) -> str:
    """The name the press uses: 'NVIDIA Corporation' is 'NVIDIA'."""
    name = re.sub(r"^the\s+", "", legal_name.strip(), flags=re.IGNORECASE)
    while (shorter := _LEGAL_SUFFIX.sub("", name)) != name:
        name = shorter
    return name


def _parse_headlines(feed: str, company: str) -> list[dict]:
    """The first few distinct titles in a Google News feed that name the
    company, leaving out the company's own newsroom."""
    names_company = re.compile(rf"\b{re.escape(company)}\b", re.IGNORECASE)
    headlines: list[dict] = []
    for item in ElementTree.fromstring(feed).iter("item"):
        source = item.findtext("source") or ""
        # The feed writes a title as 'Headline - Source'.
        title = (item.findtext("title") or "").removesuffix(f" - {source}")
        if (not names_company.search(title) or names_company.search(source)
                or any(h["title"] == title for h in headlines)):
            continue
        headlines.append({"title": title[:_TITLE_LIMIT], "source": source,
                          "published": iso(parsedate_to_datetime(item.findtext("pubDate")))})
        if len(headlines) == _HEADLINES:
            break
    return headlines


def _headlines(company: str, start: date, end: date) -> list[dict] | None:
    """Press headlines naming the company between two dates, or None when the
    feed could not be read."""
    time.sleep(_NEWS_PAUSE)
    try:
        response = httpx.get(
            _NEWS_URL, timeout=15.0, follow_redirects=True,
            params={"q": f"{company} after:{start} before:{end}",
                    "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; EquityTrader)"})
        response.raise_for_status()
        return _parse_headlines(response.text, company)
    except (httpx.HTTPError, ElementTree.ParseError) as exc:
        log.warning("Headlines unavailable for %s, %s to %s: %s", company, start, end, exc)
        return None


def _reaction_session(accepted: str, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first session that could trade on a filing, or None if none has yet."""
    when = pd.Timestamp(accepted).tz_convert(_MARKET_TIME).tz_localize(None)
    day = when.normalize() + pd.Timedelta(days=int(when.hour >= _CLOSE_HOUR))
    later = sessions[sessions >= day]
    return later[0] if len(later) else None


@cached_for_run
def get_recent_events(ticker: str) -> dict:
    """The five most recent 8-Ks and the heaviest-volume sessions of the last
    60, each with that session's price reaction and the press headlines."""
    bars = technical_tools.history(ticker)
    bars = bars.set_axis(bars.index.tz_localize(None).normalize())
    info = yft.get_info(ticker)
    sector_etf = yft.SECTOR_ETF.get(info.get("sector"))

    prior_close = bars["Close"].shift()
    reaction = pd.DataFrame({
        "opening_gap": bars["Open"] / prior_close - 1.0,
        "close_to_close_return": bars["Close"] / prior_close - 1.0,
        "volume_vs_50d_avg": bars["Volume"] / bars["Volume"].rolling(50).mean(),
        "sector_etf_return": float("nan"),
    })
    if sector_etf:
        etf = yft.get_price_history(sector_etf, period="1y", interval="1d")["Close"]
        etf = etf.set_axis(etf.index.tz_localize(None).normalize())
        reaction["sector_etf_return"] = etf.pct_change().reindex(bars.index)

    # One event per session: its filings, and whether it was a heavy-volume day.
    by_session: dict = {}

    def event(key, session) -> dict:
        return by_session.setdefault(key, {"session": session, "filings": [],
                                           "heavy_volume": False})

    for filing in edgar_tools.get_recent_8ks(ticker):
        session = _reaction_session(filing["accepted"], bars.index)
        event(session or filing["filing_date"], session)["filings"].append({
            "filing_date": filing["filing_date"], "items": filing["items"],
            "excerpt": filing["text"][:_EXCERPT] if filing["text"] else None})
    for heavy in technical_tools.volume_analysis(ticker)["heaviest_days_60"]:
        session = pd.Timestamp(heavy["date"])
        event(session, session)["heavy_volume"] = True

    company = company_name(info.get("longName") or info.get("shortName") or ticker)
    events = []
    for entry in by_session.values():
        session, filings = entry["session"], entry["filings"]
        row = reaction.loc[session] if session is not None else {}
        gap, change = num(row.get("opening_gap")), num(row.get("close_to_close_return"))
        etf_change = num(row.get("sector_etf_return"))
        excess = num(change - etf_change) if etf_change is not None else None
        move = max(abs(gap), abs(change)) if gap is not None else None
        days = ([date.fromisoformat(f["filing_date"]) for f in filings]
                + ([session.date()] if session is not None else []))
        events.append({
            "session": iso(session),
            "filings": filings,
            "heavy_volume": entry["heavy_volume"],
            "unexplained": entry["heavy_volume"] and not filings,
            "opening_gap": gap,
            "close_to_close_return": change,
            "volume_vs_50d_avg": num(row.get("volume_vs_50d_avg"), 2),
            "sector_etf_return": etf_change,
            "return_vs_sector_etf": excess,
            "move": move,
            "material": move is not None and move >= _MATERIAL_MOVE,
            "headlines": _headlines(company, min(days) - timedelta(days=1),
                                    max(days) + timedelta(days=1)),
        })
    # Largest move first; a filing with no session yet goes last.
    events.sort(key=lambda e: -1.0 if e["move"] is None else e["move"], reverse=True)
    return {
        "as_of": iso(bars.index[-1]),
        "sector_etf": sector_etf,
        "events": [{"rank": rank, **e} for rank, e in enumerate(events, start=1)],
        "notes": _NOTES,
    }
