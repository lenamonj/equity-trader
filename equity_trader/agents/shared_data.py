"""Context every desk is given, and sections that more than one desk reads."""
from equity_trader.data import (
    company_tools,
    event_tools,
    fred_tools,
    market_tools,
    technical_tools,
)
from equity_trader.data import yfinance_tools as yft


def company_snapshot(ticker: str) -> dict:
    """Today's date, the current price, sector and industry, 52-week range, beta,
    next earnings date, short interest and ownership."""
    return company_tools.get_company_snapshot(ticker)


def macro_dashboard() -> dict:
    """About fifty FRED readings on rates, inflation, growth, labor, credit,
    liquidity, markets and recession risk. Each has its value now, three months
    ago and a year ago, and its ten-year percentile. Also lists scheduled CPI,
    jobs, GDP and PCE releases."""
    return fred_tools.get_macro_dashboard()


def sector_macro(ticker: str) -> dict:
    """FRED series specific to the ticker's sector, read like the dashboard."""
    return fred_tools.get_sector_macro(yft.get_info(ticker).get("sector"))


def recent_events(ticker: str) -> dict:
    """The 8-K filings of the last six months, among the five most recent, and
    the heaviest-volume sessions of the last 60, joined in code to each session's
    opening gap, close-to-close
    return, volume against its 50-day average and return against the sector
    ETF, with up to six press headlines per event. Ranked by size of move."""
    return event_tools.get_recent_events(ticker)


SHARED_DATA = [company_snapshot, macro_dashboard, sector_macro, recent_events]


# Sections not tied to one desk's own module.

def relative_strength(ticker: str) -> dict:
    """The stock against the S&P 500 and its sector ETF: excess returns over 1 to
    12 months, where the price ratio sits in its one-year range, and correlation."""
    return technical_tools.relative_strength(ticker)


def sector_rotation() -> dict:
    """All eleven sector ETFs ranked against the S&P 500 over 1, 3 and 6 months,
    plus equal-weight against cap-weight as a breadth reading."""
    return market_tools.sector_rotation()


def index_event_calendar() -> list[dict]:
    """Monthly options expirations and quarterly S&P index rebalance dates inside
    the next six months, computed from the calendar."""
    return market_tools.index_event_calendar()


def estimates(ticker: str) -> dict:
    """Analyst consensus EPS and revenue for the next two quarters and years, how
    estimates moved over 90 days, revision counts, and the last twelve earnings
    surprises."""
    return company_tools.get_estimates(ticker)
