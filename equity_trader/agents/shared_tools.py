"""Context tools every desk can call, and tools that more than one desk uses."""
from agents import function_tool

from equity_trader.data import company_tools, fred_tools, market_tools, technical_tools
from equity_trader.data import yfinance_tools as yft


@function_tool
def tool_get_company_snapshot(ticker: str) -> dict:
    """Today's date, the current price, sector and industry, 52-week range, beta,
    next earnings date, short interest and ownership. Call this first."""
    return company_tools.get_company_snapshot(ticker)


@function_tool
def tool_get_macro_dashboard() -> dict:
    """About fifty FRED readings on rates, inflation, growth, labor, credit,
    liquidity, markets and recession risk. Each has its value now, three months
    ago and a year ago, and its ten-year percentile. Also lists scheduled CPI,
    jobs, GDP and PCE releases."""
    return fred_tools.get_macro_dashboard()


@function_tool
def tool_get_sector_macro(ticker: str) -> dict:
    """FRED series specific to the ticker's sector, read like the dashboard."""
    return fred_tools.get_sector_macro(yft.get_info(ticker).get("sector"))


@function_tool
def tool_get_fred_series(series_id: str) -> dict:
    """Any FRED series by id: title, units, latest value, direction and its last
    twelve readings. Use tool_search_fred first if you do not know the id."""
    return fred_tools.get_fred_series(series_id)


@function_tool
def tool_search_fred(query: str) -> list[dict]:
    """Find FRED series ids by keyword. Returns the most-used matches."""
    return fred_tools.search_fred(query)


SHARED_TOOLS = [tool_get_company_snapshot, tool_get_macro_dashboard, tool_get_sector_macro,
                tool_get_fred_series, tool_search_fred]


# Tools used by more than one desk.

@function_tool
def tool_get_relative_strength(ticker: str) -> dict:
    """The stock against the S&P 500 and its sector ETF: excess returns over 1 to
    12 months, where the price ratio sits in its one-year range, and correlation."""
    return technical_tools.relative_strength(ticker)


@function_tool
def tool_get_sector_rotation() -> dict:
    """All eleven sector ETFs ranked against the S&P 500 over 1, 3 and 6 months,
    plus equal-weight against cap-weight as a breadth reading."""
    return market_tools.sector_rotation()


@function_tool
def tool_get_index_event_calendar() -> list[dict]:
    """Monthly options expirations and quarterly S&P index rebalance dates inside
    the next six months, computed from the calendar."""
    return market_tools.index_event_calendar()


@function_tool
def tool_get_estimates(ticker: str) -> dict:
    """Analyst consensus EPS and revenue for the next two quarters and years, how
    estimates moved over 90 days, revision counts, and the last twelve earnings
    surprises."""
    return company_tools.get_estimates(ticker)
