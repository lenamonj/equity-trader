import time
from datetime import date
from functools import wraps

import pandas as pd
import yfinance as yf
from yfinance.data import YfData

from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import iso, num
from equity_trader.exceptions import DataFetchError, TickerNotFoundError

# yfinance hands back a refused request as an empty result, which then reads
# as "no options listed" or "no estimates". A request Yahoo still refuses
# after yfinance's own retry gets one more try after a pause, then raises.
_REFUSED = {401, 403, 429}
_RETRY_PAUSE = 2.0  # seconds


def _refusals_raise(make_request):
    @wraps(make_request)
    def checked(self, url, *args, **kwargs):
        response = make_request(self, url, *args, **kwargs)
        if response.status_code in _REFUSED:
            time.sleep(_RETRY_PAUSE)
            response = make_request(self, url, *args, **kwargs)
        if response.status_code in _REFUSED:
            raise DataFetchError(
                f"Yahoo refused {url.split('?')[0]} with HTTP {response.status_code}")
        return response
    return checked


YfData._make_request = _refusals_raise(YfData._make_request)

_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.1  # seconds


def throttle() -> None:
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL["t"] = time.monotonic()


def as_of(df: pd.DataFrame) -> str:
    """Date of the last bar, so every derived number can be dated."""
    return df.index[-1].date().isoformat()


@cached_for_run
def get_price_history(ticker: str, period: str = "2y",
                      interval: str = "1d") -> pd.DataFrame:
    throttle()
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise TickerNotFoundError(ticker)
    return df


@cached_for_run
def get_info(ticker: str) -> dict:
    """Yahoo's company profile and summary statistics, fetched once per run."""
    throttle()
    return yf.Ticker(ticker).info or {}


@cached_for_run
def get_closes(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Daily closes for several symbols in one request, one column per symbol."""
    throttle()
    data = yf.download(symbols, period=period, progress=False, auto_adjust=True)
    if data.empty:
        raise DataFetchError(f"No price data for {symbols}")
    return data["Close"]


@cached_for_run
def get_earnings_dates(ticker: str) -> pd.DataFrame:
    """Past and scheduled earnings dates with estimate, reported EPS and surprise."""
    throttle()
    frame = yf.Ticker(ticker).get_earnings_dates(limit=28)
    if frame is None:
        return pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
    return frame


@cached_for_run
def get_quote(ticker: str) -> dict:
    throttle()
    info = yf.Ticker(ticker).fast_info
    try:
        return {
            "last_price": float(info["lastPrice"]),
            "market_cap": float(info.get("marketCap") or 0.0),
            "day_high": float(info.get("dayHigh") or 0.0),
            "day_low": float(info.get("dayLow") or 0.0),
        }
    except (KeyError, TypeError) as e:
        # yfinance returns an empty quote for symbols it does not know.
        raise TickerNotFoundError(ticker) from e


# yfinance statement row -> the name used in tool output.
_STATEMENT_LINES = {
    "Total Revenue": "revenue", "Gross Profit": "gross_profit",
    "Operating Income": "operating_income", "Net Income": "net_income",
    "Diluted EPS": "eps_diluted", "Operating Cash Flow": "operating_cash_flow",
    "Capital Expenditure": "capex", "Free Cash Flow": "free_cash_flow",
    "Cash And Cash Equivalents": "cash", "Total Debt": "total_debt",
    "Stockholders Equity": "stockholders_equity",
}


def _statement_periods(frames: list, limit: int) -> list[dict]:
    """Selected lines from yfinance statement frames, one dict per period, newest first."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return []
    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated()]
    out = []
    for end in sorted(merged.columns, reverse=True)[:limit]:
        row: dict = {"period_end": iso(end)}
        for label, name in _STATEMENT_LINES.items():
            if label in merged.index:
                value = num(merged.loc[label, end], 2)
                # yfinance reports capital spending as a negative number.
                row[name] = abs(value) if name == "capex" and value is not None else value
        out.append(row)
    return out


def _ratio(a, b) -> float | None:
    return num(a / b) if a is not None and b else None


def _add_ratios(periods: list[dict], lag: int) -> list[dict]:
    """Margins, growth on the period `lag` rows earlier, and leverage for each
    period, so a desk handed these statements does not have to compute them."""
    for i, row in enumerate(periods):
        revenue = row.get("revenue")
        earlier = periods[i + lag].get("revenue") if i + lag < len(periods) else None
        row["ratios"] = {
            "gross_margin": _ratio(row.get("gross_profit"), revenue),
            "operating_margin": _ratio(row.get("operating_income"), revenue),
            "net_margin": _ratio(row.get("net_income"), revenue),
            "free_cash_flow_margin": _ratio(row.get("free_cash_flow"), revenue),
            "revenue_growth_yoy": (num(revenue / earlier - 1.0)
                                   if revenue is not None and earlier and earlier > 0 else None),
            "debt_to_equity": _ratio(row.get("total_debt"), row.get("stockholders_equity")),
        }
    return periods


@cached_for_run
def get_fundamentals(ticker: str) -> dict:
    """Statements from yfinance, for a company whose SEC statements cannot be
    used: four fiscal years and five quarters, with ratios."""
    throttle()
    t = yf.Ticker(ticker)
    return {
        "annual": _add_ratios(
            _statement_periods([t.income_stmt, t.cashflow, t.balance_sheet], 4), lag=1),
        "quarterly": _add_ratios(_statement_periods(
            [t.quarterly_income_stmt, t.quarterly_cashflow, t.quarterly_balance_sheet], 5),
            lag=4),
        "valuation": get_valuation(ticker),
    }


@cached_for_run
def get_valuation(ticker: str) -> dict:
    info = get_info(ticker)
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "peg": info.get("trailingPegRatio"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "as_of": date.today().isoformat(),
    }


@cached_for_run
def get_analyst_targets(ticker: str) -> dict:
    info = get_info(ticker)
    return {
        "mean_target": info.get("targetMeanPrice"),
        "high_target": info.get("targetHighPrice"),
        "low_target": info.get("targetLowPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        "as_of": date.today().isoformat(),
    }


# Yahoo / yfinance sector strings -> primary SPDR sector ETF symbol, the
# benchmark for every sector-relative reading.
SECTOR_ETF = {
    "Technology":              "XLK",
    "Financial Services":      "XLF",
    "Healthcare":              "XLV",
    "Communication Services":  "XLC",
    "Consumer Cyclical":       "XLY",
    "Consumer Defensive":      "XLP",
    "Energy":                  "XLE",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
}

