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


@cached_for_run
def get_fundamentals(ticker: str) -> dict:
    """Statements from yfinance, for companies with no US-GAAP filings on EDGAR."""
    throttle()
    t = yf.Ticker(ticker)
    return {
        "annual": _statement_periods([t.income_stmt, t.cashflow, t.balance_sheet], 4),
        "quarterly": _statement_periods(
            [t.quarterly_income_stmt, t.quarterly_cashflow, t.quarterly_balance_sheet], 5),
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


# Yahoo / yfinance sector strings -> primary SPDR sector ETF symbol. Used by
# the Jane Street ETF Observer agent to anchor relative-value comparisons.
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


def _window_return(df, days: int) -> float | None:
    """Return over the last `days` sessions, or None if history is too short."""
    if df is None or len(df) < days + 1:
        return None
    return num(df["Close"].iloc[-1] / df["Close"].iloc[-(days + 1)] - 1.0)


def _beta(ticker_df, market_df) -> float | None:
    if ticker_df is None or market_df is None:
        return None
    try:
        t = ticker_df["Close"].pct_change().dropna()
        m = market_df["Close"].pct_change().dropna()
        joined = pd.concat([t, m], axis=1, join="inner").dropna()
        if len(joined) < 30:
            return None
        cov = joined.iloc[:, 0].cov(joined.iloc[:, 1])
        var = joined.iloc[:, 1].var()
        return num(cov / var, 3) if var > 0 else None
    except Exception:
        return None


@cached_for_run
def get_etf_landscape(ticker: str) -> dict:
    """Return ETF-flow and sector-rotation context for a single ticker.

    Fields:
      - sector: yfinance sector string (or None)
      - sector_etf: primary SPDR sector ETF symbol (or None)
      - ticker_return_1m/3m/6m: returns over those windows
      - spy_return_1m/3m/6m: SPY benchmark returns over the same windows
      - sector_etf_return_1m/3m/6m: sector ETF returns
      - beta_to_spy: 1y daily-return beta vs SPY
      - beta_to_sector: 1y daily-return beta vs sector ETF
      - recent_volume_vs_60d_avg: trailing-5d avg volume / trailing-60d avg
    """
    info = get_info(ticker)
    sector = info.get("sector")
    sector_etf = SECTOR_ETF.get(sector)

    ticker_df = get_price_history(ticker, period="1y", interval="1d")
    try:
        spy_df = get_price_history("SPY", period="1y", interval="1d")
    except Exception:
        spy_df = None
    sector_df = None
    if sector_etf:
        try:
            sector_df = get_price_history(sector_etf, period="1y", interval="1d")
        except Exception:
            sector_df = None

    avg_60 = float(ticker_df["Volume"].tail(60).mean()) if len(ticker_df) >= 60 else None
    recent_5 = float(ticker_df["Volume"].tail(5).mean()) if len(ticker_df) >= 5 else None
    vol_ratio = num(recent_5 / avg_60, 2) if (avg_60 and avg_60 > 0) else None

    return {
        "as_of": as_of(ticker_df),
        "sector": sector,
        "sector_etf": sector_etf,
        "ticker_return_1m":     _window_return(ticker_df, 21),
        "ticker_return_3m":     _window_return(ticker_df, 63),
        "ticker_return_6m":     _window_return(ticker_df, 126),
        "spy_return_1m":        _window_return(spy_df, 21),
        "spy_return_3m":        _window_return(spy_df, 63),
        "spy_return_6m":        _window_return(spy_df, 126),
        "sector_etf_return_1m": _window_return(sector_df, 21),
        "sector_etf_return_3m": _window_return(sector_df, 63),
        "sector_etf_return_6m": _window_return(sector_df, 126),
        "beta_to_spy":          _beta(ticker_df, spy_df),
        "beta_to_sector":       _beta(ticker_df, sector_df),
        "recent_volume_vs_60d_avg": vol_ratio,
    }
