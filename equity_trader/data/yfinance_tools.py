import logging
import time
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import DataFetchError, TickerNotFoundError

log = logging.getLogger(__name__)

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


@cached_for_run
def get_fundamentals(ticker: str) -> dict:
    throttle()
    t = yf.Ticker(ticker)
    return {
        "income_stmt": t.income_stmt.to_dict() if t.income_stmt is not None else {},
        "balance_sheet": t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
        "cash_flow": t.cashflow.to_dict() if t.cashflow is not None else {},
        "quarterly_income_stmt": (t.quarterly_income_stmt.iloc[:, :5].to_dict()
                                  if t.quarterly_income_stmt is not None else {}),
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


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


@cached_for_run
def compute_technicals(ticker: str) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return {
        "as_of": as_of(df),
        "last_close": float(close.iloc[-1]),
        "sma_50": float(close.rolling(50).mean().iloc[-1]),
        "sma_200": float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None,
        "rsi_14": _rsi(close),
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "atr_14": _atr(df),
        "vol_20": float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5)),
    }


# Simple sector-peer heuristic. Real peer discovery would use a richer source;
# yfinance's `info` doesn't expose peers directly, so we use a small hardcoded
# map keyed on sector/industry. Override at call-site if richer peers needed.
_SECTOR_PEERS = {
    ("Technology", "Semiconductors"): ["AMD", "INTC", "AVGO", "TSM", "QCOM"],
    ("Technology", "Software - Infrastructure"): ["MSFT", "ORCL", "CRM", "ADBE"],
    ("Technology", "Software - Application"): ["CRM", "NOW", "SNOW", "DDOG"],
    ("Communication Services", "Internet Content & Information"):
        ["GOOGL", "META", "PINS", "SNAP"],
    ("Consumer Cyclical", "Internet Retail"): ["AMZN", "MELI", "EBAY"],
    ("Financial Services", "Banks - Diversified"): ["JPM", "BAC", "C", "WFC"],
}


@cached_for_run
def get_peer_set(ticker: str) -> list[str]:
    info = get_info(ticker)
    key = (info.get("sector"), info.get("industry"))
    peers = _SECTOR_PEERS.get(key, [])
    return [p for p in peers if p.upper() != ticker.upper()]


_MIN_OPTION_DAYS = 90  # the horizon starts at three months


def _days_to(expiry: str, today: date) -> int:
    return (date.fromisoformat(expiry) - today).days


def _horizon_expiry(expiries: list[str], today: date) -> str:
    """First expiry inside the 3-6 month horizon, else the furthest listed."""
    for e in expiries:
        if _days_to(e, today) >= _MIN_OPTION_DAYS:
            return e
    return expiries[-1]


@cached_for_run
def get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    throttle()
    t = yf.Ticker(ticker)
    expiries = list(t.options or [])
    if not expiries:
        return {"expiries": [], "calls": [], "puts": [], "selected_expiry": None,
                "days_to_expiry": None}
    today = date.today()
    chosen = expiry or _horizon_expiry(expiries, today)
    chain = t.option_chain(chosen)
    return {
        "expiries": expiries,
        "selected_expiry": chosen,
        "days_to_expiry": _days_to(chosen, today),
        "calls": chain.calls[["strike", "lastPrice", "impliedVolatility",
                              "openInterest", "volume"]].to_dict(orient="records"),
        "puts": chain.puts[["strike", "lastPrice", "impliedVolatility",
                            "openInterest", "volume"]].to_dict(orient="records"),
    }


@cached_for_run
def compute_iv_stats(ticker: str) -> dict:
    chain = get_options_chain(ticker)
    calls = chain["calls"]
    puts = chain["puts"]
    if not calls or not puts:
        return {"atm_iv": None, "put_call_iv_skew": None,
                "selected_expiry": chain["selected_expiry"],
                "days_to_expiry": chain["days_to_expiry"]}
    quote = get_quote(ticker)
    spot = quote["last_price"]
    atm_call = min(calls, key=lambda c: abs(c["strike"] - spot))
    atm_put = min(puts, key=lambda p: abs(p["strike"] - spot))
    return {
        "atm_iv": float(atm_call["impliedVolatility"]),
        "put_call_iv_skew": float(atm_put["impliedVolatility"]
                                   - atm_call["impliedVolatility"]),
        "selected_expiry": chain["selected_expiry"],
        "days_to_expiry": chain["days_to_expiry"],
    }


@cached_for_run
def compute_factor_loads(ticker: str, peers: list[str]) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    close = df["Close"]
    # 12-1 momentum: return from t-252 to t-21
    if len(close) >= 252:
        mom_12_1 = float(close.iloc[-21] / close.iloc[-252] - 1.0)
    else:
        mom_12_1 = float(close.iloc[-1] / close.iloc[0] - 1.0)
    vol_60 = float(close.pct_change().tail(60).std() * (252 ** 0.5))

    # Relative strength vs peers, both measured over the same 90 sessions
    window = 90
    base_ret = _window_return(df, window)
    peer_rets = []
    for p in peers:
        try:
            peer_ret = _window_return(get_price_history(p, period="1y", interval="1d"), window)
        except Exception as exc:
            log.warning("Peer %s unavailable for %s: %s", p, ticker, exc)
            continue
        if peer_ret is not None:
            peer_rets.append(peer_ret)
    rel = (base_ret - sum(peer_rets) / len(peer_rets)
           if peer_rets and base_ret is not None else None)

    return {
        "as_of": as_of(df),
        "momentum_12_1": mom_12_1,
        "volatility_60d": vol_60,
        "rel_strength_vs_peers": rel,
        "peer_count_used": len(peer_rets),
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
    return float(df["Close"].iloc[-1] / df["Close"].iloc[-(days + 1)] - 1.0)


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
        return float(cov / var) if var > 0 else None
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
    vol_ratio = (recent_5 / avg_60) if (avg_60 and avg_60 > 0) else None

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


@cached_for_run
def compute_statistical_patterns(ticker: str) -> dict:
    df = get_price_history(ticker, period="2y", interval="1d")
    rets = df["Close"].pct_change().dropna()
    ac1 = float(rets.autocorr(lag=1) or 0.0)
    ac5 = float(rets.autocorr(lag=5) or 0.0)
    by_month = rets.groupby(rets.index.month).mean().to_dict()
    realized_vol = float(rets.tail(30).std() * (252 ** 0.5))
    if ac1 > 0.05:
        regime = "trending"
    elif ac1 < -0.05:
        regime = "mean_reverting"
    else:
        regime = "mixed"
    return {
        "as_of": as_of(df),
        "autocorr_1d": ac1,
        "autocorr_5d": ac5,
        "monthly_seasonality": {int(k): float(v) for k, v in by_month.items()},
        "realized_vol_30d": realized_vol,
        "regime": regime,
    }
