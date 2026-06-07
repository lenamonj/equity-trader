import time
import yfinance as yf
import pandas as pd

from equity_trader.data.cache import cached_for_run
from equity_trader.exceptions import TickerNotFoundError, DataFetchError

_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.1  # seconds


def _throttle() -> None:
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_CALL["t"] = time.monotonic()


@cached_for_run
def get_price_history(ticker: str, period: str = "2y",
                      interval: str = "1d") -> pd.DataFrame:
    _throttle()
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise TickerNotFoundError(ticker)
    return df


@cached_for_run
def get_quote(ticker: str) -> dict:
    _throttle()
    info = yf.Ticker(ticker).fast_info
    try:
        return {
            "last_price": float(info["lastPrice"]),
            "market_cap": float(info.get("marketCap") or 0.0),
            "day_high": float(info.get("dayHigh") or 0.0),
            "day_low": float(info.get("dayLow") or 0.0),
        }
    except (KeyError, TypeError) as e:
        raise DataFetchError(f"Quote unavailable for {ticker}: {e}")


@cached_for_run
def get_fundamentals(ticker: str) -> dict:
    _throttle()
    t = yf.Ticker(ticker)
    return {
        "income_stmt": t.income_stmt.to_dict() if t.income_stmt is not None else {},
        "balance_sheet": t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
        "cash_flow": t.cashflow.to_dict() if t.cashflow is not None else {},
    }


@cached_for_run
def get_analyst_targets(ticker: str) -> dict:
    _throttle()
    info = yf.Ticker(ticker).info or {}
    return {
        "mean_target": info.get("targetMeanPrice"),
        "high_target": info.get("targetHighPrice"),
        "low_target": info.get("targetLowPrice"),
        "num_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
    }


import numpy as np


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
    _throttle()
    info = yf.Ticker(ticker).info or {}
    key = (info.get("sector"), info.get("industry"))
    peers = _SECTOR_PEERS.get(key, [])
    return [p for p in peers if p.upper() != ticker.upper()]


@cached_for_run
def get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    _throttle()
    t = yf.Ticker(ticker)
    expiries = list(t.options or [])
    if not expiries:
        return {"expiries": [], "calls": [], "puts": [], "selected_expiry": None}
    chosen = expiry or expiries[min(2, len(expiries) - 1)]  # ~near-term but not weeklies
    chain = t.option_chain(chosen)
    return {
        "expiries": expiries,
        "selected_expiry": chosen,
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
                "selected_expiry": chain["selected_expiry"]}
    quote = get_quote(ticker)
    spot = quote["last_price"]
    atm_call = min(calls, key=lambda c: abs(c["strike"] - spot))
    atm_put = min(puts, key=lambda p: abs(p["strike"] - spot))
    return {
        "atm_iv": float(atm_call["impliedVolatility"]),
        "put_call_iv_skew": float(atm_put["impliedVolatility"]
                                   - atm_call["impliedVolatility"]),
        "selected_expiry": chain["selected_expiry"],
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

    # Relative strength vs peers over last 90 sessions
    base_ret = float(close.iloc[-1] / close.iloc[-90] - 1.0) if len(close) >= 90 else 0.0
    peer_rets = []
    for p in peers:
        try:
            pdf = get_price_history(p, period="6mo", interval="1d")
            peer_rets.append(float(pdf["Close"].iloc[-1] / pdf["Close"].iloc[0] - 1.0))
        except Exception:
            continue
    rel = base_ret - (sum(peer_rets) / len(peer_rets)) if peer_rets else 0.0

    return {
        "momentum_12_1": mom_12_1,
        "volatility_60d": vol_60,
        "rel_strength_vs_peers": rel,
        "peer_count_used": len(peer_rets),
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
        "autocorr_1d": ac1,
        "autocorr_5d": ac5,
        "monthly_seasonality": {int(k): float(v) for k, v in by_month.items()},
        "realized_vol_30d": realized_vol,
        "regime": regime,
    }
