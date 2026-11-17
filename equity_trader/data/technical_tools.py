"""Chart reading done in code: trend, levels, volume and relative strength."""
import numpy as np
import pandas as pd

from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import iso, num

YEAR = 252  # trading sessions
_ANNUAL = YEAR ** 0.5


def history(ticker: str) -> pd.DataFrame:
    """Full daily history, shared by the technical, pattern and backtest desks."""
    return yft.get_price_history(ticker, period="max")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    value = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    # A window with gains and no losing session is the maximum reading.
    return value.where(~((loss == 0) & (gain > 0)), 100.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(),
                      (df["Low"] - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smooth = {"alpha": 1 / period, "adjust": False}
    avg_range = _true_range(df).ewm(**smooth).mean()
    plus_di = 100 * plus_dm.ewm(**smooth).mean() / avg_range
    minus_di = 100 * minus_dm.ewm(**smooth).mean() / avg_range
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return float(dx.ewm(**smooth).mean().iloc[-1])


def _last_cross(fast: pd.Series, slow: pd.Series) -> dict | None:
    above = (fast > slow)[slow.notna()]
    flips = above[above != above.shift()].iloc[1:]
    if flips.empty:
        return None
    return {"type": "golden cross" if flips.iloc[-1] else "death cross",
            "date": iso(flips.index[-1])}


def _vs(series: pd.Series, n: int) -> float | None:
    """Latest value relative to its own n-period average."""
    if len(series) < n:
        return None
    return num(series.iloc[-1] / series.tail(n).mean() - 1.0)


def _change(series: pd.Series, n: int) -> float | None:
    if len(series) <= n:
        return None
    return num(series.iloc[-1] / series.iloc[-(n + 1)] - 1.0)


@cached_for_run
def compute_technicals(ticker: str) -> dict:
    """Trend, momentum and volatility on daily, weekly and monthly bars."""
    df = history(ticker)
    close = df["Close"]
    last = float(close.iloc[-1])

    averages = {}
    for n in (20, 50, 100, 200):
        if len(close) >= n + 21:
            sma = close.rolling(n).mean()
            averages[f"sma_{n}"] = {"value": num(sma.iloc[-1], 2),
                                    "price_vs": num(last / sma.iloc[-1] - 1.0),
                                    "slope_20d": num(sma.iloc[-1] / sma.iloc[-21] - 1.0)}

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    mid, sd = close.rolling(20).mean().iloc[-1], close.rolling(20).std().iloc[-1]
    upper, lower = mid + 2 * sd, mid - 2 * sd
    vol_20 = close.pct_change().rolling(20).std() * _ANNUAL
    atr_now = float(atr(df).iloc[-1])
    weekly = close.resample("W-FRI").last().dropna()
    monthly = close.resample("ME").last().dropna()

    return {
        "as_of": iso(df.index[-1]),
        "last_close": num(last, 2),
        "averages": averages,
        "last_50_200_cross": _last_cross(close.rolling(50).mean(), close.rolling(200).mean()),
        "rsi_14": num(rsi(close).iloc[-1], 1),
        "macd": {"macd": num(macd.iloc[-1], 3), "signal": num(macd_signal.iloc[-1], 3),
                 "histogram": num(macd.iloc[-1] - macd_signal.iloc[-1], 3)},
        "bollinger_20": {"upper": num(upper, 2), "lower": num(lower, 2),
                         "position": num((last - lower) / (upper - lower), 2),
                         "width": num((upper - lower) / mid)},
        "atr_14": num(atr_now, 2),
        "atr_pct_of_price": num(atr_now / last),
        "adx_14": num(_adx(df), 1),
        "volatility_20d": num(vol_20.iloc[-1]),
        "volatility_20d_pctile_1y": round(float(
            (vol_20.tail(YEAR) <= vol_20.iloc[-1]).mean() * 100)),
        "drawdown_from_52w_high": num(last / close.tail(YEAR).max() - 1.0),
        "weekly": {"price_vs_10w_avg": _vs(weekly, 10), "price_vs_40w_avg": _vs(weekly, 40),
                   "rsi_14": num(rsi(weekly).iloc[-1], 1),
                   "return_13w": _change(weekly, 13), "return_26w": _change(weekly, 26),
                   "return_52w": _change(weekly, 52)},
        "monthly": {"price_vs_10m_avg": _vs(monthly, 10), "return_12m": _change(monthly, 12)},
    }


_PIVOT_SPAN = 10  # sessions on each side that a swing high or low must dominate
_GAP = 0.02


@cached_for_run
def key_levels(ticker: str) -> dict:
    """Support and resistance from the last year of trading."""
    full = history(ticker)
    df = full.tail(YEAR)
    close, high, low = df["Close"], df["High"], df["Low"]
    last = float(close.iloc[-1])
    atr_now = float(atr(full).iloc[-1])

    def level(price: float, when, kind: str) -> dict:
        return {"price": num(price, 2), "date": iso(when), "kind": kind,
                "pct_from_price": num(price / last - 1.0),
                "atr_multiple": num((price - last) / atr_now, 1)}

    window = 2 * _PIVOT_SPAN + 1
    swing_highs = high[high == high.rolling(window, center=True).max()]
    swing_lows = low[low == low.rolling(window, center=True).min()]
    pivots = ([level(p, d, "swing high") for d, p in swing_highs.items()]
              + [level(p, d, "swing low") for d, p in swing_lows.items()])
    resistance = sorted((p for p in pivots if p["price"] > last), key=lambda p: p["price"])
    support = sorted((p for p in pivots if p["price"] < last), key=lambda p: -p["price"])

    volume_by_price, edges = np.histogram(close, bins=20, weights=df["Volume"])
    busiest = np.argsort(volume_by_price)[::-1][:3]
    zones = [{"from": num(edges[i], 2), "to": num(edges[i + 1], 2),
              "share_of_volume": num(volume_by_price[i] / volume_by_price.sum(), 3)}
             for i in busiest]

    prev_close = close.shift()
    gap = df["Open"] / prev_close - 1.0
    open_gaps = []
    for when in gap[gap.abs() >= _GAP].index[::-1]:
        after = df.loc[when:]
        up = gap[when] > 0
        filled = (after["Low"].min() <= prev_close[when] if up
                  else after["High"].max() >= prev_close[when])
        if not filled:
            open_gaps.append({"date": iso(when), "direction": "up" if up else "down",
                              "from": num(prev_close[when], 2), "to": num(df["Open"][when], 2),
                              "pct": num(gap[when])})

    return {
        "as_of": iso(df.index[-1]),
        "last_close": num(last, 2),
        "atr_14": num(atr_now, 2),
        "resistance": resistance[:3],
        "support": support[:3],
        "high_52w": level(high.max(), high.idxmax(), "52-week high"),
        "low_52w": level(low.min(), low.idxmin(), "52-week low"),
        "all_time_high": level(full["High"].max(), full["High"].idxmax(), "all-time high"),
        "heaviest_volume_zones": zones,
        "unfilled_gaps": open_gaps[:3],
    }


@cached_for_run
def volume_analysis(ticker: str) -> dict:
    """Whether volume confirms the price trend."""
    df = history(ticker).tail(YEAR)
    close, volume = df["Close"], df["Volume"]
    ret = close.pct_change()
    avg_50 = volume.rolling(50).mean()
    obv = (np.sign(ret).fillna(0) * volume).cumsum()
    heavy = volume > avg_50

    def up_down(n: int) -> float | None:
        r, v = ret.tail(n), volume.tail(n)
        down = v[r < 0].sum()
        return num(v[r > 0].sum() / down, 2) if down else None

    recent = df.tail(60)
    heaviest = recent["Volume"].nlargest(3)
    return {
        "as_of": iso(df.index[-1]),
        "avg_volume_20d": num(volume.tail(20).mean(), 0),
        "avg_volume_50d": num(avg_50.iloc[-1], 0),
        "volume_20d_vs_1y_avg": num(volume.tail(20).mean() / volume.mean(), 2),
        "up_down_volume_ratio_20d": up_down(20),
        "up_down_volume_ratio_50d": up_down(50),
        "obv_trend_20d": "rising" if obv.iloc[-1] > obv.iloc[-21] else "falling",
        "price_change_20d": _change(close, 20),
        "accumulation_days_25": int(((ret >= 0.01) & heavy).tail(25).sum()),
        "distribution_days_25": int(((ret <= -0.01) & heavy).tail(25).sum()),
        "heaviest_days_60": [{"date": iso(when), "volume_vs_50d_avg": num(v / avg_50[when], 2),
                              "return": num(ret[when])} for when, v in heaviest.items()],
    }


@cached_for_run
def relative_strength(ticker: str) -> dict:
    """Performance against the S&P 500 and the sector ETF, and how stretched it is."""
    ticker = ticker.upper()
    sector_etf = yft.SECTOR_ETF.get(yft.get_info(ticker).get("sector"))
    benchmarks = ["SPY"] + ([sector_etf] if sector_etf else [])
    closes = yft.get_closes([ticker, *benchmarks], period="2y").dropna()
    own = closes[ticker]
    out: dict = {"as_of": iso(closes.index[-1]), "sector_etf": sector_etf, "versus": {}}
    for bench in benchmarks:
        ratio = (own / closes[bench]).tail(YEAR)
        low, high = ratio.min(), ratio.max()
        returns = closes[[ticker, bench]].pct_change().tail(60)
        out["versus"][bench] = {
            **{f"excess_return_{label}": num(_change(own, n) - _change(closes[bench], n))
               for label, n in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252))
               if _change(own, n) is not None},
            "ratio_position_in_1y_range": num((ratio.iloc[-1] - low) / (high - low), 2),
            "ratio_zscore_1y": num((ratio.iloc[-1] - ratio.mean()) / ratio.std(), 2),
            "ratio_change_50d": _change(ratio, 50),
            "correlation_60d": num(returns[ticker].corr(returns[bench]), 2),
        }
    return out
