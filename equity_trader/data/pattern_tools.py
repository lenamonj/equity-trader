"""Statistical studies of a ticker's own history, each with its sample size."""
import numpy as np
import pandas as pd

from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.technical_tools import YEAR, history, rsi
from equity_trader.data.util import iso, num

_ANNUAL = YEAR ** 0.5
_HORIZONS = (5, 20, 63, 126)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _mean_stats(x: pd.Series, scale: float = 1.0, digits: int = 4) -> dict:
    """Mean, hit rate and t-statistic of a sample of returns."""
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "hit_rate": None, "t_stat": None}
    sd = x.std()
    return {"n": n, "mean": num(x.mean() * scale, digits), "hit_rate": num((x > 0).mean(), 2),
            "t_stat": num(x.mean() / (sd / n ** 0.5), 2) if n > 1 and sd > 0 else None}


@cached_for_run
def seasonality(ticker: str) -> dict:
    """Calendar effects over the full history: month, weekday and turn of month."""
    close = history(ticker)["Close"]
    monthly = close.resample("ME").last().pct_change().dropna()
    daily = close.pct_change().dropna()

    day_in_month = daily.groupby([daily.index.year, daily.index.month]).cumcount()
    days_left = daily.iloc[::-1].groupby([daily.index[::-1].year,
                                          daily.index[::-1].month]).cumcount().iloc[::-1]
    turn = (day_in_month < 3) | (days_left == 0)
    return {
        "as_of": iso(close.index[-1]),
        "history_start": iso(close.index[0]),
        "current_month": _MONTHS[close.index[-1].month - 1],
        "monthly_return_by_month": {
            _MONTHS[m - 1]: _mean_stats(monthly[monthly.index.month == m])
            for m in range(1, 13)},
        "daily_return_bps_by_weekday": {
            _WEEKDAYS[d]: _mean_stats(daily[daily.index.day_of_week == d], scale=1e4, digits=1)
            for d in range(5)},
        "turn_of_month": {
            "last_day_and_first_three_bps": _mean_stats(daily[turn], scale=1e4, digits=1),
            "all_other_days_bps": _mean_stats(daily[~turn], scale=1e4, digits=1)},
        "note": ("A t_stat beyond about 2 in either direction is the usual bar. Seventeen "
                 "calendar buckets are tested here, so about one will clear it by chance."),
    }


@cached_for_run
def earnings_event_study(ticker: str) -> dict:
    """How the stock has traded around each past earnings report."""
    close = history(ticker)["Close"].copy()
    close.index = pd.DatetimeIndex(close.index.date)
    reported = yft.get_earnings_dates(ticker).dropna(subset=["Reported EPS"])

    events = []
    for when, row in reported.iterrows():
        # A report after the close moves the next session; one before it, the same day.
        side = "right" if when.hour >= 16 else "left"
        pos = close.index.searchsorted(pd.Timestamp(when.date()), side=side)
        if pos < 6 or pos >= len(close):
            continue

        def after(n: int, pos: int = pos) -> float | None:
            if pos + n >= len(close):
                return None
            return num(close.iloc[pos + n] / close.iloc[pos] - 1.0)

        events.append({
            "date": iso(when), "surprise_pct": num(row["Surprise(%)"], 2),
            "pre_5d": num(close.iloc[pos - 1] / close.iloc[pos - 6] - 1.0),
            "reaction": num(close.iloc[pos] / close.iloc[pos - 1] - 1.0),
            "post_5d": after(5), "post_20d": after(20), "post_60d": after(60)})

    frame = pd.DataFrame(events)
    out: dict = {"as_of": iso(close.index[-1]), "reports": len(events), "events": events[:8]}
    if frame.empty:
        return out
    out["reaction_day"] = {"avg_absolute_move": num(frame["reaction"].abs().mean()),
                           "largest_drop": num(frame["reaction"].min()),
                           "largest_gain": num(frame["reaction"].max()),
                           "up_rate": num((frame["reaction"] > 0).mean(), 2)}
    out["pre_5d"] = _mean_stats(frame["pre_5d"])
    for label, group in (("after_beats", frame[frame["surprise_pct"] > 0]),
                         ("after_misses", frame[frame["surprise_pct"] <= 0])):
        out[label] = {"reaction": _mean_stats(group["reaction"]),
                      "post_20d": _mean_stats(group["post_20d"]),
                      "post_60d": _mean_stats(group["post_60d"])}
    return out


def _forward(close: pd.Series) -> dict[int, pd.Series]:
    return {h: close.shift(-h) / close - 1.0 for h in _HORIZONS}


def _episodes(mask: pd.Series) -> int:
    """Number of separate stretches in which a condition held."""
    return int((mask & ~mask.shift(fill_value=False)).sum())


def _outcomes(forward: dict[int, pd.Series], mask: pd.Series, horizons=_HORIZONS) -> dict:
    out: dict = {"days": int(mask.sum()), "episodes": _episodes(mask)}
    for h in horizons:
        x = forward[h][mask].dropna()
        out[f"next_{h}d"] = ({"mean": num(x.mean()), "median": num(x.median()),
                              "hit_rate": num((x > 0).mean(), 2)} if len(x) else None)
    return out


# Drawdown from the 52-week high: (upper edge, lower edge, label).
_DRAWDOWN_BUCKETS = [(0.0, -0.10, "within 10% of the 52-week high"),
                     (-0.10, -0.20, "10% to 20% below the 52-week high"),
                     (-0.20, -0.35, "20% to 35% below the 52-week high"),
                     (-0.35, -1.01, "more than 35% below the 52-week high")]


@cached_for_run
def forward_return_study(ticker: str) -> dict:
    """What followed past shocks, and what followed days that looked like today."""
    close = history(ticker)["Close"]
    ret = close.pct_change()
    zscore = ret / ret.rolling(60).std().shift()
    forward = _forward(close)

    baseline = {}
    for h in _HORIZONS:
        x = forward[h].dropna()
        baseline[f"next_{h}d"] = {"mean": num(x.mean()), "median": num(x.median()),
                                  "hit_rate": num((x > 0).mean(), 2),
                                  "p10": num(x.quantile(0.1)), "p90": num(x.quantile(0.9))}

    below_200 = close < close.rolling(200).mean()
    drawdown = close / close.rolling(YEAR).max() - 1.0
    upper, lower, bucket = next(b for b in _DRAWDOWN_BUCKETS if drawdown.iloc[-1] > b[1])
    states = [
        ("price below its 200-day average" if below_200.iloc[-1]
         else "price above its 200-day average",
         below_200 if below_200.iloc[-1] else ~below_200 & close.rolling(200).mean().notna()),
        (bucket, (drawdown <= upper) & (drawdown > lower)),
    ]
    rsi_now = rsi(close)
    if rsi_now.iloc[-1] < 30:
        states.append(("14-day RSI below 30", rsi_now < 30))
    elif rsi_now.iloc[-1] > 70:
        states.append(("14-day RSI above 70", rsi_now > 70))

    recent = zscore.tail(20)
    shocks = recent[recent.abs() >= 2]
    return {
        "as_of": iso(close.index[-1]),
        "history_start": iso(close.index[0]),
        "baseline_all_days": baseline,
        "after_down_shock": _outcomes(forward, zscore <= -2),
        "after_up_shock": _outcomes(forward, zscore >= 2),
        "shocks_in_last_20_sessions": [{"date": iso(d), "sigma": num(z, 1)}
                                       for d, z in shocks.items()],
        "days_like_today": [{"condition": name, **_outcomes(forward, mask, (63, 126))}
                            for name, mask in states],
        "note": ("A shock is a daily move of two or more standard deviations of the prior "
                 "60 sessions. Forward windows overlap, so `days` overstates the evidence; "
                 "`episodes` counts separate occurrences. Compare with baseline_all_days."),
    }


@cached_for_run
def volatility_and_gaps(ticker: str) -> dict:
    """Volatility regime, persistence of returns, and how opening gaps resolve."""
    df = history(ticker)
    close = df["Close"]
    vol_20 = close.pct_change().rolling(20).std() * _ANNUAL
    ret = close.pct_change().dropna()
    weekly = close.resample("W-FRI").last().pct_change().dropna()
    monthly = close.resample("ME").last().pct_change().dropna()

    forward_63 = close.shift(-63) / close - 1.0
    low_cut, high_cut = vol_20.quantile(1 / 3), vol_20.quantile(2 / 3)
    regimes = {"low": vol_20 <= low_cut, "middle": (vol_20 > low_cut) & (vol_20 <= high_cut),
               "high": vol_20 > high_cut}
    now = next(name for name, mask in regimes.items() if mask.iloc[-1])

    recent = df.tail(5 * YEAR)
    prev_close = recent["Close"].shift()
    gap = (recent["Open"] / prev_close - 1.0).dropna()
    gaps = {}
    for label, days in (("up", gap[gap >= 0.02].index), ("down", gap[gap <= -0.02].index)):
        filled, follow = 0, []
        for when in days:
            window = recent.loc[when:].head(5)
            filled += bool(window["Low"].min() <= prev_close[when] if label == "up"
                           else window["High"].max() >= prev_close[when])
            follow.append(recent["Close"][when] / recent["Open"][when] - 1.0)
        gaps[label] = {"count": len(days),
                       "filled_within_5_sessions": (num(filled / len(days), 2)
                                                    if len(days) else None),
                       "avg_open_to_close": num(np.mean(follow)) if follow else None}

    return {
        "as_of": iso(close.index[-1]),
        "realized_volatility": {"20d": num(vol_20.iloc[-1]),
                                "60d": num(ret.tail(60).std() * _ANNUAL),
                                "20d_pctile_5y": round(float(
                                    (vol_20.tail(5 * YEAR) <= vol_20.iloc[-1]).mean() * 100))},
        "volatility_clustering": {"squared_return_autocorr_1d": num((ret ** 2).autocorr(1), 2),
                                  "squared_return_autocorr_5d": num((ret ** 2).autocorr(5), 2)},
        "return_autocorrelation": {
            "daily_lag_1": {"value": num(ret.autocorr(1), 3), "n": len(ret)},
            "daily_lag_5": {"value": num(ret.autocorr(5), 3), "n": len(ret)},
            "weekly_lag_1": {"value": num(weekly.autocorr(1), 3), "n": len(weekly)},
            "monthly_lag_1": {"value": num(monthly.autocorr(1), 3), "n": len(monthly)}},
        "next_63d_return_by_volatility_regime": {
            name: {"days": int(mask.sum()), "mean": num(forward_63[mask].mean()),
                   "hit_rate": num((forward_63[mask].dropna() > 0).mean(), 2)}
            for name, mask in regimes.items()},
        "volatility_regime_now": now,
        "gaps_of_2pct_or_more_5y": gaps,
        "note": ("Positive return autocorrelation means moves tend to continue; negative "
                 "means they tend to reverse. Values within about 2 / sqrt(n) of zero are noise."),
    }
