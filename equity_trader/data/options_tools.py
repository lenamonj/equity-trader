"""What the options market is pricing: term structure, skew, positioning, premium."""
from datetime import date

import pandas as pd
import yfinance as yf

from equity_trader.data import calendar_tools, pattern_tools
from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.technical_tools import YEAR, history
from equity_trader.data.util import iso, num

_ANNUAL = YEAR ** 0.5
_HORIZON_DAYS = 180
_TARGET_DAYS = (7, 30, 60, 90, _HORIZON_DAYS, 365)
_MIN_OPTION_DAYS = 90  # the horizon starts at three months
_NOTE = ("Quotes are delayed and implied volatility on thinly traded strikes can be stale. "
         "implied_move is the at-the-money straddle price over spot: the move priced in by "
         "that expiry. skew_90_110 is the IV of the put 10% below spot minus the IV of the "
         "call 10% above; positive means downside protection costs more.")


def _days_to(expiry: str, today: date) -> int:
    return (date.fromisoformat(expiry) - today).days


def horizon_expiry(expiries: list[str], today: date) -> str:
    """First expiry inside the 3-6 month horizon, else the furthest listed."""
    for e in expiries:
        if _days_to(e, today) >= _MIN_OPTION_DAYS:
            return e
    return expiries[-1]


def pick_expiries(expiries: list[str], today: date) -> list[str]:
    """The listed expiry nearest each target maturity, at least a week out.

    The six-month pick is the last expiry inside the horizon, not the nearest
    to it: the nearest was often a month past the horizon's end.
    """
    usable = [e for e in expiries if _days_to(e, today) >= 7]
    inside = [e for e in usable if _days_to(e, today) <= _HORIZON_DAYS]
    picked: list[str] = []
    for target in _TARGET_DAYS:
        if not usable:
            break
        if target == _HORIZON_DAYS and inside:
            best = inside[-1]
        else:
            best = min(usable, key=lambda e, target=target: abs(_days_to(e, today) - target))
        if best not in picked:
            picked.append(best)
    return sorted(picked)


@cached_for_run
def _expiries(ticker: str) -> list[str]:
    yft.throttle()
    return list(yf.Ticker(ticker).options or [])


@cached_for_run
def _chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    yft.throttle()
    chain = yf.Ticker(ticker).option_chain(expiry)
    return chain.calls, chain.puts


def _nearest(frame: pd.DataFrame, strike: float) -> pd.Series:
    return frame.iloc[(frame["strike"] - strike).abs().argmin()]


def _mid(row: pd.Series) -> float:
    if row["bid"] > 0 and row["ask"] > 0:
        return float(row["bid"] + row["ask"]) / 2
    return float(row["lastPrice"])


def _iv(row: pd.Series) -> float | None:
    value = row["impliedVolatility"]
    return float(value) if value and value > 0.01 else None  # Yahoo zero-fills stale quotes


def _total(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].fillna(0).sum())


def _largest_oi(frame: pd.DataFrame) -> list[dict]:
    top = frame.assign(oi=frame["openInterest"].fillna(0)).nlargest(3, "oi")
    return [{"strike": num(r["strike"], 2), "open_interest": int(r["oi"])}
            for _, r in top.iterrows()]


def _expiry_summary(ticker: str, expiry: str, spot: float, today: date,
                    earnings: str | None) -> dict:
    calls, puts = _chain(ticker, expiry)
    atm_call, atm_put = _nearest(calls, spot), _nearest(puts, spot)
    atm_ivs = [v for v in (_iv(atm_call), _iv(atm_put)) if v is not None]
    iv_put, iv_call = _iv(_nearest(puts, 0.9 * spot)), _iv(_nearest(calls, 1.1 * spot))
    call_volume, put_volume = _total(calls, "volume"), _total(puts, "volume")
    call_oi, put_oi = _total(calls, "openInterest"), _total(puts, "openInterest")
    return {
        "expiry": expiry,
        "days": _days_to(expiry, today),
        "spans_next_earnings": bool(earnings and earnings <= expiry),
        "atm_strike": num(atm_call["strike"], 2),
        "atm_iv": num(sum(atm_ivs) / len(atm_ivs)) if atm_ivs else None,
        "implied_move": num((_mid(atm_call) + _mid(atm_put)) / spot),
        "iv_put_90pct": num(iv_put),
        "iv_call_110pct": num(iv_call),
        "skew_90_110": num(iv_put - iv_call) if iv_put and iv_call else None,
        "put_call_volume_ratio": num(put_volume / call_volume, 2) if call_volume else None,
        "put_call_oi_ratio": num(put_oi / call_oi, 2) if call_oi else None,
        "call_volume": call_volume, "put_volume": put_volume,
        "call_open_interest": call_oi, "put_open_interest": put_oi,
        "largest_call_oi": _largest_oi(calls),
        "largest_put_oi": _largest_oi(puts),
    }


def _unusual(ticker: str, expiries: list[str]) -> list[dict]:
    """Strikes trading at least three times their open interest today."""
    rows = []
    for expiry in expiries:
        for kind, frame in zip(("call", "put"), _chain(ticker, expiry), strict=True):
            # Yahoo leaves volume or open interest empty on a strike with none.
            frame = frame.fillna({"volume": 0, "openInterest": 0})
            volume, oi = frame["volume"], frame["openInterest"]
            for _, r in frame[(volume >= 1000) & (volume > 3 * oi)].iterrows():
                rows.append({"expiry": expiry, "type": kind, "strike": num(r["strike"], 2),
                             "volume": int(r["volume"]),
                             "open_interest": int(r["openInterest"])})
    return sorted(rows, key=lambda r: -r["volume"])[:5]


@cached_for_run
def options_surface(ticker: str) -> dict:
    """Implied volatility, implied move, skew and positioning across maturities."""
    today = date.today()
    expiries = pick_expiries(_expiries(ticker), today)
    if not expiries:
        return {"as_of": today.isoformat(), "expiries": [], "note": "No listed options."}
    spot = yft.get_quote(ticker)["last_price"]
    earnings = calendar_tools.next_earnings_date(ticker, today)
    rows = [_expiry_summary(ticker, e, spot, today, earnings) for e in expiries]
    calls = sum(r["call_volume"] for r in rows)
    call_oi = sum(r["call_open_interest"] for r in rows)
    return {
        "as_of": today.isoformat(),
        "spot": num(spot, 2),
        "next_earnings_date": earnings,
        "expiries": rows,
        "all_listed_expiries_shown": {
            "put_call_volume_ratio": (num(sum(r["put_volume"] for r in rows) / calls, 2)
                                      if calls else None),
            "put_call_oi_ratio": (num(sum(r["put_open_interest"] for r in rows) / call_oi, 2)
                                  if call_oi else None)},
        "unusual_volume": _unusual(ticker, expiries),
        "note": _NOTE,
    }


def _closest(rows: list[dict], days: int) -> dict | None:
    priced = [r for r in rows if r["atm_iv"] is not None]
    return min(priced, key=lambda r: abs(r["days"] - days)) if priced else None


@cached_for_run
def volatility_premium(ticker: str) -> dict:
    """Implied volatility against what the stock has actually delivered."""
    surface = options_surface(ticker)
    returns = history(ticker)["Close"].pct_change().dropna()
    vol_20 = returns.rolling(20).std() * _ANNUAL
    realized = {f"{n}d": num(returns.tail(n).std() * _ANNUAL) for n in (20, 60, 120)}
    month, quarter = _closest(surface["expiries"], 30), _closest(surface["expiries"], 90)

    study = pattern_tools.earnings_event_study(ticker)
    past_moves = [abs(e["reaction"]) for e in study["events"]]
    next_event = next((r for r in surface["expiries"] if r["spans_next_earnings"]), None)
    return {
        "as_of": surface["as_of"],
        "realized_volatility": realized,
        "realized_20d_pctile_5y": round(float(
            (vol_20.tail(5 * YEAR) <= vol_20.iloc[-1]).mean() * 100)),
        "implied_1m": month and {"expiry": month["expiry"], "atm_iv": month["atm_iv"]},
        "implied_3m": quarter and {"expiry": quarter["expiry"], "atm_iv": quarter["atm_iv"]},
        "implied_minus_realized_1m": (num(month["atm_iv"] - realized["20d"])
                                      if month else None),
        "implied_minus_realized_3m": (num(quarter["atm_iv"] - realized["60d"])
                                      if quarter else None),
        "earnings": {
            "avg_absolute_move_last_reports": (num(sum(past_moves) / len(past_moves))
                                               if past_moves else None),
            "reports_measured": len(past_moves),
            "first_expiry_after_next_report": next_event and {
                "expiry": next_event["expiry"], "days": next_event["days"],
                "implied_move": next_event["implied_move"]}},
        "note": ("A positive implied_minus_realized means options price more movement than "
                 "the stock has recently delivered. The implied move for an expiry covers "
                 "every day to that expiry, not the earnings day alone."),
    }


@cached_for_run
def options_chain_near_money(ticker: str) -> dict:
    """Prices for the horizon expiry, limited to liquid strikes within 20% of spot."""
    today = date.today()
    expiries = _expiries(ticker)
    if not expiries:
        return {"selected_expiry": None, "calls": [], "puts": []}
    expiry = horizon_expiry(expiries, today)
    spot = yft.get_quote(ticker)["last_price"]

    def trim(frame: pd.DataFrame) -> list[dict]:
        near = frame[(frame["strike"] >= 0.8 * spot) & (frame["strike"] <= 1.2 * spot)
                     & (frame["openInterest"].fillna(0) > 0)]
        return [{"strike": num(r["strike"], 2), "price": num(_mid(r), 2), "iv": num(_iv(r)),
                 "open_interest": int(r["openInterest"]), "volume": int(r["volume"] or 0)}
                for _, r in near.fillna({"volume": 0}).iterrows()]

    calls, puts = _chain(ticker, expiry)
    return {"selected_expiry": expiry, "days_to_expiry": _days_to(expiry, today),
            "spot": num(spot, 2), "calls": trim(calls), "puts": trim(puts)}


_VOL_INDEXES = {"VIX (30-day S&P 500 implied volatility)": "^VIX",
                "VIX9D (9-day)": "^VIX9D",
                "VIX3M (3-month)": "^VIX3M",
                "SKEW (S&P 500 tail-risk pricing)": "^SKEW",
                "MOVE (Treasury implied volatility)": "^MOVE"}


@cached_for_run
def market_volatility() -> dict:
    """Index volatility: level, one-month change and where it sits in the past year."""
    closes = yft.get_closes(list(_VOL_INDEXES.values()), period="1y").ffill()
    rows = {}
    for label, symbol in _VOL_INDEXES.items():
        series = closes[symbol].dropna()
        rows[label] = {"value": num(series.iloc[-1], 2),
                       "one_month_ago": num(series.iloc[-22], 2) if len(series) > 22 else None,
                       "pctile_1y": round(float((series <= series.iloc[-1]).mean() * 100))}
    slope = closes["^VIX3M"].iloc[-1] - closes["^VIX"].iloc[-1]
    return {
        "as_of": iso(closes.index[-1]),
        "indexes": rows,
        "vix_term_structure": {
            "vix3m_minus_vix": num(slope, 2),
            "state": ("normal: near-term risk priced below longer-term" if slope > 0
                      else "inverted: near-term stress priced above longer-term")},
    }
