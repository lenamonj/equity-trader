"""The market around the stock: other assets, sectors, styles, ETFs and past stress."""
import calendar
import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from equity_trader.data import fred_tools
from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.technical_tools import history
from equity_trader.data.util import iso, num

log = logging.getLogger(__name__)

CROSS_ASSET = {"S&P 500": "SPY", "Nasdaq 100": "QQQ", "Small caps": "IWM",
               "Long Treasuries": "TLT", "High yield bonds": "HYG", "Gold": "GC=F",
               "Copper": "HG=F", "Crude oil": "CL=F", "US dollar index": "DX-Y.NYB"}
STYLES = {"Momentum": "MTUM", "Quality": "QUAL", "Value": "VLUE",
          "Low volatility": "USMV", "Small caps": "IWM"}
_WINDOWS = (("1m", 21), ("3m", 63), ("6m", 126))


def _returns(series: pd.Series) -> dict:
    series = series.dropna()
    return {f"return_{label}": (num(series.iloc[-1] / series.iloc[-(n + 1)] - 1.0)
                                if len(series) > n else None)
            for label, n in _WINDOWS}


def _board(symbols: dict[str, str]) -> tuple[str, dict]:
    closes = yft.get_closes(list(symbols.values()), period="1y")
    return iso(closes.index[-1]), {
        label: {"symbol": symbol, **_returns(closes[symbol])} for label, symbol in symbols.items()}


def _with_excess(rows: dict, benchmark: dict) -> dict:
    """Add each row's return beyond the benchmark, and order by the 3-month figure."""
    for row in rows.values():
        for label, _ in _WINDOWS:
            own, base = row[f"return_{label}"], benchmark[f"return_{label}"]
            row[f"excess_{label}"] = (num(own - base)
                                      if own is not None and base is not None else None)
    return dict(sorted(rows.items(), key=lambda kv: -(kv[1]["excess_3m"] or -9)))


@cached_for_run
def cross_asset_board() -> dict:
    """One, three and six month returns across the major asset classes."""
    as_of, rows = _board(CROSS_ASSET)
    return {"as_of": as_of, "assets": rows}


@cached_for_run
def sector_rotation() -> dict:
    """Every sector against the S&P 500, best to worst over three months."""
    sectors = {sector: etf for sector, etf in yft.SECTOR_ETF.items()}
    as_of, rows = _board({**sectors, "S&P 500": "SPY", "S&P 500 equal weight": "RSP"})
    market, equal_weight = rows.pop("S&P 500"), rows.pop("S&P 500 equal weight")
    return {
        "as_of": as_of,
        "market": market,
        "sectors": _with_excess(rows, market),
        "breadth_equal_weight_minus_cap_weight": {
            label: num(equal_weight[f"return_{label}"] - market[f"return_{label}"])
            for label, _ in _WINDOWS},
    }


@cached_for_run
def style_board() -> dict:
    """Which investment styles have been paid, measured against the S&P 500."""
    as_of, rows = _board({**STYLES, "S&P 500": "SPY"})
    market = rows.pop("S&P 500")
    return {"as_of": as_of, "market": market, "styles": _with_excess(rows, market)}


def _liquidity(ticker: str, info: dict) -> dict:
    df = yft.get_price_history(ticker, period="1y", interval="1d").tail(60)
    dollars = df["Close"] * df["Volume"]
    impact = (df["Close"].pct_change().abs() / dollars).mean()
    float_shares = info.get("floatShares")
    return {
        "avg_daily_dollar_volume_20d": num(dollars.tail(20).mean(), 0),
        "avg_daily_dollar_volume_60d": num(dollars.mean(), 0),
        "volume_5d_vs_60d": num(df["Volume"].tail(5).mean() / df["Volume"].mean(), 2),
        "daily_turnover_of_float": (num(df["Volume"].tail(20).mean() / float_shares, 5)
                                    if float_shares else None),
        "price_move_per_1bn_traded": num(impact * 1e9, 5),
    }


@cached_for_run
def etf_exposure(ticker: str) -> dict:
    """The stock's weight in the broad, sector and industry ETFs that hold it."""
    ticker = ticker.upper()
    info = yft.get_info(ticker)
    etfs = ["SPY", "QQQ"]
    sector_etf = yft.SECTOR_ETF.get(info.get("sector"))
    if sector_etf:
        etfs.append(sector_etf)
    if info.get("sectorKey"):
        yft.throttle()
        try:
            etfs += [e for e in yf.Sector(info["sectorKey"]).top_etfs if e not in etfs]
        except Exception as exc:
            log.warning("Sector ETF list unavailable for %s: %s", ticker, exc)

    rows = []
    for etf in etfs[:6]:
        yft.throttle()
        try:
            holdings = yf.Ticker(etf).funds_data.top_holdings
            fund = yft.get_info(etf)
        except Exception as exc:
            log.warning("ETF holdings unavailable for %s: %s", etf, exc)
            continue
        held = ticker in holdings.index
        volume, avg_volume = fund.get("volume"), fund.get("averageVolume")
        rows.append({
            "etf": etf, "name": fund.get("shortName"),
            "weight": num(holdings.loc[ticker, "Holding Percent"]) if held else None,
            "rank_in_fund": list(holdings.index).index(ticker) + 1 if held else None,
            "fund_assets": fund.get("totalAssets"),
            "volume_vs_average": num(volume / avg_volume, 2) if volume and avg_volume else None,
        })
    return {
        "as_of": date.today().isoformat(),
        "etfs": rows,
        "liquidity": _liquidity(ticker, info),
        "note": ("Each fund reports its ten largest holdings. A null weight means the stock "
                 "is outside the top ten, not absent from the fund. Creation and redemption "
                 "flow history is not available; volume_vs_average is the nearest "
                 "proxy."),
    }


def _third_friday(year: int, month: int) -> date:
    fridays = [d for d in calendar.Calendar().itermonthdates(year, month)
               if d.month == month and d.weekday() == calendar.FRIDAY]
    return fridays[2]


def index_event_calendar(today: date | None = None, horizon_days: int = 180) -> list[dict]:
    """Options expirations and quarterly S&P index rebalances inside the horizon."""
    today = today or date.today()
    cutoff = today + timedelta(days=horizon_days)
    out, year, month = [], today.year, today.month
    while date(year, month, 1) <= cutoff:
        friday = _third_friday(year, month)
        if today <= friday <= cutoff:
            quarterly = month in (3, 6, 9, 12)
            out.append({"date": friday.isoformat(),
                        "event": ("Quarterly options and futures expiration; S&P index "
                                  "rebalance takes effect after the close" if quarterly
                                  else "Monthly options expiration")})
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# S&P 500 peak to trough dates for the major drawdowns since 2007.
STRESS_WINDOWS = {"2008 financial crisis": ("2007-10-09", "2009-03-09"),
                  "Late 2018 selloff": ("2018-09-20", "2018-12-24"),
                  "2020 pandemic crash": ("2020-02-19", "2020-03-23"),
                  "2022 rate shock": ("2022-01-03", "2022-10-12")}


def _daily_close(ticker: str) -> pd.Series:
    close = history(ticker)["Close"].copy()
    close.index = pd.DatetimeIndex(close.index.date)
    return close


def _worst_drawdowns(close: pd.Series, n: int = 5) -> list[dict]:
    peak_id = (close.cummax() != close.cummax().shift()).cumsum()
    starts = close.groupby(peak_id).head(1).index
    rows = []
    for i, (_, run) in enumerate(close.groupby(peak_id)):
        depth = run.min() / run.iloc[0] - 1.0
        if depth < 0:
            rows.append({"peak": iso(run.index[0]), "trough": iso(run.idxmin()),
                         "depth": num(depth),
                         "recovered": iso(starts[i + 1]) if i + 1 < len(starts) else None})
    return sorted(rows, key=lambda r: r["depth"])[:n]


@cached_for_run
def stress_history(ticker: str) -> dict:
    """How the stock held up in past market drawdowns, and its own worst declines."""
    close, market = _daily_close(ticker), _daily_close("SPY")
    episodes = {}
    for name, (start, end) in STRESS_WINDOWS.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if close.index[0] > start:
            episodes[name] = {"traded": False}
            continue
        base = close.asof(start)
        episodes[name] = {"traded": True,
                          "stock_return": num(close.asof(end) / base - 1.0),
                          "stock_worst_point": num(close[start:end].min() / base - 1.0),
                          "market_return": num(market.asof(end) / market.asof(start) - 1.0)}
    return {
        "as_of": iso(close.index[-1]),
        "market_drawdowns": episodes,
        "current_drawdown_from_all_time_high": num(close.iloc[-1] / close.max() - 1.0),
        "worst_drawdowns_of_the_stock": _worst_drawdowns(close),
    }


# FRED series, how to turn it into a weekly change, and the size of move to quote.
_FACTORS = {
    "10-year Treasury yield": ("DGS10", "diff", 0.10, "a 10 basis point rise"),
    "High yield spread": ("BAMLH0A0HYM2", "diff", 0.10, "a 10 basis point widening"),
    "Broad dollar": ("DTWEXBGS", "pct", 0.01, "a 1% rise"),
    "WTI crude oil": ("DCOILWTICO", "pct", 0.10, "a 10% rise"),
    "VIX": ("VIXCLS", "diff", 1.0, "a 1 point rise"),
}


@cached_for_run
def macro_sensitivity(ticker: str) -> dict:
    """How the stock has moved, week by week, with rates, credit, the dollar and oil."""
    ticker = ticker.upper()
    closes = yft.get_closes([ticker, "SPY"], period="3y")
    weekly = closes.resample("W-FRI").last().pct_change().dropna()
    stock = weekly[ticker]
    start = (date.today() - timedelta(days=3 * 366)).isoformat()

    def relation(change: pd.Series, unit: float, per: str) -> dict:
        joined = pd.concat([stock, change], axis=1, join="inner").dropna()
        own, factor = joined.iloc[:, 0], joined.iloc[:, 1]
        return {"correlation": num(own.corr(factor), 2),
                "stock_move": num(own.cov(factor) / factor.var() * unit),
                "per": per, "weeks": len(joined)}

    factors = {"S&P 500": relation(weekly["SPY"], 0.01, "a 1% rise")}
    for label, (series_id, kind, unit, per) in _FACTORS.items():
        try:
            series = fred_tools.get_series(series_id, start=start).dropna()
        except Exception as exc:
            log.warning("Macro sensitivity: %s unavailable: %s", series_id, exc)
            continue
        level = series.resample("W-FRI").last()
        factors[label] = relation(level.diff() if kind == "diff" else level.pct_change(),
                                  unit, per)
    return {"as_of": iso(closes.index[-1]), "window": "weekly changes over three years",
            "factors": factors,
            "note": ("stock_move is the average change in the stock that accompanied the "
                     "stated move in the factor. It is a historical association, measured "
                     "one factor at a time, not a forecast.")}
