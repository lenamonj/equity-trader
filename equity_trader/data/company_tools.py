"""Company-level context: snapshot, estimates, analysts, insiders, peers, factors."""
import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import yfinance as yf

from equity_trader.data import calendar_tools
from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import cached_for_run
from equity_trader.data.util import iso, num

log = logging.getLogger(__name__)


def _ratio(a, b) -> float | None:
    return num(a / b - 1.0) if a and b else None


@cached_for_run
def get_company_snapshot(ticker: str) -> dict:
    """Today's date, price and the basic facts every desk needs before it starts."""
    info = yft.get_info(ticker)
    price = yft.get_quote(ticker)["last_price"]
    avg_volume = info.get("averageVolume")
    short_date = info.get("dateShortInterest")
    return {
        "today": date.today().isoformat(),
        "ticker": ticker.upper(),
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "business": (info.get("longBusinessSummary") or "")[:700] or None,
        "price": num(price, 2),
        "market_cap": info.get("marketCap"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low": info.get("fiftyTwoWeekLow"),
        "pct_from_52w_high": _ratio(price, info.get("fiftyTwoWeekHigh")),
        "pct_from_52w_low": _ratio(price, info.get("fiftyTwoWeekLow")),
        "beta": num(info.get("beta"), 2),
        "avg_daily_dollar_volume": num(avg_volume * price, 0) if avg_volume else None,
        "dividend_yield": num(info["dividendRate"] / price) if info.get("dividendRate") else None,
        "next_earnings_date": calendar_tools.next_earnings_date(ticker, date.today()),
        "short_interest": {
            "pct_of_float": num(info.get("shortPercentOfFloat")),
            "days_to_cover": num(info.get("shortRatio"), 2),
            "change_vs_prior_month": _ratio(info.get("sharesShort"),
                                            info.get("sharesShortPriorMonth")),
            "as_of": (datetime.fromtimestamp(short_date, UTC).date().isoformat()
                      if short_date else None),
        },
        "ownership": {
            "institutions_pct": num(info.get("heldPercentInstitutions")),
            "insiders_pct": num(info.get("heldPercentInsiders")),
        },
    }


_PERIODS = {"0q": "current_quarter", "+1q": "next_quarter",
            "0y": "current_year", "+1y": "next_year"}


def _cell(frame, row: str, col: str) -> float | None:
    if frame is None or row not in frame.index or col not in frame.columns:
        return None
    return num(frame.loc[row, col])


@cached_for_run
def get_estimates(ticker: str) -> dict:
    """Analyst consensus, how it has moved, and the record of past surprises."""
    yft.throttle()
    t = yf.Ticker(ticker)
    eps, revenue = t.earnings_estimate, t.revenue_estimate
    trend, revisions = t.eps_trend, t.eps_revisions
    consensus = {}
    for code, name in _PERIODS.items():
        now, ago_90 = _cell(trend, code, "current"), _cell(trend, code, "90daysAgo")
        consensus[name] = {
            "eps_avg": _cell(eps, code, "avg"),
            "eps_low": _cell(eps, code, "low"),
            "eps_high": _cell(eps, code, "high"),
            "eps_year_ago": _cell(eps, code, "yearAgoEps"),
            "eps_growth": _cell(eps, code, "growth"),
            "analysts": _cell(eps, code, "numberOfAnalysts"),
            "revenue_avg": _cell(revenue, code, "avg"),
            "revenue_growth": _cell(revenue, code, "growth"),
            "eps_30d_ago": _cell(trend, code, "30daysAgo"),
            "eps_90d_ago": ago_90,
            "eps_change_90d": _ratio(now, ago_90),
            "revisions_up_30d": _cell(revisions, code, "upLast30days"),
            "revisions_down_30d": _cell(revisions, code, "downLast30days"),
        }

    reported = yft.get_earnings_dates(ticker).dropna(subset=["Reported EPS"])
    surprises = [{"date": iso(when), "eps_estimate": num(row["EPS Estimate"]),
                  "eps_reported": num(row["Reported EPS"]),
                  "surprise_pct": num(row["Surprise(%)"], 2)}
                 for when, row in reported.head(12).iterrows()]
    return {
        "as_of": date.today().isoformat(),
        "source": "yfinance analyst consensus",
        "consensus": consensus,
        "surprises": surprises,
        "beats": sum(1 for s in surprises if (s["surprise_pct"] or 0) > 0),
        "reports": len(surprises),
    }


@cached_for_run
def get_analyst_view(ticker: str) -> dict:
    """Price targets, the rating mix over three months, and recent rating actions."""
    yft.throttle()
    t = yf.Ticker(ticker)
    ratings = t.recommendations
    actions = t.upgrades_downgrades
    out: dict = {**yft.get_analyst_targets(ticker),
                 "rating_mix": [] if ratings is None else ratings.to_dict(orient="records"),
                 "recent_actions": [], "last_90_days": {}, "last_30_days": {}}
    if actions is None or actions.empty:
        return out

    actions = actions.copy()
    actions.index = pd.to_datetime(actions.index).tz_localize(None)
    now = pd.Timestamp.now()
    for label, days in (("last_90_days", 90), ("last_30_days", 30)):
        window = actions[actions.index >= now - pd.Timedelta(days=days)]
        out[label] = {
            "upgrades": int((window["Action"] == "up").sum()),
            "downgrades": int((window["Action"] == "down").sum()),
            "targets_raised": int((window["priceTargetAction"] == "Raises").sum()),
            "targets_lowered": int((window["priceTargetAction"] == "Lowers").sum()),
        }
    recent = actions[actions.index >= now - pd.Timedelta(days=90)].head(15)
    out["recent_actions"] = [
        {"date": iso(when), "firm": row["Firm"], "action": row["Action"],
         "from": row["FromGrade"], "to": row["ToGrade"],
         "target": num(row["currentPriceTarget"], 2) or None,
         "prior_target": num(row["priorPriceTarget"], 2) or None}
        for when, row in recent.iterrows()]
    return out


def _insider_kind(text: str) -> str:
    if text.startswith("Sale"):
        return "sale"
    if text.startswith("Purchase"):
        return "purchase"
    return "other"  # awards, gifts, option exercises


@cached_for_run
def get_insider_activity(ticker: str) -> dict:
    """Open-market insider buying and selling over six months, and top holders."""
    yft.throttle()
    t = yf.Ticker(ticker)
    out: dict = {"source": "yfinance, from SEC Form 4 and 13F filings",
                 "window": "last 6 months",
                 "sales": {"count": 0, "shares": 0, "value": 0},
                 "purchases": {"count": 0, "shares": 0, "value": 0},
                 "recent": [], "top_institutions": []}

    tx = t.insider_transactions
    if tx is not None and not tx.empty:
        tx = tx.assign(when=pd.to_datetime(tx["Start Date"]),
                       kind=tx["Text"].fillna("").map(_insider_kind))
        tx = tx[tx["when"] >= pd.Timestamp.now() - pd.DateOffset(months=6)]
        for kind, key in (("sale", "sales"), ("purchase", "purchases")):
            rows = tx[tx["kind"] == kind]
            out[key] = {"count": len(rows), "shares": int(rows["Shares"].sum()),
                        "value": int(rows["Value"].fillna(0).sum())}
        out["recent"] = [
            {"date": iso(row["when"]), "insider": row["Insider"], "position": row["Position"],
             "transaction": row["Text"], "shares": int(row["Shares"]),
             "value": num(row["Value"], 0)}
            for _, row in tx.sort_values("when", ascending=False).head(10).iterrows()]

    holders = t.institutional_holders
    if holders is not None and not holders.empty:
        out["top_institutions"] = [
            {"holder": row["Holder"], "pct_held": num(row["pctHeld"]),
             "pct_change": num(row["pctChange"]), "reported": iso(row["Date Reported"])}
            for _, row in holders.head(5).iterrows()]
    return out


@cached_for_run
def get_peers(ticker: str, n: int = 5) -> list[str]:
    """The largest companies in the ticker's industry, by market weight."""
    key = yft.get_info(ticker).get("industryKey")
    if not key:
        return []
    yft.throttle()
    try:
        top = yf.Industry(key).top_companies
    except Exception as exc:
        log.warning("Industry peers unavailable for %s: %s", ticker, exc)
        return []
    if top is None:
        return []
    return [s for s in top.index if s.upper() != ticker.upper()][:n]


# Factor -> metric -> True when a higher value is the more attractive one.
_FACTORS = {
    "value": {"trailing_pe": False, "forward_pe": False, "ev_to_ebitda": False,
              "price_to_sales": False, "fcf_yield": True},
    "quality": {"return_on_equity": True, "gross_margin": True, "operating_margin": True,
                "debt_to_equity": False},
    "growth": {"revenue_growth": True, "earnings_growth": True},
    "momentum": {"return_12_1": True, "return_6m": True, "return_3m": True},
    "low_volatility": {"volatility_60d": False, "beta": False},
}


def _window_return(close: pd.Series, start: int, end: int = 1) -> float | None:
    """Return from `start` sessions ago to `end` sessions ago (1 = the latest close)."""
    close = close.dropna()
    if len(close) < start:
        return None
    return float(close.iloc[-end] / close.iloc[-start] - 1.0)


def _factor_metrics(symbol: str, close: pd.Series) -> dict:
    info = yft.get_info(symbol)
    cap, fcf = info.get("marketCap"), info.get("freeCashflow")
    returns = close.dropna().pct_change().dropna()
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "fcf_yield": fcf / cap if fcf and cap else None,
        "return_on_equity": info.get("returnOnEquity"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "return_12_1": _window_return(close, 252, 21),
        "return_6m": _window_return(close, 126),
        "return_3m": _window_return(close, 63),
        "volatility_60d": (float(returns.tail(60).std() * 252 ** 0.5)
                           if len(returns) >= 60 else None),
        "beta": info.get("beta"),
    }


@cached_for_run
def get_factor_scorecard(ticker: str) -> dict:
    """Value, quality, growth, momentum and volatility metrics ranked against peers.

    Rank 1 is the most attractive reading among the companies with data.
    """
    ticker = ticker.upper()
    peers = get_peers(ticker)
    symbols = [ticker, *peers]
    closes = yft.get_closes(symbols, period="2y")
    metrics = {}
    for symbol in symbols:
        try:
            metrics[symbol] = _factor_metrics(symbol, closes[symbol])
        except Exception as exc:
            log.warning("Factor metrics unavailable for %s: %s", symbol, exc)

    factors: dict = {}
    for factor, spec in _FACTORS.items():
        factors[factor] = {}
        for metric, higher_is_better in spec.items():
            values = {s: num(m.get(metric)) for s, m in metrics.items()}
            have = {s: v for s, v in values.items() if v is not None}
            value = values.get(ticker)
            peer_values = [v for s, v in have.items() if s != ticker]
            rank = None
            if value is not None:
                better = [v for v in have.values()
                          if (v > value if higher_is_better else v < value)]
                rank = len(better) + 1
            factors[factor][metric] = {
                "value": value,
                "peer_median": num(np.median(peer_values)) if peer_values else None,
                "rank": rank,
                "of": len(have),
            }
    return {
        "as_of": iso(closes.index[-1]),
        "peers": peers,
        "market_cap": yft.get_info(ticker).get("marketCap"),
        "factors": factors,
    }


def _max_drawdown(close: pd.Series) -> float:
    return float((close / close.cummax() - 1.0).min())


@cached_for_run
def get_risk_stats(ticker: str) -> dict:
    """Return, volatility, drawdown and market dependence over one and three years."""
    closes = yft.get_closes([ticker.upper(), "SPY"], period="3y").dropna()
    close, market = closes[ticker.upper()], closes["SPY"]
    returns = pd.concat([close.pct_change(), market.pct_change()], axis=1).dropna().tail(252)
    own, mkt = returns.iloc[:, 0], returns.iloc[:, 1]
    beta = float(own.cov(mkt) / mkt.var())
    corr = float(own.corr(mkt))
    vol = float(own.std() * 252 ** 0.5)
    total = float(close.iloc[-1] / close.iloc[-len(returns) - 1] - 1.0)
    return {
        "as_of": iso(closes.index[-1]),
        "window_sessions": len(returns),
        "return_1y": num(total),
        "volatility_1y": num(vol),
        "return_per_unit_of_volatility": num(total / vol),
        "max_drawdown_1y": num(_max_drawdown(close.tail(252))),
        "max_drawdown_3y": num(_max_drawdown(close)),
        "worst_day": num(own.min()),
        "best_day": num(own.max()),
        "skew": num(own.skew(), 2),
        "excess_kurtosis": num(own.kurt(), 2),
        "beta_to_spy": num(beta, 2),
        "correlation_to_spy": num(corr, 2),
        "stock_specific_volatility": num(float((own - beta * mkt).std() * 252 ** 0.5)),
        "stock_specific_share_of_variance": num(1.0 - corr ** 2, 2),
    }
