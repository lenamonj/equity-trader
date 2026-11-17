from datetime import date

import numpy as np
import pandas as pd
import pytest

from equity_trader.data import company_tools as ct


def _closes(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range(end="2026-11-17", periods=n))


def test_snapshot_gives_date_price_and_derived_context(monkeypatch):
    monkeypatch.setattr(ct.yft, "get_info", lambda t: {
        "shortName": "Acme", "sector": "Technology", "industry": "Semiconductors",
        "fiftyTwoWeekHigh": 200.0, "fiftyTwoWeekLow": 80.0, "averageVolume": 1_000_000,
        "sharesShort": 90, "sharesShortPriorMonth": 100, "dateShortInterest": 1793318400,
        "dividendRate": 2.0, "heldPercentInstitutions": 0.7})
    monkeypatch.setattr(ct.yft, "get_quote", lambda t: {"last_price": 100.0})
    monkeypatch.setattr(ct.calendar_tools, "next_earnings_date", lambda t, today: "2027-02-01")

    out = ct.get_company_snapshot("acme")

    assert out["today"] == date.today().isoformat()
    assert out["ticker"] == "ACME" and out["price"] == 100.0
    assert out["pct_from_52w_high"] == -0.5 and out["pct_from_52w_low"] == 0.25
    assert out["avg_daily_dollar_volume"] == 100_000_000
    assert out["dividend_yield"] == 0.02
    assert out["next_earnings_date"] == "2027-02-01"
    assert out["short_interest"]["change_vs_prior_month"] == -0.1
    assert out["short_interest"]["as_of"] == "2026-10-30"


def test_estimates_report_consensus_revisions_and_surprises(monkeypatch):
    periods = ["0q", "+1q", "0y", "+1y"]

    class FakeTicker:
        earnings_estimate = pd.DataFrame(
            {"avg": [2.0, 2.2, 8.0, 10.0], "numberOfAnalysts": [40, 41, 50, 51]}, index=periods)
        revenue_estimate = pd.DataFrame({"avg": [100.0, 110.0, 400.0, 500.0]}, index=periods)
        eps_trend = pd.DataFrame({"current": [2.0, 2.2, 8.0, 10.0],
                                  "90daysAgo": [1.6, 2.2, 8.0, 10.0]}, index=periods)
        eps_revisions = None  # yfinance returns None when a table is missing

    reported = pd.DataFrame(
        {"EPS Estimate": [2.5, 2.0, 1.8], "Reported EPS": [np.nan, 2.2, 1.7],
         "Surprise(%)": [np.nan, 10.0, -5.56]},
        index=pd.to_datetime(["2026-11-17", "2026-08-26", "2026-05-20"]))
    monkeypatch.setattr(ct.yf, "Ticker", lambda t: FakeTicker())
    monkeypatch.setattr(ct.yft, "get_earnings_dates", lambda t: reported)

    out = ct.get_estimates("NVDA")

    quarter = out["consensus"]["current_quarter"]
    assert quarter["eps_avg"] == 2.0 and quarter["revenue_avg"] == 100.0
    assert quarter["eps_change_90d"] == 0.25
    assert quarter["revisions_up_30d"] is None
    assert [s["date"] for s in out["surprises"]] == ["2026-08-26", "2026-05-20"]
    assert (out["beats"], out["reports"]) == (1, 2)


def test_analyst_view_counts_actions_inside_each_window(monkeypatch):
    now = pd.Timestamp.now()
    actions = pd.DataFrame(
        {"Firm": ["A", "B", "C"], "ToGrade": ["Hold", "Buy", "Buy"],
         "FromGrade": ["Buy", "Buy", "Hold"], "Action": ["down", "main", "up"],
         "priceTargetAction": ["Lowers", "Raises", "Raises"],
         "currentPriceTarget": [180.0, 300.0, 250.0], "priorPriceTarget": [315.0, 280.0, 0.0]},
        index=[now - pd.Timedelta(days=5), now - pd.Timedelta(days=60),
               now - pd.Timedelta(days=200)])

    class FakeTicker:
        recommendations = pd.DataFrame({"period": ["0m"], "buy": [36], "hold": [14]})
        upgrades_downgrades = actions

    monkeypatch.setattr(ct.yf, "Ticker", lambda t: FakeTicker())
    monkeypatch.setattr(ct.yft, "get_analyst_targets", lambda t: {"mean_target": 267.0})

    out = ct.get_analyst_view("NVDA")

    assert out["mean_target"] == 267.0
    assert out["last_30_days"] == {"upgrades": 0, "downgrades": 1,
                                   "targets_raised": 0, "targets_lowered": 1}
    assert out["last_90_days"]["targets_raised"] == 1
    assert [a["firm"] for a in out["recent_actions"]] == ["A", "B"]
    assert out["recent_actions"][0]["prior_target"] == 315.0


def test_insider_activity_separates_open_market_trades_from_awards(monkeypatch):
    now = pd.Timestamp.now()
    tx = pd.DataFrame({
        "Shares": [100, 50, 999, 70],
        "Value": [10_000.0, 6_000.0, 0.0, 9_000.0],
        "Text": ["Sale at price 100.00 per share.", "Purchase at price 120.00 per share.",
                 "Stock Award(Grant) at price 0.00 per share.", "Sale at price 128.57 per share."],
        "Insider": ["A", "B", "C", "D"], "Position": ["CFO", "Director", "Officer", "CEO"],
        "Start Date": [now - pd.Timedelta(days=10), now - pd.Timedelta(days=20),
                       now - pd.Timedelta(days=30), now - pd.Timedelta(days=400)],
    })

    class FakeTicker:
        insider_transactions = tx
        institutional_holders = None

    monkeypatch.setattr(ct.yf, "Ticker", lambda t: FakeTicker())

    out = ct.get_insider_activity("NVDA")

    assert out["sales"] == {"count": 1, "shares": 100, "value": 10_000}
    assert out["purchases"] == {"count": 1, "shares": 50, "value": 6_000}
    assert len(out["recent"]) == 3  # the 400-day-old sale is outside the window
    assert out["top_institutions"] == []


def test_peers_are_the_largest_in_the_industry_excluding_the_ticker(monkeypatch):
    class FakeIndustry:
        top_companies = pd.DataFrame({"name": list("abcd")}, index=["NVDA", "AVGO", "AMD", "MU"])

    monkeypatch.setattr(ct.yft, "get_info", lambda t: {"industryKey": "semiconductors"})
    monkeypatch.setattr(ct.yf, "Industry", lambda key: FakeIndustry())
    assert ct.get_peers("nvda", n=2) == ["AVGO", "AMD"]


def test_peers_empty_when_the_industry_is_unknown(monkeypatch):
    monkeypatch.setattr(ct.yft, "get_info", lambda t: {})
    assert ct.get_peers("XYZ") == []


def test_factor_scorecard_ranks_in_the_attractive_direction(monkeypatch):
    info = {"AAA": {"trailingPE": 10.0, "returnOnEquity": 0.30},
            "BBB": {"trailingPE": 20.0, "returnOnEquity": 0.10},
            "CCC": {"trailingPE": 30.0, "returnOnEquity": 0.20}}
    flat = np.linspace(100.0, 110.0, 300)
    monkeypatch.setattr(ct, "get_peers", lambda t: ["BBB", "CCC"])
    monkeypatch.setattr(ct.yft, "get_info", lambda s: info[s])
    monkeypatch.setattr(ct.yft, "get_closes",
                        lambda symbols, period: _closes({s: flat for s in symbols}))

    out = ct.get_factor_scorecard("aaa")

    pe = out["factors"]["value"]["trailing_pe"]
    assert (pe["value"], pe["peer_median"], pe["rank"], pe["of"]) == (10.0, 25.0, 1, 3)
    assert out["factors"]["quality"]["return_on_equity"]["rank"] == 1
    assert out["factors"]["value"]["forward_pe"] == {
        "value": None, "peer_median": None, "rank": None, "of": 0}
    assert out["peers"] == ["BBB", "CCC"]


def test_risk_stats_recover_beta_and_market_dependence(monkeypatch):
    rng = np.random.default_rng(0)
    market_returns = rng.normal(0.0005, 0.01, 400)
    market = 100 * np.cumprod(1 + market_returns)
    stock = 100 * np.cumprod(1 + 2 * market_returns)
    monkeypatch.setattr(ct.yft, "get_closes",
                        lambda symbols, period: _closes({"ACME": stock, "SPY": market}))

    out = ct.get_risk_stats("acme")

    assert out["beta_to_spy"] == pytest.approx(2.0, abs=0.01)
    assert out["correlation_to_spy"] == 1.0
    assert out["stock_specific_share_of_variance"] == 0.0
    assert out["max_drawdown_3y"] <= out["max_drawdown_1y"] <= 0
