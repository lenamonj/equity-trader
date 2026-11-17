from datetime import date

import numpy as np
import pandas as pd
import pytest

from equity_trader.data import yfinance_tools as yft
from equity_trader.data.cache import reset_run_cache
from equity_trader.exceptions import TickerNotFoundError


def test_get_price_history_empty_raises(monkeypatch):
    class FakeTicker:
        def history(self, period, interval):
            return pd.DataFrame()

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    with pytest.raises(TickerNotFoundError):
        yft.get_price_history("ZZZZ")


def test_get_quote_returns_dict(monkeypatch):
    class FakeTicker:
        fast_info = {"lastPrice": 190.5, "marketCap": 4.6e12,
                     "dayHigh": 192.0, "dayLow": 188.0}

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    q = yft.get_quote("NVDA")
    assert q["last_price"] == 190.5
    assert q["market_cap"] == 4.6e12


@pytest.mark.slow
def test_get_price_history_real_aapl():
    df = yft.get_price_history("AAPL", period="1mo", interval="1d")
    assert len(df) > 10
    assert "Close" in df.columns


def _fake_history(monkeypatch, n=300):
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    df = pd.DataFrame({
        "Open": prices, "High": prices + 0.5, "Low": prices - 0.5,
        "Close": prices, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))

    class FakeTicker:
        def history(self, period, interval):
            return df

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    return df


def test_get_etf_landscape_returns_keys(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()

    rng = np.random.default_rng(7)
    prices = 100 + np.cumsum(rng.normal(0.05, 1.0, 300))
    df = pd.DataFrame({
        "Open": prices, "High": prices + 0.5, "Low": prices - 0.5,
        "Close": prices, "Volume": rng.integers(1_000_000, 5_000_000, 300),
    }, index=pd.date_range("2024-01-01", periods=300, freq="B"))

    class FakeTicker:
        info = {"sector": "Technology", "industry": "Semiconductors"}
        def history(self, period, interval):
            return df

    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    out = yft.get_etf_landscape("NVDA")
    expected = {
        "sector", "sector_etf",
        "ticker_return_1m", "ticker_return_3m", "ticker_return_6m",
        "spy_return_1m", "spy_return_3m", "spy_return_6m",
        "sector_etf_return_1m", "sector_etf_return_3m", "sector_etf_return_6m",
        "beta_to_spy", "beta_to_sector",
        "recent_volume_vs_60d_avg",
    }
    assert expected.issubset(out.keys())
    assert out["sector_etf"] == "XLK"  # Technology -> XLK


def test_valuation_and_analyst_targets_are_dated(monkeypatch):
    class FakeTicker:
        info = {"trailingPE": 19.5, "targetMeanPrice": 267.65}

    reset_run_cache()
    monkeypatch.setattr(yft.yf, "Ticker", lambda t: FakeTicker())
    today = date.today().isoformat()
    assert yft.get_valuation("NVDA")["as_of"] == today
    assert yft.get_analyst_targets("NVDA")["as_of"] == today


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


def _yahoo(statuses, monkeypatch):
    """A fake yfinance request method answering with `statuses` in turn."""
    seen = []
    monkeypatch.setattr(yft.time, "sleep", lambda seconds: seen.append(("pause", seconds)))

    def make_request(self, url):
        seen.append(url)
        return _Response(statuses[len([s for s in seen if s == url]) - 1])

    return yft._refusals_raise(make_request), seen


def test_every_yahoo_request_goes_through_the_refusal_check():
    from yfinance.data import YfData
    assert YfData._make_request.__wrapped__.__name__ == "_make_request"


def test_a_refused_yahoo_request_is_retried_once_after_a_pause(monkeypatch):
    request, seen = _yahoo([401, 200], monkeypatch)
    assert request(None, "https://q/options/NVDA").status_code == 200
    assert seen == ["https://q/options/NVDA", ("pause", yft._RETRY_PAUSE),
                    "https://q/options/NVDA"]


def test_a_yahoo_request_refused_twice_raises(monkeypatch):
    from equity_trader.exceptions import DataFetchError
    request, _ = _yahoo([401, 401], monkeypatch)
    with pytest.raises(DataFetchError, match="Yahoo refused https://q/options/NVDA with HTTP 401"):
        request(None, "https://q/options/NVDA?crumb=abc")


def test_other_yahoo_answers_pass_through_untouched(monkeypatch):
    for status in (200, 404):
        request, seen = _yahoo([status], monkeypatch)
        assert request(None, "https://q/x").status_code == status
        assert seen == ["https://q/x"]


def test_fallback_statements_are_compact_and_dated():
    periods = pd.to_datetime(["2025-12-31", "2024-12-31"])
    income = pd.DataFrame({periods[0]: [100.0, 20.0, 1.0], periods[1]: [80.0, 10.0, 2.0]},
                          index=["Total Revenue", "Net Income", "Tax Rate For Calcs"])
    cash = pd.DataFrame({periods[0]: [-7.0], periods[1]: [-5.0]}, index=["Capital Expenditure"])
    out = yft._statement_periods([income, cash, None], limit=4)
    assert out == [
        {"period_end": "2025-12-31", "revenue": 100.0, "net_income": 20.0, "capex": 7.0},
        {"period_end": "2024-12-31", "revenue": 80.0, "net_income": 10.0, "capex": 5.0}]
    assert yft._statement_periods([None], limit=4) == []
