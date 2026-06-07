import pytest
import pandas as pd
from equity_trader.data import yfinance_tools as yft
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


import numpy as np


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


def test_compute_technicals_returns_keys(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()
    _fake_history(monkeypatch)
    out = yft.compute_technicals("AAPL")
    assert {"rsi_14", "macd", "macd_signal", "sma_50", "sma_200",
            "atr_14", "last_close", "vol_20"}.issubset(out.keys())
    assert 0 <= out["rsi_14"] <= 100


def test_compute_statistical_patterns_returns_keys(monkeypatch):
    from equity_trader.data.cache import reset_run_cache
    reset_run_cache()
    _fake_history(monkeypatch)
    out = yft.compute_statistical_patterns("AAPL")
    assert {"autocorr_1d", "autocorr_5d", "monthly_seasonality",
            "regime", "realized_vol_30d"}.issubset(out.keys())
    assert out["regime"] in {"trending", "mean_reverting", "mixed"}
