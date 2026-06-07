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
