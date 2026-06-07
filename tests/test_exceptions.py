import pytest
from equity_trader.exceptions import (
    DataFetchError,
    TickerNotFoundError,
    RateLimitError,
    AgentExecutionError,
)


def test_exception_hierarchy():
    assert issubclass(TickerNotFoundError, DataFetchError)
    assert issubclass(RateLimitError, DataFetchError)
    assert issubclass(AgentExecutionError, Exception)


def test_ticker_not_found_carries_ticker():
    err = TickerNotFoundError("XYZAB")
    assert err.ticker == "XYZAB"
    assert "XYZAB" in str(err)
