from equity_trader.exceptions import DataFetchError, TickerNotFoundError


def test_exception_hierarchy():
    assert issubclass(TickerNotFoundError, DataFetchError)


def test_ticker_not_found_carries_ticker():
    err = TickerNotFoundError("XYZAB")
    assert err.ticker == "XYZAB"
    assert "XYZAB" in str(err)


def test_a_failed_committee_names_the_ticker_and_the_first_error():
    from equity_trader.exceptions import CommitteeFailedError
    err = CommitteeFailedError("TGT", "RuntimeError: boom")
    assert str(err) == "Every desk failed on TGT. First error: RuntimeError: boom"
