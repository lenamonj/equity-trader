class DataFetchError(Exception):
    """Base for data layer failures."""


class CommitteeFailedError(Exception):
    """No desk produced a report, so there is nothing to base a verdict on."""

    def __init__(self, ticker: str, first_error: str):
        super().__init__(f"Every desk failed on {ticker}. First error: {first_error}")
        self.ticker = ticker


class TickerNotFoundError(DataFetchError):
    def __init__(self, ticker: str):
        super().__init__(f"Ticker not found: {ticker}")
        self.ticker = ticker
