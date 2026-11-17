class DataFetchError(Exception):
    """Base for data layer failures."""


class TickerNotFoundError(DataFetchError):
    def __init__(self, ticker: str):
        super().__init__(f"Ticker not found: {ticker}")
        self.ticker = ticker
