class DataFetchError(Exception):
    """Base for data layer failures."""


class TickerNotFoundError(DataFetchError):
    def __init__(self, ticker: str):
        super().__init__(f"Ticker not found: {ticker}")
        self.ticker = ticker


class RateLimitError(DataFetchError):
    """Raised when an upstream provider rate-limits us."""


class AgentExecutionError(Exception):
    """Raised when an agent fails after retries; carries the agent name."""

    def __init__(self, agent: str, cause: Exception):
        super().__init__(f"Agent {agent} failed: {cause}")
        self.agent = agent
        self.cause = cause
