import os

import pytest

from equity_trader.data import fred_tools
from equity_trader.data.cache import reset_run_cache

# Set env vars at module level so they are present when agent modules are
# imported during collection (build_agent runs at module import time).
_TEST_ENV = {
    "OPENAI_API_KEY": "test",
    "GROQ_API_KEY": "test",
    "DEEPSEEK_API_KEY": "test",
    "GOOGLE_API_KEY": "test",
    "FRED_API_KEY": "test",
    "SEC_EDGAR_USER_AGENT_EMAIL": "test@example.com",
}
for _k, _v in _TEST_ENV.items():
    os.environ.setdefault(_k, _v)


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    for k, v in _TEST_ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _fresh_run_cache():
    reset_run_cache()
    fred_tools._DASHBOARD_BY_DAY.clear()
