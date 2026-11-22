"""Live check that the statements handed to the fundamental desk agree with
Yahoo's own. Run with `uv run pytest -m slow`.

The statement code was first tuned on three technology companies, and broke
on others: wrong tags, stale facts, missing lines. This runs it across the
sectors, so a company that files differently is found here and not by a desk.

What is compared, and why not more:
- Trailing-twelve-month revenue against Yahoo's summary figure. Yahoo's
  quarterly table is not used for revenue: it restates banks' revenue to its
  own definition, and it differences an annual figure against quarters that
  were later restated (Occidental's fourth quarter of 2025).
- Net income for the latest quarter both sources have, against Yahoo's
  quarterly table. Yahoo's trailing net income figure is not used: it leaves
  out discontinued operations.
"""
from datetime import date

import pandas as pd
import pytest
import yfinance as yf

from equity_trader.agents.jpm_fundamental import get_fundamentals
from equity_trader.data.cache import reset_run_cache

# One or more large companies from each sector.
COMPANIES = ["AAPL", "NVDA", "AVGO", "ORCL", "TGT", "MS", "OXY", "LLY", "CAT", "NFLX", "HD",
             "NEE", "FCX", "PLD"]
# Fiscal quarters that end on a weekday near month end are dated a few days apart.
_SAME_QUARTER_DAYS = 7


@pytest.mark.slow
@pytest.mark.parametrize("ticker", COMPANIES)
def test_statements_agree_with_yahoo(ticker):
    reset_run_cache()
    ours = get_fundamentals(ticker)
    quarters = ours["quarterly"]
    latest = date.fromisoformat(quarters[0]["period_end"])
    assert (date.today() - latest).days < 200, f"latest quarter {latest} is stale"

    if ours["source"].startswith("yfinance"):
        # The fallback is Yahoo's own statements, so there is nothing independent
        # to compare. It is acceptable for one reason only: the SEC's facts feed
        # has not caught up with the company's latest report. A company that
        # files US-GAAP statements must never land here for want of a tag.
        assert "latest report covers" in ours["source"], ours["source"]
        return

    yahoo = yf.Ticker(ticker).quarterly_income_stmt
    # Yahoo takes a quarter from the earnings release, weeks before the report
    # that puts it in the SEC's facts. Its trailing figure is comparable only
    # when both end on the same quarter.
    ahead = (pd.Timestamp(yahoo.columns[0]) - pd.Timestamp(latest)).days > _SAME_QUARTER_DAYS
    if not ahead:
        trailing = sum(quarter["revenue"] for quarter in quarters[:4])
        assert trailing == pytest.approx(yf.Ticker(ticker).info["totalRevenue"], rel=0.03), (
            f"trailing revenue, {ticker}")

    for quarter in quarters[:2]:
        end = pd.Timestamp(quarter["period_end"])
        column = next((c for c in yahoo.columns
                       if abs((pd.Timestamp(c) - end).days) <= _SAME_QUARTER_DAYS), None)
        # Net income to the company, or to its common stockholders after preferred dividends.
        rows = [] if column is None else [
            yahoo.loc[row, column] for row in ("Net Income", "Net Income Common Stockholders")
            if row in yahoo.index and pd.notna(yahoo.loc[row, column])]
        if not rows:
            continue  # Yahoo has not filled this quarter in yet
        assert any(quarter["net_income"] == pytest.approx(value, rel=0.05, abs=5e7)
                   for value in rows), (
            f"net income {quarter['net_income']} vs {rows}, {ticker} {quarter['period_end']}")
        break
    else:
        pytest.fail(f"no quarter of {ticker} could be matched to Yahoo's")
