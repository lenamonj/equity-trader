"""Grading saved verdicts against what the stock did next."""
import json

import pandas as pd
import pytest

from equity_trader import grading
from equity_trader.exceptions import DataFetchError

DAYS = pd.bdate_range("2026-01-05", periods=200)


def closes() -> pd.DataFrame:
    """ACME gains 0.1% a session and SPY 0.05%, so every forward return is known."""
    steps = pd.Series(range(len(DAYS)), index=DAYS)
    return pd.DataFrame({"ACME": 100 * 1.001 ** steps, "SPY": 100 * 1.0005 ** steps})


STOCK_3M, SPY_3M = 1.001 ** 63 - 1, 1.0005 ** 63 - 1      # 6.50% and 3.20%
STOCK_6M = 1.001 ** 126 - 1                                # 13.42%


def run(timestamp="2026-02-10T21:30:00Z", call="BUY", desks=(), ticker="ACME") -> dict:
    return {"ticker": ticker, "run_timestamp": timestamp, "final_recommendation": call,
            "conviction": 7,
            "agent_verdicts": [{"agent": name, "recommendation": rec, "conviction": conv,
                                "error_note": err} for name, rec, conv, err in desks]}


def graded(runs, horizon="3m", who="committee") -> pd.DataFrame:
    rows = grading.grade(runs, closes())
    return rows[(rows["horizon"] == horizon) & (rows["who"] == who)].reset_index(drop=True)


def test_runs_are_read_oldest_first_whatever_their_age_or_encoding(tmp_path):
    new = run("2026-11-22T18:05:11.196933Z")
    old = {**run("2026-06-07T20:44:30.931630"), "weight_overrides_rationale": None}
    (tmp_path / "ACME_2026-11-22_180511.json").write_text(json.dumps(new), encoding="utf-8")
    # Early runs were saved in the Windows default encoding.
    (tmp_path / "ACME_2026-06-07_2044.json").write_bytes(
        json.dumps({**old, "synthesis": "a – b"}, ensure_ascii=False).encode("cp1252"))
    runs = grading.load_runs(tmp_path)
    assert [r["run_timestamp"][:10] for r in runs] == ["2026-06-07", "2026-11-22"]


@pytest.mark.parametrize("timestamp, entry", [
    ("2026-02-10T21:30:00Z", "2026-02-10"),        # 4:30pm in New York: that day's close
    ("2026-02-10T20:44:30", "2026-02-09"),         # 3:44pm, no time zone given: the close before
    ("2026-02-15T15:00:00Z", "2026-02-13"),        # a Sunday: Friday's close
])
def test_a_run_is_entered_at_the_last_close_before_it(timestamp, entry):
    row = graded([run(timestamp)]).iloc[0]
    assert row["entry"] == pd.Timestamp(entry)
    assert row["stock_return"] == pytest.approx(STOCK_3M)


def test_a_call_is_graded_on_the_stocks_return_and_shown_against_the_market():
    buy = graded([run(call="BUY")]).iloc[0]
    assert buy["exit"] == DAYS[DAYS.get_loc(pd.Timestamp("2026-02-10")) + 63]
    assert (buy["benchmark_return"], buy["excess_return"]) == pytest.approx(
        (SPY_3M, STOCK_3M - SPY_3M))
    assert (buy["signed_return"], buy["signed_excess"]) == pytest.approx(
        (STOCK_3M, STOCK_3M - SPY_3M))
    assert buy["hit"] is True
    sell = graded([run(call="SELL")]).iloc[0]
    assert sell["signed_return"] == pytest.approx(-STOCK_3M) and sell["hit"] is False
    hold = graded([run(call="HOLD")]).iloc[0]
    assert hold["signed_return"] == 0.0 and hold["hit"] is None


def test_six_months_is_graded_over_126_sessions():
    assert graded([run()], horizon="6m").iloc[0]["stock_return"] == pytest.approx(STOCK_6M)


def test_a_horizon_that_has_not_elapsed_is_pending():
    late = run("2026-08-03T21:30:00Z")              # 50 sessions before the data ends
    row = graded([late]).iloc[0]
    assert pd.isna(row["stock_return"]) and pd.isna(row["exit"]) and row["hit"] is None


def test_each_desk_that_reported_is_graded_and_a_failed_desk_is_not():
    runs = [run(call="HOLD", desks=[("jpm_fundamental", "BUY", 8, None),
                                     ("gs_technical", "SELL", 6, None),
                                     ("citadel_quant", "HOLD", 1, "RuntimeError: boom")])]
    rows = grading.grade(runs, closes())
    rows = rows[rows["horizon"] == "3m"].set_index("who")
    assert list(rows.index) == ["committee", "jpm_fundamental", "gs_technical"]
    assert rows.loc["jpm_fundamental", "hit"] is True
    assert rows.loc["gs_technical", "hit"] is False
    assert rows.loc["gs_technical", "conviction"] == 6


def test_the_summary_gives_hit_rate_and_average_return_per_caller():
    runs = [run("2026-02-10T21:30:00Z", "BUY", [("jpm_fundamental", "SELL", 8, None)]),
            run("2026-02-10T22:30:00Z", "BUY", [("jpm_fundamental", "BUY", 8, None)]),
            run("2026-03-10T21:30:00Z", "HOLD", [("jpm_fundamental", "BUY", 5, None)]),
            run("2026-08-03T21:30:00Z", "SELL", [("jpm_fundamental", "SELL", 5, None)])]
    summary = grading.summarize(grading.grade(runs, closes()))
    three = summary[summary["horizon"] == "3m"].set_index("who")
    committee, morgan = three.loc["committee"], three.loc["jpm_fundamental"]
    # Three runs are old enough; the August run is pending.
    assert (committee["graded"], committee["pending"], committee["ticker_days"]) == (3, 1, 2)
    assert (committee["directional"], committee["hits"]) == (2, 2)
    assert committee["hit_rate"] == 1.0
    assert committee["avg_signed_return"] == pytest.approx(2 * STOCK_3M / 3)
    assert (morgan["directional"], morgan["hits"]) == (3, 2)
    assert morgan["hit_rate"] == pytest.approx(2 / 3)
    assert morgan["avg_signed_excess"] == pytest.approx((STOCK_3M - SPY_3M) / 3)


def test_a_caller_with_no_buy_or_sell_has_no_hit_rate():
    summary = grading.summarize(grading.grade([run(call="HOLD")], closes()))
    assert pd.isna(summary[summary["horizon"] == "3m"].iloc[0]["hit_rate"])


def test_a_ticker_with_no_prices_is_an_error():
    with pytest.raises(DataFetchError, match="GONE"):
        grading.grade([run(ticker="GONE")], closes().assign(GONE=float("nan")))
