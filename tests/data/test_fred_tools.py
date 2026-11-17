from datetime import date, timedelta

import httpx
import pandas as pd
import pytest
import respx

from equity_trader.data import fred_tools as ft


class FakeFred:
    def __init__(self, *_, **__):
        pass

    def get_series(self, series_id, observation_start=None):
        idx = pd.date_range("2024-01-01", periods=10, freq="MS")
        return pd.Series([1.0 + i for i in range(10)], index=idx, name=series_id)


def test_get_series(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    s = ft.get_series("DGS10")
    assert s.iloc[-1] == 10.0
    assert s.name == "DGS10"


def _local_file(tmp_path, monkeypatch):
    """A local history file: DGS10 monthly through 2023, one row overlapping FRED."""
    dates = pd.to_datetime(["2022-06-01", "2023-06-01", "2023-12-01", "2024-01-01"])
    frame = pd.DataFrame({"date": dates, "series_id": "DGS10",
                          "value": [-3.0, -2.0, -1.0, 99.0], "source": "test"})
    path = tmp_path / "history.parquet"
    frame.to_parquet(path)
    monkeypatch.setenv("ICE_HISTORY_FILE", str(path))


def test_local_history_is_joined_in_front_of_fred(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    _local_file(tmp_path, monkeypatch)
    s = ft.get_series("DGS10", start="2023-01-01")
    # From `start` on, and FRED's own value wins where the two overlap.
    assert list(s.iloc[:3]) == [-2.0, -1.0, 1.0]
    assert s.index.is_monotonic_increasing and s.index.is_unique and s.name == "DGS10"
    assert len(ft.get_series("UNRATE")) == 10  # a series the file does not hold


def test_dashboard_readings_use_local_history(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    _local_file(tmp_path, monkeypatch)
    row = ft._readings([("DGS10", "10-year Treasury yield", "level")])[0]
    assert row["pctile_years"] == 2.3  # 2022-06 to 2024-10, not FRED's ten months


def test_a_missing_local_history_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    monkeypatch.setenv("ICE_HISTORY_FILE", str(tmp_path / "absent.parquet"))
    with pytest.raises(FileNotFoundError):
        ft.get_series("DGS10")
    with pytest.raises(FileNotFoundError):
        ft._readings([("DGS10", "10-year Treasury yield", "level")])


def _monthly(values, end="2026-10-01"):
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values), freq="MS"))


def test_reading_shows_level_direction_and_percentile():
    series = _monthly([float(v) for v in range(1, 25)])  # 1..24, rising
    row = ft._reading("DGS10", "10-year Treasury yield", "level", series)
    assert (row["value"], row["as_of"]) == (24.0, "2026-10-01")
    assert (row["three_months_ago"], row["year_ago"]) == (21.0, 12.0)
    assert row["pctile"] == 100
    assert row["label"] == "10-year Treasury yield"


def test_yoy_reading_survives_a_month_that_was_never_published():
    values = [100.0 + i for i in range(14)]
    values[1] = float("nan")  # the month twelve before the latest
    row = ft._reading("CPIAUCSL", "CPI", "yoy", _monthly(values))
    assert row["label"] == "CPI (YoY %)"
    assert row["value"] == pytest.approx((113.0 / 101.0 - 1.0) * 100.0, abs=1e-3)


def test_reading_of_an_unavailable_series_is_explicitly_empty():
    assert ft._reading("X", "Missing", "level", None) == {
        "series": "X", "label": "Missing", "value": None, "as_of": None}


def test_macro_dashboard_covers_every_group_and_lists_releases(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    monkeypatch.setattr(ft, "get_release_calendar",
                        lambda: [{"date": "2026-12-10", "release": "CPI"}])
    out = ft.get_macro_dashboard()
    assert set(out["groups"]) == set(ft.DASHBOARD)
    assert [r["series"] for r in out["groups"]["credit"]] == [
        s for s, _, _ in ft.DASHBOARD["credit"]]
    assert out["groups"]["rates"][0]["as_of"] == "2024-10-01"
    assert out["upcoming_releases"] == [{"date": "2026-12-10", "release": "CPI"}]


def test_macro_dashboard_is_fetched_once_a_day_when_complete(monkeypatch):
    calls = []

    class CountingFred(FakeFred):
        def get_series(self, series_id, observation_start=None):
            calls.append(series_id)
            idx = pd.date_range("2022-01-01", periods=40, freq="MS")
            return pd.Series([1.0 + i for i in range(40)], index=idx)

    monkeypatch.setattr(ft, "Fred", CountingFred)
    monkeypatch.setattr(ft, "get_release_calendar", lambda: [])
    first = ft.get_macro_dashboard()
    fetched = len(calls)
    assert ft.get_macro_dashboard() is first and len(calls) == fetched


def test_incomplete_macro_dashboard_is_not_kept(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)  # too short for year-over-year readings
    monkeypatch.setattr(ft, "get_release_calendar", lambda: [])
    assert ft.get_macro_dashboard() is not ft.get_macro_dashboard()


def test_sector_macro_uses_yfinance_sector_names(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    assert set(ft.SECTOR_SERIES) == {
        "Technology", "Financial Services", "Energy", "Consumer Cyclical", "Consumer Defensive",
        "Healthcare", "Industrials", "Basic Materials", "Real Estate", "Utilities",
        "Communication Services"}
    out = ft.get_sector_macro("Financial Services")
    assert [r["series"] for r in out["readings"]][:2] == ["T10Y2Y", "DRTSCILM"]
    assert ft.get_sector_macro(None) == {"sector": None, "readings": []}


@respx.mock
def test_release_calendar_keeps_only_dates_inside_the_horizon():
    today = date.today()
    inside, outside = today + timedelta(days=20), today + timedelta(days=400)
    past = today - timedelta(days=5)
    respx.get("https://api.stlouisfed.org/fred/release/dates").mock(
        return_value=httpx.Response(200, json={"release_dates": [
            {"date": outside.isoformat()}, {"date": inside.isoformat()},
            {"date": past.isoformat()}]}))
    out = ft.get_release_calendar()
    assert {r["date"] for r in out} == {inside.isoformat()}
    assert {r["release"] for r in out} == set(ft._RELEASES.values())


def test_fred_series_describes_and_reads_any_series(monkeypatch):
    class DescribedFred(FakeFred):
        def get_series_info(self, series_id):
            return {"title": "Some Series", "units_short": "Index", "frequency_short": "M"}

    monkeypatch.setattr(ft, "Fred", DescribedFred)
    out = ft.get_fred_series(" abc ")
    assert (out["series"], out["label"], out["units"]) == ("ABC", "Some Series", "Index")
    assert out["value"] == 10.0 and len(out["recent"]) == 10


def test_search_fred_returns_ids_and_titles(monkeypatch):
    class SearchFred(FakeFred):
        def search(self, text, **kwargs):
            return pd.DataFrame(
                {"title": ["Semis"], "frequency_short": ["M"], "units_short": ["Index"],
                 "observation_end": [pd.Timestamp("2026-10-01")]}, index=["IPG3344S"])

    monkeypatch.setattr(ft, "Fred", SearchFred)
    assert ft.search_fred("semiconductor") == [{
        "series": "IPG3344S", "title": "Semis", "frequency": "M", "units": "Index",
        "last_observation": "2026-10-01"}]
