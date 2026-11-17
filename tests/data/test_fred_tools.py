import pandas as pd
import pytest

from equity_trader.data import fred_tools as ft
from equity_trader.data.cache import reset_run_cache


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


def test_get_macro_snapshot(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    snap = ft.get_macro_snapshot()
    for k in ["dgs10", "dgs2", "t10y2y", "cpi_yoy", "unrate", "fedfunds", "nfci", "vix"]:
        assert k in snap
        assert snap[k] is None or isinstance(snap[k], (int, float))


def test_macro_snapshot_dates_every_reading_and_survives_a_missing_cpi_month(monkeypatch):
    class GappyFred(FakeFred):
        def get_series(self, series_id, observation_start=None):
            idx = pd.date_range("2025-01-01", periods=14, freq="MS")
            values = [100.0 + i for i in range(14)]
            values[1] = float("nan")  # the month twelve before the latest
            return pd.Series(values, index=idx, name=series_id)

    reset_run_cache()  # earlier tests cached these series ids
    monkeypatch.setattr(ft, "Fred", GappyFred)
    snap = ft.get_macro_snapshot()
    assert snap["cpi_yoy"] == pytest.approx((113.0 / 101.0 - 1.0) * 100.0)
    assert snap["as_of"]["dgs10"] == "2026-02-01"
    assert set(snap["as_of"]) == set(snap) - {"as_of"}
