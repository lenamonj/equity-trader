import pytest
import pandas as pd
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


def test_get_macro_snapshot(monkeypatch):
    monkeypatch.setattr(ft, "Fred", FakeFred)
    snap = ft.get_macro_snapshot()
    for k in ["dgs10", "dgs2", "t10y2y", "cpi_yoy", "unrate", "fedfunds", "nfci", "vix"]:
        assert k in snap
        assert snap[k] is None or isinstance(snap[k], (int, float))
