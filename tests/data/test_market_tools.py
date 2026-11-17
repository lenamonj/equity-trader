from datetime import date

import numpy as np
import pandas as pd
import pytest

from equity_trader.data import market_tools as mt
from equity_trader.exceptions import DataFetchError


def _closes(columns: dict, periods: int = 200) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.bdate_range(end="2026-11-17", periods=periods))


def test_sector_rotation_orders_sectors_by_excess_return(monkeypatch):
    slopes = {etf: 1.0 for etf in mt.yft.SECTOR_ETF.values()}
    slopes.update({"XLK": 1.5, "XLE": 0.8, "SPY": 1.1, "RSP": 1.05})
    closes = _closes({s: np.linspace(100, 100 * k, 200) for s, k in slopes.items()})
    monkeypatch.setattr(mt.yft, "get_closes", lambda symbols, period: closes[symbols])

    out = mt.sector_rotation()

    sectors = list(out["sectors"])
    assert sectors[0] == "Technology" and sectors[-1] == "Energy"
    assert len(sectors) == 11 and "S&P 500" not in out["sectors"]
    tech = out["sectors"]["Technology"]
    assert tech["excess_3m"] == pytest.approx(tech["return_3m"] - out["market"]["return_3m"],
                                              abs=1e-4)
    assert out["breadth_equal_weight_minus_cap_weight"]["3m"] < 0


def test_boards_report_missing_history_as_none(monkeypatch):
    closes = _closes({s: np.linspace(100, 110, 30) for s in mt.CROSS_ASSET.values()}, periods=30)
    monkeypatch.setattr(mt.yft, "get_closes", lambda symbols, period: closes[symbols])
    gold = mt.cross_asset_board()["assets"]["Gold"]
    assert gold["return_1m"] is not None and gold["return_3m"] is None


def test_index_event_calendar_lists_third_fridays_in_the_horizon():
    events = mt.index_event_calendar(date(2026, 11, 17), horizon_days=180)
    assert events[0] == {"date": "2026-11-20", "event": "Monthly options expiration"}
    quarterly = [e["date"] for e in events if "rebalance" in e["event"]]
    assert quarterly == ["2026-12-18", "2027-03-19"]
    assert all(date.fromisoformat(e["date"]).weekday() == 4 for e in events)
    assert events[-1]["date"] == "2027-04-16"


def test_worst_drawdowns_measure_peak_to_trough_and_recovery():
    close = pd.Series([100.0, 120.0, 60.0, 130.0, 117.0, 140.0],
                      index=pd.bdate_range("2026-01-05", periods=6))
    worst = mt._worst_drawdowns(close)
    assert worst[0] == {"peak": "2026-01-06", "trough": "2026-01-07", "depth": -0.5,
                        "recovered": "2026-01-08"}
    assert worst[1]["depth"] == -0.1 and worst[1]["recovered"] == "2026-01-12"
    assert len(worst) == 2


def test_stress_history_marks_episodes_before_the_stock_traded(monkeypatch):
    def fake_history(ticker):
        start = "2005-01-03" if ticker == "SPY" else "2019-06-03"
        index = pd.bdate_range(start, "2023-12-29")
        level = np.where(index < "2020-03-01", 100.0, 70.0 if ticker == "SPY" else 50.0)
        return pd.DataFrame({"Close": level}, index=index)

    monkeypatch.setattr(mt, "history", fake_history)

    out = mt.stress_history("ACME")

    assert out["market_drawdowns"]["2008 financial crisis"] == {"traded": False}
    assert out["market_drawdowns"]["2020 pandemic crash"] == {
        "traded": True, "stock_return": -0.5, "stock_worst_point": -0.5, "market_return": -0.3}
    assert out["current_drawdown_from_all_time_high"] == -0.5
    assert out["worst_drawdowns_of_the_stock"][0]["recovered"] is None


def test_macro_sensitivity_reports_the_move_per_unit_of_each_factor(monkeypatch):
    rng = np.random.default_rng(0)
    market = 100 * np.cumprod(1 + rng.normal(0, 0.01, 600))
    closes = _closes({"ACME": market ** 2 / 100, "SPY": market}, periods=600)
    monkeypatch.setattr(mt.yft, "get_closes", lambda symbols, period: closes[symbols])

    def no_fred(series_id, start=None):
        raise DataFetchError("offline")

    monkeypatch.setattr(mt.fred_tools, "get_series", no_fred)

    out = mt.macro_sensitivity("acme")

    assert list(out["factors"]) == ["S&P 500"]  # unavailable factors are left out, and logged
    market_factor = out["factors"]["S&P 500"]
    assert market_factor["correlation"] == 1.0
    assert market_factor["stock_move"] == pytest.approx(0.02, abs=0.001)


def test_etf_exposure_reports_weight_rank_and_liquidity(monkeypatch):
    info = {"ACME": {"sector": "Technology", "sectorKey": "technology", "floatShares": 1e9},
            "SPY": {"shortName": "S&P 500 ETF", "totalAssets": 5e11, "volume": 50,
                    "averageVolume": 100}}
    holdings = {"SPY": pd.DataFrame({"Holding Percent": [0.07, 0.05]}, index=["AAPL", "ACME"]),
                "XLK": pd.DataFrame({"Holding Percent": [0.15]}, index=["AAPL"])}

    class FakeFund:
        def __init__(self, symbol):
            if symbol not in holdings:
                raise KeyError(symbol)
            self.funds_data = type("F", (), {"top_holdings": holdings[symbol]})()

    class FakeSector:
        top_etfs = {"XLK": "Tech", "SMH": "Semis"}

    prices = pd.DataFrame({"Close": 100.0, "Volume": 1e6},
                          index=pd.bdate_range(end="2026-11-17", periods=80))
    prices.iloc[-1, 0] = 101.0
    monkeypatch.setattr(mt.yft, "get_info", lambda s: info.get(s, {}))
    monkeypatch.setattr(mt.yf, "Ticker", FakeFund)
    monkeypatch.setattr(mt.yf, "Sector", lambda key: FakeSector())
    monkeypatch.setattr(mt.yft, "get_price_history", lambda t, period, interval: prices)

    out = mt.etf_exposure("acme")

    by_etf = {row["etf"]: row for row in out["etfs"]}
    assert set(by_etf) == {"SPY", "XLK"}  # funds whose holdings cannot be read are skipped
    assert by_etf["SPY"]["weight"] == 0.05 and by_etf["SPY"]["rank_in_fund"] == 2
    assert by_etf["SPY"]["volume_vs_average"] == 0.5
    assert by_etf["XLK"]["weight"] is None and by_etf["XLK"]["rank_in_fund"] is None
    assert out["liquidity"]["avg_daily_dollar_volume_60d"] == pytest.approx(1e8, rel=0.01)
    assert out["liquidity"]["daily_turnover_of_float"] == 0.001
