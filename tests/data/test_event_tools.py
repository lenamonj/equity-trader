import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from equity_trader.data import event_tools as ev

NEWS = "https://news.google.com/rss/search"


def bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A flat stock and a flat sector ETF, 120 sessions to Friday 2026-11-20,
    with four sessions that stand out."""
    days = pd.bdate_range(end="2026-11-20", periods=120)
    stock = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
                          "Volume": 1e6}, index=days)
    etf = pd.DataFrame({"Open": 50.0, "High": 50.0, "Low": 50.0, "Close": 50.0,
                        "Volume": 1e6}, index=days)
    # Heavy volume, nothing else.
    stock.loc["2026-10-20", "Volume"] = 3e6
    # A small move the session after a filing accepted after the close.
    stock.loc["2026-11-05", ["Open", "Close"]] = [100.5, 101.0]
    stock.loc["2026-11-06":, ["Open", "Close"]] = 101.0
    # A gap down on heavy volume, on a filing accepted before the open.
    stock.loc["2026-11-12", ["Open", "Close", "Volume"]] = [87.87, 96.96, 10e6]
    stock.loc["2026-11-13":, ["Open", "Close"]] = 96.96
    etf.loc["2026-11-12":, "Close"] = 49.5
    # A fall on heavy volume with no filing, on a day the sector fell too.
    stock.loc["2026-11-18", ["Close", "Volume"]] = [87.264, 6e6]
    stock.loc["2026-11-19":, ["Open", "Close"]] = 87.264
    etf.loc["2026-11-18":, "Close"] = 46.53
    return stock, etf


RESIGNATION = {"filing_date": "2026-11-12", "accepted": "2026-11-12T13:52:41.000Z",
               "items": ["5.02 Departure or appointment of directors or officers"],
               "text": "Item 5.02. The Chief Executive Officer resigned. " + "x" * 600}
RESULTS = {"filing_date": "2026-11-04", "accepted": "2026-11-04T21:20:51.000Z",
           "items": ["2.02 Results of operations and financial condition"], "text": None}


def item(title: str, source: str = "Reuters") -> str:
    return (f"<item><title>{title} - {source}</title>"
            "<link>https://news.google.com/rss/articles/x</link>"
            "<pubDate>Thu, 12 Nov 2026 14:05:00 GMT</pubDate>"
            f'<source url="https://example.com">{source}</source></item>')


def feed(*items: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
            f"<title>Google News</title>{''.join(items)}</channel></rss>")


def use(monkeypatch, filings, stock=None):
    default_stock, etf = bars()
    stock = default_stock if stock is None else stock
    monkeypatch.setattr(ev.yft, "get_price_history",
                        lambda ticker, period="max", interval="1d":
                        etf if ticker == "XLK" else stock)
    monkeypatch.setattr(ev.yft, "get_info", lambda ticker: {
        "sector": "Technology", "longName": "Acme Corporation"})
    monkeypatch.setattr(ev.edgar_tools, "get_recent_8ks", lambda ticker: filings)
    monkeypatch.setattr(ev, "_NEWS_PAUSE", 0.0)


def events(monkeypatch, filings, stock=None) -> dict[str | None, dict]:
    use(monkeypatch, filings, stock)
    return {event["session"]: event for event in ev.get_recent_events("ACME")["events"]}


@respx.mock
def test_a_filing_is_joined_to_the_price_reaction_of_its_session(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    event = events(monkeypatch, [RESIGNATION, RESULTS])["2026-11-12"]
    assert event["filings"] == [{
        "filing_date": "2026-11-12",
        "items": ["5.02 Departure or appointment of directors or officers"],
        "excerpt": RESIGNATION["text"][:500]}]
    assert event["opening_gap"] == -0.13               # 87.87 against the prior close of 101
    assert event["close_to_close_return"] == -0.04     # 96.96 against 101
    assert event["volume_vs_50d_avg"] == 8.2           # 10m against a 50-day average of 1.22m
    assert event["sector_etf_return"] == -0.01
    assert event["return_vs_sector_etf"] == -0.03
    assert (event["heavy_volume"], event["unexplained"]) == (True, False)


@respx.mock
def test_a_filing_accepted_after_the_close_is_read_on_the_next_session(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    by_session = events(monkeypatch, [RESIGNATION, RESULTS])
    assert "2026-11-04" not in by_session
    event = by_session["2026-11-05"]
    assert event["filings"][0]["filing_date"] == "2026-11-04"
    assert event["filings"][0]["excerpt"] is None      # the filing's text was unavailable
    assert (event["opening_gap"], event["close_to_close_return"]) == (0.005, 0.01)
    assert event["heavy_volume"] is False


@respx.mock
def test_a_heavy_volume_day_with_no_filing_is_unexplained(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    event = events(monkeypatch, [RESIGNATION, RESULTS])["2026-11-18"]
    assert (event["filings"], event["heavy_volume"], event["unexplained"]) == ([], True, True)
    assert event["close_to_close_return"] == -0.1
    assert (event["sector_etf_return"], event["return_vs_sector_etf"]) == (-0.06, -0.04)


@respx.mock
def test_events_are_ranked_by_the_size_of_the_move_and_a_tenth_is_material(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    use(monkeypatch, [RESIGNATION, RESULTS])
    out = ev.get_recent_events("ACME")
    assert (out["as_of"], out["sector_etf"]) == ("2026-11-20", "XLK")
    assert [(e["rank"], e["session"], e["move"], e["material"]) for e in out["events"]] == [
        (1, "2026-11-12", 0.13, True),     # the gap, larger than the close-to-close move
        (2, "2026-11-18", 0.1, True),      # the stock's own move, though the sector fell 6%
        (3, "2026-11-05", 0.01, False),
        (4, "2026-10-20", 0.0, False)]


@respx.mock
def test_two_filings_read_on_one_session_are_one_event(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    same_day = {**RESULTS, "filing_date": "2026-11-12", "accepted": "2026-11-12T18:00:00.000Z"}
    event = events(monkeypatch, [RESIGNATION, same_day])["2026-11-12"]
    assert [f["items"][0][:4] for f in event["filings"]] == ["5.02", "2.02"]


@respx.mock
def test_a_filing_no_session_has_traded_on_yet_has_no_reaction(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    tonight = {**RESULTS, "filing_date": "2026-11-20", "accepted": "2026-11-20T21:30:00.000Z"}
    use(monkeypatch, [tonight])
    last = ev.get_recent_events("ACME")["events"][-1]
    assert (last["session"], last["filings"][0]["filing_date"]) == (None, "2026-11-20")
    assert (last["opening_gap"], last["move"], last["material"]) == (None, None, False)


@respx.mock
def test_headlines_are_asked_for_around_each_event(monkeypatch):
    route = respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed()))
    use(monkeypatch, [RESIGNATION, RESULTS])
    ev.get_recent_events("ACME")
    asked = {call.request.url.params["q"] for call in route.calls}
    assert asked == {
        "Acme stock after:2026-11-11 before:2026-11-13",
        "Acme stock after:2026-11-17 before:2026-11-19",
        "Acme stock after:2026-11-03 before:2026-11-06",     # from the filing to its session
        "Acme stock after:2026-10-19 before:2026-10-21"}


@respx.mock
def test_headlines_keep_up_to_six_titles_that_name_the_company(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(200, text=feed(
        item("Acme chief executive resigns", "Reuters"),
        item("Chip stocks slide as rates climb", "CNBC"),
        item("Acme chief executive resigns", "Reuters"),
        item("ACME shares fall 13% &amp; analysts cut targets", "Bloomberg"),
        item("Acmeology conference opens", "Wired"),
        item("Acme announces a new product", "Acme Newsroom"),
        *[item(f"Acme story {n}", "AP") for n in range(8)])))
    event = events(monkeypatch, [RESIGNATION])["2026-11-12"]
    assert event["headlines"][:2] == [
        {"title": "Acme chief executive resigns", "source": "Reuters",
         "published": "2026-11-12"},
        {"title": "ACME shares fall 13% & analysts cut targets", "source": "Bloomberg",
         "published": "2026-11-12"}]
    assert [h["title"] for h in event["headlines"][2:]] == [f"Acme story {n}"
                                                            for n in range(4)]


@respx.mock
def test_a_feed_that_cannot_be_read_leaves_headlines_null(monkeypatch):
    respx.get(NEWS).mock(return_value=httpx.Response(503))
    event = events(monkeypatch, [RESIGNATION])["2026-11-12"]
    assert event["headlines"] is None
    assert event["opening_gap"] == -0.13


@pytest.mark.parametrize("legal, plain", [
    ("NVIDIA Corporation", "NVIDIA"),
    ("Apple Inc.", "Apple"),
    ("Meta Platforms, Inc.", "Meta Platforms"),
    ("JPMorgan Chase & Co.", "JPMorgan Chase"),
    ("The Coca-Cola Company", "Coca-Cola"),
    ("Amazon.com, Inc.", "Amazon"),
    ("Shell plc", "Shell"),
    ("Eli Lilly and Company", "Eli Lilly"),
])
def test_the_company_is_searched_for_by_its_plain_name(legal, plain):
    assert ev.company_name(legal) == plain


def test_a_stock_with_no_sector_etf_has_no_sector_comparison(monkeypatch):
    use(monkeypatch, [])
    monkeypatch.setattr(ev.yft, "get_info", lambda ticker: {"longName": "Acme Corporation"})
    monkeypatch.setattr(ev, "_headlines", lambda company, start, end: [])
    out = ev.get_recent_events("ACME")
    top = out["events"][0]
    assert out["sector_etf"] is None
    assert (top["session"], top["sector_etf_return"], top["return_vs_sector_etf"]) == (
        "2026-11-12", None, None)
    assert top["move"] == 0.13
    assert np.isfinite(top["volume_vs_50d_avg"])
