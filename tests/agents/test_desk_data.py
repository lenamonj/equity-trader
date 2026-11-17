import importlib
import inspect
import re
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from equity_trader.agents import ALL_AGENTS, ALL_DATA
from equity_trader.agents.shared_data import SHARED_DATA
from equity_trader.schemas import desk_report_for
from tests.agents import gathered_for

SHARED = {section.__name__ for section in SHARED_DATA}
EVERY_SECTION = {section.__name__ for section in ALL_DATA}

# The sections that make each desk what it is.
DESK_DATA = {
    "jpm_fundamental": {"fundamentals", "earnings_release", "estimates", "analyst_view",
                        "insider_activity", "recent_8ks"},
    "bridgewater_macro": {"macro_sensitivity", "stress_history", "cross_asset_board",
                          "sector_rotation"},
    "gs_technical": {"technicals", "key_levels", "volume_analysis", "relative_strength",
                     "price_history"},
    "citadel_quant": {"factor_scorecard", "estimates", "style_board", "risk_stats"},
    "jane_street_etf": {"etf_landscape", "etf_exposure", "relative_strength",
                        "sector_rotation", "index_event_calendar"},
    "de_shaw_options": {"options_surface", "volatility_premium", "market_volatility",
                        "options_chain", "index_event_calendar"},
    "renaissance_pattern": {"seasonality", "earnings_event_study", "forward_return_study",
                            "volatility_and_gaps"},
    "two_sigma_backtest": {"rule_backtest"},
}
DESKS = [name for name, _ in ALL_AGENTS]


def _module(name):
    return importlib.import_module(f"equity_trader.agents.{name}")


@pytest.mark.parametrize("name", DESKS)
def test_every_desk_reads_the_shared_context_and_its_own_sections(name):
    names = [section.__name__ for section in _module(name).DATA]
    assert len(names) == len(set(names))
    assert set(names) == SHARED | DESK_DATA[name]


@pytest.mark.parametrize("name", DESKS)
def test_a_prompt_names_only_sections_the_desk_is_given(name):
    module = _module(name)
    given = {section.__name__ for section in module.DATA}
    named = set(re.findall(r"`([a-z0-9_]+)`", module.INSTRUCTIONS)) & EVERY_SECTION
    assert named <= given, named - given


@pytest.mark.parametrize("name", DESKS)
def test_a_desk_has_no_tools(name):
    assert _module(name).agent.tools == []


@pytest.mark.parametrize("name", DESKS)
def test_a_desks_schema_is_enforced_and_can_only_cite_its_own_sections(name):
    module = _module(name)
    output = module.agent.output_type
    assert output.is_strict_json_schema()
    schema = output.json_schema()
    assert list(schema["properties"])[0] == "evidence"  # evidence first, the call after
    assert list(schema["properties"]).index("thesis") < list(schema["properties"]).index(
        "recommendation")
    assert schema["$defs"]["Evidence"]["properties"]["source"]["enum"] == [
        section.__name__ for section in module.DATA]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", DESKS)
async def test_a_desk_sends_its_brief_and_returns_the_models_report(name):
    module = _module(name)
    sections = [section.__name__ for section in module.DATA]
    report = desk_report_for(sections)(
        evidence=[{"metric": "Price", "value": 153.82, "unit": "usd",
                   "as_of": date(2026, 11, 17), "source": "company_snapshot",
                   "signal": "neutral"}],
        thesis=["finding"], risks=["risk"], recommendation="HOLD", conviction=5,
        price_target_6mo=None)
    with patch.object(module, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": report})())
        out = await module.run("NVDA", gathered_for(module))
    assert out is report
    sent = RunnerCls.run.await_args.kwargs["input"]
    assert sent.startswith("Analyze NVDA") and all(f"## {s}" in sent for s in sections)


def test_the_gatherer_fetches_33_sections_shared_context_first():
    names = [section.__name__ for section in ALL_DATA]
    assert len(names) == len(set(names)) == 33
    assert names[:3] == ["company_snapshot", "macro_dashboard", "sector_macro"]


def test_every_section_describes_itself_and_takes_the_ticker_or_nothing():
    # A section with any other argument could not be gathered ahead of the desks.
    for section in ALL_DATA:
        assert len(" ".join(section.__doc__.split())) > 40, section.__name__
        assert list(inspect.signature(section).parameters) in ([], ["ticker"]), section.__name__
