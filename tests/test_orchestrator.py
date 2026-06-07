import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict, CatalystEvent, OrchestratorVerdict
from equity_trader import orchestrator as orch


def _v(agent, rec, conv=7, tgt=200.0, err=None):
    return AgentVerdict(agent=agent, ticker="NVDA", recommendation=rec,
                        conviction=conv, price_target_6mo=tgt,
                        thesis=["t"], risks=["r"], data_cited=["d"],
                        error_note=err)


@pytest.fixture(autouse=True)
def _no_real_calendar(monkeypatch):
    # Default: orchestrator tests do NOT hit yfinance for the calendar.
    # Individual tests can override by setting orch.get_real_catalysts again.
    monkeypatch.setattr(orch, "get_real_catalysts", lambda ticker, today: [])


@pytest.mark.asyncio
async def test_orchestrator_run_returns_orchestrator_verdict():
    verdicts = [
        _v("jpm_fundamental", "BUY", 8, 215),
        _v("bridgewater_macro", "HOLD", 6, None),
        _v("gs_technical", "BUY", 7, 208),
        _v("citadel_quant", "BUY", 6, 205),
        _v("renaissance_pattern", "HOLD", 5, None),
        _v("de_shaw_options", "BUY", 6, 210),
        _v("two_sigma_backtest", "HOLD", 4, None),
    ]
    fake = OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime.utcnow(), current_price=190.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=210.0,
        weighted_score=3.55, weights_used={"jpm_fundamental": 25.0},
        weight_overrides_rationale=None,
        synthesis="Strong fundamentals confirmed by technicals; macro acceptable for horizon.",
        key_agreements=["FCF and technicals align"],
        key_disagreements=["Macro vs fundamental on rate path"],
        dominant_drivers=["jpm_fundamental", "gs_technical"],
        red_flags=["Customer concentration"],
        position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[], agent_verdicts=verdicts,
    )
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": fake})())
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert out.final_recommendation == "BUY"


@pytest.mark.asyncio
async def test_orchestrator_stamps_deterministic_fields_over_llm_hallucination():
    # LLM returns a verdict with garbage timestamp/price/score/verdicts.
    # The orchestrator must overwrite those with the runner-supplied truth.
    real_verdicts = [_v("jpm_fundamental", "BUY", 8, 215)]
    hallucinated = OrchestratorVerdict(
        ticker="WRONG", run_timestamp=datetime(2020, 1, 1),
        current_price=999.0, final_recommendation="BUY", conviction=7,
        price_target_6mo=210.0, weighted_score=99.9,
        weights_used={"made_up_agent": 100.0}, weight_overrides_rationale=None,
        synthesis="A synthesis sufficiently long to clear the schema minimum length.",
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=None, catalyst_calendar=[],
        agent_verdicts=[_v("fabricated_agent", "BUY", 9)],
    )
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": hallucinated})())
        out = await orch.run("NVDA", real_verdicts, current_price=190.0)
    assert out.ticker == "NVDA"
    assert out.current_price == 190.0
    assert out.run_timestamp.year >= 2026
    assert out.agent_verdicts == real_verdicts
    # weighted_score must be the deterministic value, not 99.9
    assert out.weighted_score != 99.9
    # The judgment fields (final_recommendation, synthesis, etc.) survive
    assert out.final_recommendation == "BUY"
    assert "synthesis sufficiently long" in out.synthesis


@pytest.mark.asyncio
async def test_orchestrator_drops_out_of_window_catalysts(monkeypatch):
    # The LLM occasionally fabricates plausible-looking but past-dated earnings
    # (e.g., "Q4 2023" when today is 2026). Out-of-window entries are dropped.
    today = date.today()
    past = today - timedelta(days=365)
    future_in = today + timedelta(days=60)
    future_out = today + timedelta(days=365)

    monkeypatch.setattr(orch, "get_real_catalysts",
                         lambda ticker, t: [])

    hallucinated = OrchestratorVerdict(
        ticker="AVGO", run_timestamp=datetime.utcnow(), current_price=300.0,
        final_recommendation="HOLD", conviction=5, price_target_6mo=320.0,
        weighted_score=2.0, weights_used={"jpm_fundamental": 100.0},
        weight_overrides_rationale=None,
        synthesis="A synthesis sufficiently long to clear the schema minimum length.",
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=None,
        catalyst_calendar=[
            CatalystEvent(event="Q4 2023 Earnings", date=past,
                          expected_impact="HIGH"),
            CatalystEvent(event="Q3 2026 Earnings", date=future_in,
                          expected_impact="HIGH"),
            CatalystEvent(event="FY2027 Guidance Day", date=future_out,
                          expected_impact="MEDIUM"),
        ],
        agent_verdicts=[_v("jpm_fundamental", "BUY", 8)],
    )
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": hallucinated})())
        out = await orch.run("AVGO", [_v("jpm_fundamental", "BUY", 8)],
                              current_price=300.0)
    dates = [c.date for c in out.catalyst_calendar]
    assert past not in dates
    assert future_out not in dates
    assert future_in in dates


@pytest.mark.asyncio
async def test_orchestrator_injects_real_catalysts_even_if_llm_drops_them(monkeypatch):
    today = date.today()
    real_date = today + timedelta(days=90)
    monkeypatch.setattr(orch, "get_real_catalysts", lambda ticker, t: [
        {"event": "Earnings release", "date": real_date.isoformat(),
         "expected_impact": "HIGH", "notes": "Source: yfinance calendar"}
    ])

    no_calendar = OrchestratorVerdict(
        ticker="AVGO", run_timestamp=datetime.utcnow(), current_price=300.0,
        final_recommendation="BUY", conviction=7, price_target_6mo=320.0,
        weighted_score=4.0, weights_used={"jpm_fundamental": 100.0},
        weight_overrides_rationale=None,
        synthesis="A synthesis sufficiently long to clear the schema minimum length.",
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=290.0, catalyst_calendar=[],
        agent_verdicts=[_v("jpm_fundamental", "BUY", 8)],
    )
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(return_value=type("R", (), {"final_output": no_calendar})())
        out = await orch.run("AVGO", [_v("jpm_fundamental", "BUY", 8)],
                              current_price=300.0)
    assert any(c.date == real_date and c.event == "Earnings release"
               for c in out.catalyst_calendar)


@pytest.mark.asyncio
async def test_orchestrator_falls_back_on_llm_failure():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert "mechanical aggregation" in out.synthesis.lower()
