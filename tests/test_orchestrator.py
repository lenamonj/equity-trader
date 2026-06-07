import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from equity_trader.schemas import AgentVerdict, OrchestratorVerdict
from equity_trader import orchestrator as orch


def _v(agent, rec, conv=7, tgt=200.0, err=None):
    return AgentVerdict(agent=agent, ticker="NVDA", recommendation=rec,
                        conviction=conv, price_target_6mo=tgt,
                        thesis=["t"], risks=["r"], data_cited=["d"],
                        error_note=err)


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
async def test_orchestrator_falls_back_on_llm_failure():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert "mechanical aggregation" in out.synthesis.lower()
