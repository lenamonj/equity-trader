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
async def test_orchestrator_falls_back_on_llm_failure():
    verdicts = [_v("jpm_fundamental", "BUY", 8)]
    with patch.object(orch, "Runner") as RunnerCls:
        RunnerCls.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        out = await orch.run("NVDA", verdicts, current_price=190.0)
        assert isinstance(out, OrchestratorVerdict)
        assert "mechanical aggregation" in out.synthesis.lower()
