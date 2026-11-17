import pytest
from equity_trader.schemas import AgentVerdict
from equity_trader.scoring import compute_weighted_score, redistribute_weights


def _v(agent, rec, conv=5, err=None):
    return AgentVerdict(agent=agent, ticker="T", recommendation=rec,
                        conviction=conv, thesis=["x"], risks=["y"],
                        data_cited=["z"], error_note=err)


def test_all_buy_unanimous():
    verdicts = [_v("jpm_fundamental", "BUY", 8), _v("gs_technical", "BUY", 7)]
    weights = {"jpm_fundamental": 60.0, "gs_technical": 40.0}
    score = compute_weighted_score(verdicts, weights)
    # 1*8*60 + 1*7*40 = 480 + 280 = 760; /100 = 7.6
    assert abs(score - 7.6) < 1e-6


def test_split_calls():
    verdicts = [_v("a", "BUY", 8), _v("b", "SELL", 6)]
    weights = {"a": 50.0, "b": 50.0}
    score = compute_weighted_score(verdicts, weights)
    # (1*8*50 + (-1)*6*50)/100 = (400-300)/100 = 1.0
    assert abs(score - 1.0) < 1e-6


def test_hold_contributes_zero():
    verdicts = [_v("a", "HOLD", 9)]
    score = compute_weighted_score(verdicts, {"a": 100.0})
    assert score == 0.0


def test_errored_agent_excluded_and_weights_redistributed():
    verdicts = [_v("a", "BUY", 8), _v("b", "BUY", 6, err="boom")]
    weights = {"a": 50.0, "b": 50.0}
    score = compute_weighted_score(verdicts, weights)
    # 'b' excluded; 'a' gets 100% weight: 1*8*100/100 = 8.0
    assert abs(score - 8.0) < 1e-6


def test_redistribute_weights():
    out = redistribute_weights({"a": 50.0, "b": 30.0, "c": 20.0}, excluded={"b"})
    # remaining: a=50, c=20; total=70; rescale to 100
    assert abs(out["a"] - 50.0 * 100.0 / 70.0) < 1e-6
    assert abs(out["c"] - 20.0 * 100.0 / 70.0) < 1e-6
    assert "b" not in out


def test_conviction_weighted_target_excludes_missing_and_errored():
    from equity_trader.scoring import conviction_weighted_target
    from equity_trader.schemas import AgentVerdict

    def v(conv, tgt, err=None):
        return AgentVerdict(agent="a", ticker="X", recommendation="BUY",
                            conviction=conv, price_target_6mo=tgt, error_note=err)

    verdicts = [v(8, 200.0), v(2, 100.0), v(5, None), v(9, 999.0, err="boom")]
    assert conviction_weighted_target(verdicts) == 180.0
    assert conviction_weighted_target([v(5, None)]) is None
