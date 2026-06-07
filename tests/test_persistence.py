import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pytest
from equity_trader.schemas import OrchestratorVerdict, AgentVerdict
from equity_trader.persistence import persist


def _verdict():
    av = AgentVerdict(agent="jpm_fundamental", ticker="NVDA",
                       recommendation="BUY", conviction=8,
                       thesis=["x"], risks=["y"], data_cited=["z"])
    return OrchestratorVerdict(
        ticker="NVDA", run_timestamp=datetime(2026, 6, 7, 12, 0, 0),
        current_price=190.0, final_recommendation="BUY", conviction=7,
        price_target_6mo=210.0, weighted_score=3.5,
        weights_used={"jpm_fundamental": 100.0}, weight_overrides_rationale=None,
        synthesis="A test synthesis sufficiently long to meet the minimum length.",
        key_agreements=[], key_disagreements=[], dominant_drivers=[],
        red_flags=[], position_sizing_suggestion="starter (~1%)",
        stop_loss_level=182.0, catalyst_calendar=[], agent_verdicts=[av],
    )


def test_persist_writes_sqlite_and_json(tmp_path):
    db = tmp_path / "et.db"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    v = _verdict()
    run_id = persist(v, db_path=str(db), runs_dir=str(runs_dir))
    assert run_id

    con = sqlite3.connect(db)
    row = con.execute("SELECT ticker, final_recommendation, snapshot_path FROM runs").fetchone()
    assert row[0] == "NVDA"
    assert row[1] == "BUY"
    snap = Path(row[2])
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["ticker"] == "NVDA"
