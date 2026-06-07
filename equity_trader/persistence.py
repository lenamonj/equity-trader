import json
import sqlite3
import uuid
from pathlib import Path
from equity_trader.schemas import OrchestratorVerdict


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    current_price REAL NOT NULL,
    final_recommendation TEXT NOT NULL,
    conviction INTEGER NOT NULL,
    price_target_6mo REAL,
    weighted_score REAL NOT NULL,
    snapshot_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_ticker_ts ON runs(ticker, run_timestamp DESC);
"""


def _ensure_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()


def persist(verdict: OrchestratorVerdict,
             db_path: str = "equity_trader.db",
             runs_dir: str = "runs") -> str:
    _ensure_schema(db_path)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    ts = verdict.run_timestamp.strftime("%Y-%m-%d_%H%M")
    snap_path = Path(runs_dir) / f"{verdict.ticker}_{ts}.json"
    snap_path.write_text(verdict.model_dump_json(indent=2))

    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs (run_id, ticker, run_timestamp, current_price, "
            "final_recommendation, conviction, price_target_6mo, weighted_score, "
            "snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, verdict.ticker, verdict.run_timestamp.isoformat(),
             verdict.current_price, verdict.final_recommendation,
             verdict.conviction, verdict.price_target_6mo, verdict.weighted_score,
             str(snap_path)),
        )
        con.commit()
    finally:
        con.close()
    return run_id


def list_runs(db_path: str = "equity_trader.db",
               ticker: str | None = None,
               limit: int = 50) -> list[dict]:
    _ensure_schema(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        if ticker:
            rows = con.execute(
                "SELECT * FROM runs WHERE ticker=? ORDER BY run_timestamp DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY run_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
