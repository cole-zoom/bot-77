"""Every simulator table goes through here (docs/simulator.md: tower_hp, sim_validation, sim_scenarios, sim_runs).
Rows get the engine's commit, so a number can always be traced to the engine that produced it."""
from __future__ import annotations

import subprocess
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import lancedb
import pyarrow as pa

DB = "data/lancedb"
ENGINE_DIR = Path(__file__).resolve().parents[3] / "third_party" / "clash-royale-env"


@lru_cache(maxsize=1)
def engine_commit() -> str:
    """The fork's commit, '+dirty' when its working tree has changes."""
    sha = subprocess.run(["git", "-C", str(ENGINE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ENGINE_DIR), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return sha + ("+dirty" if dirty else "")


def write(table: str, rows: list[dict], mode: str = "append", where: str | None = None) -> int:
    """Append (or overwrite) rows; `where` first deletes the rows it matches (re-running a run replaces it)."""
    if not rows:
        return 0
    db = lancedb.connect(DB)
    data = pa.Table.from_pylist(rows)
    if table in db.table_names() and mode == "append":
        t = db.open_table(table)
        if where:
            t.delete(where)
        t.add(data)
    else:
        t = db.create_table(table, data, mode="overwrite")
    t.optimize(cleanup_older_than=timedelta(0))  # one manifest per append otherwise (docs/training-architecture.md)
    return len(rows)


def read(table: str, where: str | None = None, limit: int = 10**8) -> list[dict]:
    q = lancedb.connect(DB).open_table(table).search()
    if where:
        q = q.where(where)
    return q.limit(limit).to_arrow().to_pylist()


def tower_fracs(match_id: str, t: float) -> dict[str, float]:
    """tower_hp fractions at video time t: the last accepted reading per tower (bottom-frame names)."""
    last = {}
    for r in sorted(read("tower_hp", f"match_id = '{match_id}' AND accepted = true AND t <= {t}"), key=lambda r: r["t"]):
        if r["frac"] is not None:
            last[r["tower"]] = r["frac"]
    return last
