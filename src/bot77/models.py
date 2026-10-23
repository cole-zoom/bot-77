"""The models registry (phase 5 of docs/training-architecture.md, 2026-10-22): one LanceDB row per model version,
with what it was trained on, its features and params, its scores, and whether it's the adopted one for its role.

Roles (one adopted model each): card (the bot's card model), place (the bot's placement model, shadow), detector_live
(the bot's unit detector), detector_pov / detector_replay (C's detector defaults), spell_pov / spell_replay (C's spell
probability mixes), fake_filter (C's fake-play filter). Old versions stay registered (adopted false) as the history.

The bot and scripts call adopted_path(role, default): the registry's adopted model, or `default` when the registry has
none (so nothing breaks without it). Training scripts register what they produce (not adopted);
`scripts/models.py adopt <model_id>` makes it the one in use.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pyarrow as pa

TABLE = "models"
SCHEMA = pa.schema([
    ("model_id", pa.string()),      # family:version, e.g. card:moves_v1
    ("family", pa.string()),        # card / place / detector / spell / spell_mix / spell_probs / fake_filter / crop_clf / third_party
    ("version", pa.string()),       # the name on disk
    ("path", pa.string()),          # the file the code loads
    ("role", pa.string()),          # what it's (or would be) used for
    ("adopted", pa.bool_()),        # the one in use for its role
    ("status", pa.string()),        # adopted / superseded / not adopted / reference
    ("trained_at", pa.string()),
    ("what", pa.string()),          # what changed / what it was trained on
    ("data", pa.string()),          # JSON: training data (table, version, rows / decisions, splits)
    ("features", pa.string()),      # JSON list of feature names (or a note)
    ("params", pa.string()),        # JSON
    ("metrics", pa.string()),       # JSON: {"heldback top-1": 0.466, ...}
    ("notes", pa.string()),
    ("registered_at", pa.string()),
])


def _db():
    import lancedb
    return lancedb.connect("data/lancedb")


def _table(create: bool = False):
    db = _db()
    if TABLE in db.table_names(limit=10_000):
        return db.open_table(TABLE)
    return db.create_table(TABLE, schema=SCHEMA) if create else None


def list_models(role: str | None = None) -> list[dict]:
    t = _table()
    if t is None:
        return []
    rows = t.to_arrow().to_pylist()
    return sorted((r for r in rows if role is None or r["role"] == role), key=lambda r: (r["family"], r["trained_at"] or ""))


def adopted(role: str) -> dict | None:
    return next((r for r in list_models(role) if r["adopted"]), None)


def adopted_path(role: str, default: str) -> str:
    """The adopted model's file for this role, or `default` (no registry / nothing adopted / file missing)."""
    try:
        r = adopted(role)
    except Exception:  # noqa: BLE001 -- the registry must never stop the bot
        r = None
    return r["path"] if r and r["path"] and Path(r["path"]).exists() else default


def register(model_id: str, family: str, version: str, path: str, role: str, what: str = "", data=None, features=None,
             params=None, metrics=None, notes: str = "", trained_at: str | None = None, status: str = "not adopted",
             adopt: bool = False) -> dict:
    """Add or replace a model version. adopt=True also makes it the adopted one for its role."""
    t = _table(create=True)
    old = t.search().where(f"model_id = '{model_id}'").limit(1).to_arrow().to_pylist()
    if old and old[0]["adopted"] and not adopt:  # retraining a version in place keeps its adoption
        adopt, status = True, "adopted"
    row = {"model_id": model_id, "family": family, "version": version, "path": path, "role": role, "adopted": False,
           "status": status, "trained_at": trained_at or datetime.date.today().isoformat(), "what": what,
           "data": json.dumps(data or {}), "features": json.dumps(features or []), "params": json.dumps(params or {}),
           "metrics": json.dumps(metrics or {}), "notes": notes, "registered_at": datetime.datetime.now().isoformat(timespec="seconds")}
    t.delete(f"model_id = '{model_id}'")
    t.add(pa.Table.from_pylist([row], schema=SCHEMA))
    if adopt:
        adopt_model(model_id)
    return row


def adopt_model(model_id: str) -> None:
    """Make this the adopted model for its role; the previously adopted one becomes superseded."""
    t = _table()
    rows = t.to_arrow().to_pylist()
    me = next((r for r in rows if r["model_id"] == model_id), None)
    if me is None:
        raise KeyError(model_id)
    for r in rows:
        if r["role"] == me["role"] and r["adopted"] and r["model_id"] != model_id:
            t.update(where=f"model_id = '{r['model_id']}'", values={"adopted": False, "status": "superseded"})
    t.update(where=f"model_id = '{model_id}'", values={"adopted": True, "status": "adopted"})
