"""Train A v1 (bot77.moves): which of the 4 hand cards a player plays next. Gradient boosting on (decision, candidate)
rows, the highest-scoring candidate wins. Trained on `move_samples` split train; held-back tunes; test is scored once.
Baselines: random (25%), cheapest affordable, the card waiting longest (cycle order), the card played most so far.

  uv run python scripts/train_moves.py [--league 0]   -> data/models/moves_v1/model.pkl
"""
import argparse, pickle
from collections import defaultdict
from pathlib import Path

import lancedb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from bot77.features import CAND_FEATURES, CTX_FEATURES, add_response_rates

ap = argparse.ArgumentParser()
ap.add_argument("--league", type=int, default=None, help="train only on this TV league (0 = Ultimate Champion)")
ap.add_argument("--out", default="data/models/moves_v1")
ap.add_argument("--what", default="", help="one line for the models registry: what this version changes")
ap.add_argument("--table", default="move_samples", help="samples table (move_samples_m: rebuilt from moments)")
ap.add_argument("--board", action="store_true", help="A v2: add the board features (bot77.board, table board_states)")
ap.add_argument("--eval", default="heldback,test")
ap.add_argument("--c-table", default="", help="opponent_events table: also score the bottom player with the opponent's "
                "history from C's detections instead of the replay's top hand (what B would see)")
ap.add_argument("--c-split", default="heldback")
a = ap.parse_args()

rows = lancedb.connect("data/lancedb").open_table(a.table).to_arrow().to_pylist()
if a.board:  # A v2 (2026-10-22): the units on the field at each decision
    import json
    from bot77.features import CAND_BOARD_FEATURES, SUMMARY_FEATURES, add_board_features
    _db = lancedb.connect("data/lancedb")
    _cards = {c["id"]: c for c in _db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
              .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}
    boards = {(b["game"], b["decision"]): b for b in _db.open_table("board_states").to_arrow().to_pylist()}
    by_dec = defaultdict(list)
    for r in rows:
        by_dec[(r["game"], r["decision"])].append(r)
    for key, rs in by_dec.items():
        b = boards.get(key)
        units = [{"card": u[0], "team": u[1], "x": u[2], "y": u[3]} for u in json.loads(b["units"])] if b else []
        add_board_features(rs, units, json.loads(b["towers"]) if b else {}, _cards)


RATES = add_response_rates(rows)  # bot77.features.rates: cross-fitted for train rows
FEATS = CAND_FEATURES + CTX_FEATURES + ["resp_rate", "own_rate_after", "elixir_after"]
CATS = ["cand_card", "next_card"] + [f for f in CTX_FEATURES if f.startswith(("own_prev", "opp_prev")) and not f.endswith("_dt")]
if a.board:
    FEATS += SUMMARY_FEATURES + CAND_BOARD_FEATURES + ["resp_threat"]
    CATS += ["b_threat_card"]
ids = sorted({r[f] for r in rows for f in CATS if r[f] is not None and r[f] >= 0})
code = {c: i for i, c in enumerate(ids)}  # card ids -> 0..n-1 for categorical splits
assert len(code) < 255, len(code)


def X(rs):
    return np.array([[(code.get(r[f], np.nan) if f in CATS else r[f]) for f in FEATS] for r in rs], dtype=np.float64)


def decisions(rs):
    d = defaultdict(list)
    for r in rs:
        d[(r["game"], r["decision"])].append(r)
    return list(d.values())


def accuracy(decs, score):
    top1 = top2 = 0
    for d in decs:
        s = score(d)
        order = sorted(range(len(d)), key=lambda i: -s[i])
        win = next(i for i, r in enumerate(d) if r["chosen"])
        top1 += order[0] == win; top2 += win in order[:2]
    return top1 / len(decs), top2 / len(decs)


train = [r for r in rows if r["split"] == "train" and (a.league is None or r["league"] == a.league)]
cat_mask = [f in CATS for f in FEATS]
model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30,
                                       l2_regularization=1.0, categorical_features=cat_mask, random_state=0)
model.fit(X(train), np.array([r["chosen"] for r in train]))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "model.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATS, "cats": CATS, "code": code,
                                              "rates": RATES.to_dict()}))
print(f"trained on {len(decisions(train))} decisions" + (f" (league {a.league})" if a.league is not None else ""))

baselines = {
    "random": lambda d: list(np.random.default_rng(0).random(len(d))),
    "cheapest affordable": lambda d: [r["cand_affordable"] * 10 - r["cand_cost"] for r in d],
    "waited longest": lambda d: [r["cand_waited_s"] for r in d],
    "played most so far": lambda d: [r["cand_last_own_gap"] + 0.001 * r["cand_waited_s"] for r in d],
    "A v1": lambda d: list(model.predict_proba(X(d))[:, 1]),
}
METRICS = {}
for split in a.eval.split(","):
    for subset, keep in (("all", lambda r: True), ("top ladder (league 0)", lambda r: r["league"] == 0)):
        decs = decisions([r for r in rows if r["split"] == split and keep(r)])
        if not decs:
            continue
        print(f"\n{split} / {subset}: {len(decs)} decisions")
        for name, fn in baselines.items():
            t1, t2 = accuracy(decs, fn)
            print(f"  {name:22s} top-1 {t1:.1%}  top-2 {t2:.1%}")
            if name == "A v1":
                tag = split if subset == "all" else f"{split} top ladder"
                METRICS[f"{tag} top-1"], METRICS[f"{tag} top-2"] = round(t1, 4), round(t2, 4)
                METRICS[f"{tag} decisions"] = len(decs)

if out.resolve().is_relative_to(Path("data/models").resolve()):  # register what lands in data/models (bot77.models)
    from bot77.models import register
    tbl = lancedb.connect("data/lancedb").open_table(a.table)
    register(f"card:{out.name}", "card", out.name, str(out / "model.pkl"), "card", what=a.what,
             data={"table": a.table, "table_version": tbl.version, "train_decisions": len(decisions(train)),
                   "league": a.league, "board": a.board},
             features=FEATS, params=model.get_params() | {"rates_prior": 20},
             metrics=METRICS, notes="registered by train_moves.py; adopt with scripts/models.py adopt")
    print(f"registered card:{out.name} (not adopted)")

if a.c_table:
    # same decisions, the opponent's history from C instead of the key (bottom player of each replay)
    import json
    from bot77.moves import decision_rows
    db = lancedb.connect("data/lancedb")
    cards_rows = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(
        ["id", "name", "elixir", "type"]).to_arrow().to_pylist()
    cards = {c["id"]: c for c in cards_rows}; names = {c["name"]: c["id"] for c in cards_rows}
    ev, hud_t, ct = db.open_table("events"), db.open_table("hud_states"), db.open_table(a.c_table)
    games = sorted({r["match_id"] for r in rows if r["split"] == a.c_split and r["side"] == "bottom"})
    key_rows, c_rows = [], []
    for m in games:
        own = sorted(({"t": e["t_video"], "card": "__ability" if e["kind"] == "ability" else e["card"]} for e in
                      ev.search().where(f"match_id = '{m}' AND kind IN ('play', 'ability')").limit(10**4).to_arrow().to_pylist()),
                     key=lambda p: p["t"])
        opp_c = sorted(({"t": e["t_video"], "card": e["card"]} for e in ct.search().where(f"match_id = '{m}'").limit(10**4)
                        .select(["t_video", "card"]).to_arrow().to_pylist() if e["card"]), key=lambda p: p["t"])
        if not opp_c:
            continue
        h = sorted(hud_t.search().where(f"match_id = '{m}'").limit(10**6).select(
            ["t_video", "t_match", "hand", "next", "elixir", "multiplier"]).to_arrow().to_pylist(), key=lambda r: r["t_video"])
        for r in decision_rows(own, opp_c, h, cards, names):
            c_rows.append({**r, "game": f"{m}:bottom", "split": "c", "league": -1})
        key_rows += [r for r in rows if r["match_id"] == m and r["side"] == "bottom"]
    add_response_rates(c_rows, full=RATES)
    for name, rs in (("opponent history from the replay key", key_rows), ("opponent history from C", c_rows)):
        decs = decisions(rs)
        t1, t2 = accuracy(decs, baselines["A v1"])
        print(f"{a.c_split}, bottom player, {name}: {len(decs)} decisions, top-1 {t1:.1%}  top-2 {t2:.1%}")
