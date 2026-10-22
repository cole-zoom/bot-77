"""Train the placement model (bot77.place) on `placements` split train and compare it on held-back replays with the
playbook (its typical answer spot / offset from the opponent's play), the earlier placement rules (Policy.place) and
the playbook rules (bot77.live.rules), 2026-10-22. Distances in tiles between the predicted and the real spot (box
centres; rule spots are taps, so FEET_DY is taken off them for troops).

  uv run python scripts/train_place.py [--eval heldback|test]   -> data/models/place_v1/model.pkl
"""
import argparse, json, math, pickle, statistics
from pathlib import Path

import lancedb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

from bot77.board import main_threat, summary_features, to_player_frame, SUMMARY_FEATURES
from bot77.live.policy import Policy
from bot77.live.state import LiveState, Unit
from bot77.place import FEET_DY, KINDS, spot_features, spots
from bot77.playbook import ANSWER_S, zone

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="heldback"); ap.add_argument("--out", default="data/models/place_v1")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()
elixir = {c["name"]: c["elixir"] or 0 for c in cards}; ids = {c["name"]: c["id"] for c in cards}
rows = db.open_table("placements").to_arrow().to_pylist()
for r in rows:
    r["U"] = [{"card": u[0], "team": u[1], "x": u[2], "y": u[3]} for u in json.loads(r["units"])]
    r["T"] = json.loads(r["towers"])
train = [r for r in rows if r["split"] == "train"]

# fixed spots: where players put their cards (troops / buildings on their half, spells anywhere)
pts = np.array([(r["x"], r["y"]) for r in train if r["type"] != "spell"])
spl = np.array([(r["x"], r["y"]) for r in train if r["type"] == "spell"])
fixed = [tuple(c) for c in KMeans(24, n_init=4, random_state=0).fit(pts).cluster_centers_.round(2)] + \
        [tuple(c) for c in KMeans(8, n_init=4, random_state=0).fit(spl).cluster_centers_.round(2)]
code = {i: k for k, i in enumerate(sorted({ids[r["card"]] for r in rows if r["card"] in ids}))}
CATS = ["s_kind", "p_card", "b_threat_card"]


def cand_rows(r):
    summ = summary_features(r["U"], r["T"], elixir, ids)
    cand = spots(r["card"], r["type"], r["U"], fixed, elixir)
    return cand, [{**summ, **spot_features(s, r["card"], r["type"], r["U"], elixir, r["T"]), "p_card": code.get(ids.get(r["card"]), np.nan)}
                  for s in cand]


FEATS = None
Xs, ys = [], []
for r in train:
    cand, fr = cand_rows(r)
    best = min(range(len(cand)), key=lambda i: math.dist((cand[i]["x"], cand[i]["y"]), (r["x"], r["y"])))
    if FEATS is None:
        FEATS = list(fr[0])
    Xs += [[f[k] for k in FEATS] for f in fr]; ys += [int(i == best) for i in range(len(fr))]
X = np.array(Xs, dtype=np.float64); y = np.array(ys)
for f in CATS:  # categorical codes: card ids -> 0..n-1 (b_threat_card is a raw id)
    if f == "b_threat_card":
        j = FEATS.index(f); X[:, j] = [code.get(int(v), np.nan) if v >= 0 else np.nan for v in X[:, j]]
model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30,
                                       l2_regularization=1.0, categorical_features=[f in CATS for f in FEATS], random_state=0)
model.fit(X, y)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "model.pkl").write_bytes(pickle.dumps({"model": model, "features": FEATS, "code": code, "fixed": fixed}))
print(f"trained on {len(train)} plays, {len(X)} candidate spots ({len(fixed)} fixed)")


def predict(r):
    cand, fr = cand_rows(r)
    Xp = np.array([[f[k] for k in FEATS] for f in fr], dtype=np.float64)
    j = FEATS.index("b_threat_card"); Xp[:, j] = [code.get(int(v), np.nan) if v >= 0 else np.nan for v in Xp[:, j]]
    p = model.predict_proba(Xp)[:, 1]
    best = cand[int(np.argmax(p))]
    oracle = min(cand, key=lambda s: math.dist((s["x"], s["y"]), (r["x"], r["y"])))
    return (best["x"], best["y"]), (oracle["x"], oracle["y"])


# baselines
pol = Policy(cards, playbook=False)
book = {(b["opp_card"], b["opp_zone"], b["answer"]): b for b in db.open_table("playbook").to_arrow().to_pylist()}
ev_rows = [r for r in rows if r["split"] == a.eval]
by_match = {}
for r in rows:
    by_match.setdefault(r["match_id"], []).append(r)
ctype = {c["name"]: c["type"] for c in cards}


def box(spot, typ):  # rule spots are taps (feet); compare box centres
    return None if spot is None else (spot[0], spot[1] - (FEET_DY if typ != "spell" else 0.0))


res = {}
for r in ev_rows:
    true = (r["x"], r["y"])
    st_units = [Unit(u["card"], u["card"], "enemy" if u["team"] == "opp" else "friendly", u["x"], u["y"], 0.9) for u in r["U"]]
    st = LiveState(r["t"], [None] * 4, None, 7.0, 100, 80, 1, "regulation", st_units)
    model_spot, oracle = predict(r)
    g = {"placement model": model_spot, "best possible (nearest candidate)": oracle,
         "earlier rules (Policy.place)": box(pol.place(r["card"], r["type"], st.threats(), st), r["type"])}
    R = pol.rules
    th = main_threat(r["U"], elixir)
    if r["type"] == "spell":
        g["playbook rules"] = R.spell_spot(r["card"], st)
    elif th:
        g["playbook rules"] = box(R.defend_spot(r["card"], r["type"], Unit(th["card"], th["card"], "enemy", th["x"], th["y"], 0.9), st), r["type"])
    else:
        opp_l = sum(1 for u in r["U"] if u["team"] == "opp" and u["x"] < 9)  # the policy's attack lane: fewer enemies
        opp_r = sum(1 for u in r["U"] if u["team"] == "opp") - opp_l
        g["playbook rules"] = box(R.leak_spot(r["card"], r["type"], "right" if opp_r < opp_l else "left"), r["type"])
    # the playbook: only for answers (an opponent play in the 8 s before)
    other = "top" if r["side"] == "bottom" else "bottom"
    prev = [q for q in by_match[r["match_id"]] if q["side"] == other and 0 < r["t"] - q["t"] <= ANSWER_S]
    answer = bool(prev)
    if prev:
        q = prev[-1]
        ox, oy = to_player_frame(q["x"], q["y"], "top", spell=q["type"] == "spell")  # the other player's frame -> ours
        b = book.get((q["card"], zone(oy), r["card"])) or book.get((q["card"], "any", r["card"]))
        if b and b["dx"] is not None:
            g["playbook spot"] = (b["ans_x"], b["ans_y"]) if (ox < 9) == (b["ans_x"] < 9) else (18 - b["ans_x"], b["ans_y"])
            g["playbook offset"] = (min(max(ox + b["dx"], 0.5), 17.5), min(max(oy + b["dy"], 0.5), 29.0))
    for k, s in g.items():
        if s is None:
            continue
        rec = (math.dist(s, true), (s[0] < 9) == (true[0] < 9))
        res.setdefault(("all plays", k), []).append(rec)
        if "playbook spot" in g:
            res.setdefault(("answers the playbook covers", k), []).append(rec)
        if answer:
            res.setdefault(("answers", k), []).append(rec)
            if r["side"] == "bottom":
                res.setdefault(("answers, bottom player", k), []).append(rec)
for subset in ("all plays", "answers", "answers, bottom player", "answers the playbook covers"):
    print(f"\n{a.eval} / {subset}")
    for (s, k), v in res.items():
        if s == subset:
            print(f"  {k:36s} n={len(v):5d}  median {statistics.median(x[0] for x in v):4.1f} tiles  "
                  f"within 2 tiles {sum(x[0] <= 2 for x in v) / len(v):4.0%}  same lane {sum(x[1] for x in v) / len(v):4.0%}")
