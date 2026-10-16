"""Fit "detector call -> card actually played" on one half of the development set and check it on the other
(decisions.md, 2026-10-11). Benchmark matches are never loaded.

Each enemy deploy (after the v2 play rules) is matched to the nearest true play within the scoring window; the
pairs (called card, true card) give P(true | called). The deck is then inferred from the mapped votes and each
deploy picks the deck card with the highest mapped score, instead of only the deck cards it literally voted for.

  uv run python scripts/fit_confusion.py <det tag> [--alpha 1] [--self 3] [--out config/confusion.json]
"""
import argparse, collections, copy, glob, json
from pathlib import Path

import lancedb

from bot77.layout import Layout
from bot77.playrules import apply_rules
from bot77.review import MATCH_WINDOW_S
from bot77.spawns import apply_deck, deploys, drop_pov_units, infer_deck, resolve_composites

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--self", dest="self_w", type=float, default=3.0, help="pseudo-count for 'the call is right'")
ap.add_argument("--out", default=None)
a = ap.parse_args()
db = lancedb.connect("data/lancedb")
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
is_spell = lambda c: ctype.get(c) == "spell"
data = {}
for f in sorted(glob.glob("data/ground_truth/opponent_truth_auto_*.json")):
    t = json.loads(Path(f).read_text()); m = t["matches"][0]
    if m in hold:
        continue
    vid = m.rsplit("_m", 1)[0]
    video = db.open_table("videos").search().where(f"video_id = '{vid}'").to_arrow().to_pylist()[0]
    lay = Layout.load(video["layout_id"]); ax0, ay0, ax1, ay1 = lay.katacr_arena; tile = (ax1 - ax0) / 18
    dets = db.open_table(f"unit_detections_{a.tag}").search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    ev = deploys(dets, names, tile, lambda c: ctype.get(c) == "building", is_spell, (ax0, ay0), spell_team="enemy")
    pov = [(p["t_video"], p["card"]) for p in db.open_table("events").search().where(f"match_id = '{m}'").limit(10**5)
           .select(["t_video", "card"]).to_arrow().to_pylist() if p["card"]]
    enemy, _ = drop_pov_units(resolve_composites([e for e in ev if e.team == "enemy"]), pov)
    data[m] = (apply_rules(enemy, pov, ctype, {}), t["items"])


def score(events, items):
    used_t, used_e, right = set(), set(), 0
    found = 0
    for exact in (True, False):
        for i, e in sorted(enumerate(events), key=lambda ie: ie[1].t):
            if i in used_e:
                continue
            c = [(abs(x["t_video"] - e.t), k) for k, x in enumerate(items) if k not in used_t
                 and abs(x["t_video"] - e.t) <= MATCH_WINDOW_S and (x["label"] == e.card) == exact]
            if c:
                _, k = min(c); used_t.add(k); used_e.add(i); found += 1; right += exact
    return right, len(events) - found


def fit(ms):
    """counts[called][true] from deploys matched to the nearest true play (any card) within the window."""
    counts = collections.defaultdict(collections.Counter)
    for m in ms:
        enemy, items = data[m]
        for e in enemy:
            if e.card in (None, "__ability"):
                continue
            near = [(abs(x["t_video"] - e.t), x["label"]) for x in items if abs(x["t_video"] - e.t) <= MATCH_WINDOW_S]
            counts[e.card][min(near)[1] if near else "__none"] += 1
    return counts


def mapped(e, counts, alpha, self_w):
    """Expected true-card weights of a deploy: each voted card spreads its votes by P(true | called)."""
    out = collections.Counter()
    for c, v in e.card_votes.items():
        row = counts.get(c, collections.Counter())
        tot = sum(row.values()) + self_w + alpha * len(row)
        out[c] += v * (self_w + alpha * (c in row) + row.get(c, 0)) / tot
        for t, n in row.items():
            if t not in (c, "__none"):
                out[t] += v * (n + alpha) / tot
    return out


def run(ms, counts=None, alpha=1.0, self_w=3.0):
    R = X = N = 0
    for m in ms:
        enemy, items = data[m]
        ev = copy.deepcopy(enemy)
        abil = [e for e in ev if e.card == "__ability"]
        rest = [e for e in ev if e.card != "__ability"]
        if counts is None:
            apply_deck(rest, set(infer_deck(rest)))
        else:
            maps = [mapped(e, counts, alpha, self_w) for e in rest]
            wins = collections.Counter(mp.most_common(1)[0][0] for mp, e in zip(maps, rest) if mp and not e.subunit_only)
            deck = {c for c, _ in wins.most_common(8)}
            for e, mp in zip(rest, maps):
                ind = {c: v for c, v in mp.items() if c in deck}
                e.card = max(ind, key=ind.get) if ind else None
        r, x = score(sorted(rest + abil, key=lambda e: e.t), items)
        R += r; X += x; N += len(items)
    return R / N, X / len(ms)


A = sorted(m for m in data if m.startswith("cole")); B = sorted(m for m in data if m.startswith("friend"))
print(f"dev matches: fit half A={len(A)} (cole_*), check half B={len(B)} (friend_*)")
print("baseline           A %.3f (fake/m %.1f)  B %.3f (fake/m %.1f)" % (*run(A), *run(B)))
cA, cB = fit(A), fit(B)
for al in (0.5, 1.0):
    for sw in (1.0, 3.0, 6.0):
        print(f"alpha {al} self {sw}: fit A -> B %.3f (fake/m %.1f) | fit B -> A %.3f (fake/m %.1f)"
              % (*run(B, cA, al, sw), *run(A, cB, al, sw)))
if a.out:
    full = fit(A + B)
    Path(a.out).write_text(json.dumps({"tag": a.tag, "alpha": a.alpha, "self": a.self_w,
                                       "counts": {k: dict(v) for k, v in full.items()}}, indent=1))
    print("wrote", a.out)
