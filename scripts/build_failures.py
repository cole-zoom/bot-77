"""Every keyed opponent play's outcome, as a LanceDB table to mine failure modes from (2026-10-16).

One row per true play (right / wrong / missed) and per fake event, from an events table scored against the answer keys
with the same one-to-one matching as eval (scripts/render_game.match_plays). Context columns say *why* a play may have
failed: was the true card seen by the detector near the play (and how late), how close the neighbouring plays are,
whether it repeats the previous card (Mirror), whether the spell stage had probabilities for the match.

  uv run python scripts/build_failures.py --events opponent_events_dev_fm_s0,... --dets v5l_e15 --set replay --split auto
Test and benchmark games are never written (keep them for scoring): --set replay skips config/replay_test_games.json.
"""
import argparse, glob, json, sys
from pathlib import Path

import lancedb
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
from render_game import match_plays  # noqa: E402
from bot77.detect.names import class_to_card  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--events", required=True, help="comma list of opponent events tables")
ap.add_argument("--dets", required=True, help="detections tag the events were made from (unit_detections_<tag>)")
ap.add_argument("--set", default="replay", help="truth set: replay | auto")
ap.add_argument("--split", default="", help="split label; default: heldback / train for replay, dev for auto")
ap.add_argument("--table", default="play_outcomes")
ap.add_argument("--mode", default="append", choices=["append", "overwrite"])
ap.add_argument("--spell-probs", default="data/models/spell_mix/spell_probs")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["name"]).to_arrow().to_pylist()]
ctype = {c["name"]: c["type"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["name", "type"]).to_arrow().to_pylist()}
excluded = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
excluded |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
heldback = set()
hb = Path("config/heldback_replays.json")
if hb.exists():
    heldback = set(json.loads(hb.read_text())["matches"])
truth = {}
for f in sorted(glob.glob(f"data/ground_truth/opponent_truth_{a.set}_*.json")):
    t = json.loads(Path(f).read_text())
    if t["matches"][0] not in excluded:
        truth[t["matches"][0]] = t
dets = db.open_table(f"unit_detections_{a.dets}")
cls_card = {}


def card_of(cls):
    if cls not in cls_card:
        cls_card[cls] = class_to_card(cls, names)[0]
    return cls_card[cls]


rows = []
for tn in a.events.split(","):
    tbl = db.open_table(tn)
    ms = set(tbl.search().select(["match_id"]).limit(10**7).to_arrow().column("match_id").to_pylist())
    for m in sorted(ms & set(truth)):
        items = sorted([i for i in truth[m]["items"] if i["match_id"] == m], key=lambda i: i["t_video"])
        ev = tbl.search().where(f"match_id = '{m}'").limit(100000).select(
            ["t_video", "card", "votes", "units", "x_tiles", "y_tiles", "subunit_only"]).to_arrow().to_pylist()
        d = dets.search().where(f"match_id = '{m}' AND team = 'enemy'").limit(10**6).select(
            ["t_video", "cls", "conf"]).to_arrow().to_pylist()
        seen = {}
        for x in d:
            c = card_of(x["cls"])
            if c:
                seen.setdefault(c, []).append(x["t_video"])
        split = a.split or (("heldback" if m in heldback else "train") if a.set == "replay" else "dev")
        has_sp = (Path(a.spell_probs) / f"{m}.npz").exists()
        tt = [i["t_video"] for i in items]
        for t_, kind, label, e, _ in match_plays(items, ev):
            r = {"match_id": m, "video_id": m.rsplit("_m", 1)[0], "split": split, "events_table": tn, "kind": kind,
                 "t_video": float(t_), "true_card": None, "true_form": None, "pred_card": None, "pred_votes": None,
                 "units": None, "x_tiles": None, "y_tiles": None, "card_type": None, "prev_card": None,
                 "gap_prev": None, "gap_next": None, "repeat_prev": False, "seen_n": 0, "seen_first_dt": None,
                 "nearest_event_dt": None, "nearest_event_card": None, "has_spell_probs": has_sp}
            if e is not None:
                r.update(pred_card=e["card"], pred_votes=e["votes"], units=e["units"], x_tiles=e["x_tiles"],
                         y_tiles=e["y_tiles"])
            if kind != "FAKE":
                k = next(k for k, i in enumerate(items) if i["t_video"] == t_ and i["label"] == label)
                it = items[k]
                r.update(true_card=label, true_form=it.get("form"), card_type=ctype.get(label))
                if k > 0:
                    r.update(prev_card=items[k - 1]["label"], gap_prev=t_ - tt[k - 1], repeat_prev=items[k - 1]["label"] == label)
                if k + 1 < len(items):
                    r["gap_next"] = tt[k + 1] - t_
                near = [s - t_ for s in seen.get(label, []) if -0.5 <= s - t_ <= 4.0]
                r.update(seen_n=len(near), seen_first_dt=min(near) if near else None)
                if ev:
                    j = min(range(len(ev)), key=lambda j: abs(ev[j]["t_video"] - t_))
                    r.update(nearest_event_dt=ev[j]["t_video"] - t_, nearest_event_card=ev[j]["card"])
            else:
                r["card_type"] = ctype.get(label)
                if tt:
                    k = min(range(len(tt)), key=lambda k: abs(tt[k] - t_))
                    r.update(nearest_event_dt=tt[k] - t_, nearest_event_card=items[k]["label"])
            rows.append(r)
    print(tn, len(rows), "rows so far", flush=True)

tbl = pa.Table.from_pylist(rows)
if a.mode == "overwrite" or a.table not in db.table_names(limit=10_000):
    db.create_table(a.table, tbl, mode="overwrite")
else:
    db.open_table(a.table).add(tbl)
print(a.table, db.open_table(a.table).count_rows(), "rows")
