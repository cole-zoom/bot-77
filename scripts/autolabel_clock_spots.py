"""Label the OPPONENT's units at their deploy clocks (decisions.md, 2026-10-11).

Paired matches give the card and time of every opponent play, but not the spot (mirrored spots carry two videos'
calibration error), and the detector often doesn't recognise the opponent's red unit, so there is no track to box.
The deploy clock fixes both: it starts ~1 s after the play, exactly where the unit lands. Once the clock is gone the
unit stands there, with its level badge (class "bar-level", reliably detected even when the unit is not) just above
its head; flying units are drawn 1-2 tiles above the spot, so the badge, not the clock, anchors the box. SAM, prompted
just below the badge, boxes the unit; the paired truth names it. (Trial 2026-10-11: prompting at the clock boxed the
clock icon itself.)

Only single-unit troops and buildings (no spells, abilities, Mirror, groups or two-form cards). By default only
plays the detector did NOT already name (no detection of the true card at the spot); --all keeps every play.
Benchmark matches, the V5 check games (config/v5_check_games.json) and the replay test set
(config/replay_test_games.json) are never labelled. --truth replay labels the both-hands replays (2026-10-13).

Replays (2026-10-14, after review round 2): plays V4 already named are labelled too, from V4's own box of the true
entity; otherwise every enemy badge near the clock is tried and the crop the reference check likes best is kept; plays
0.8 s apart count as alone; and each labelled unit is followed for 5 s (V4's continuing box, whatever it calls it,
re-checked every frame) for one more frame per second.

  uv run python scripts/autolabel_clock_spots.py <dets table> <out dir> [--all] [--limit N]
"""
import argparse, collections, json, random
from pathlib import Path

import lancedb
import numpy as np

from bot77.autolabel import FORM_CLASS, MULTI, UNITS, AutoLabeler, slug
from bot77.calibrate import read_frame
from bot77.clocks import clock_starts, side_team
from bot77.detect.names import class_to_card
from bot77.layout import Layout

ap = argparse.ArgumentParser()
ap.add_argument("table"); ap.add_argument("out")
ap.add_argument("--all", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--truth", default="auto", help="comma list of truth sets: auto (dev pairs), replay (both-hands replays)")
ap.add_argument("--only-new", default="", help="previous truth dir: label only plays that aren't in it (same match, time, card)")
ap.add_argument("--refs", default="data/external/replay_refs/refs.npz", help="replay reference crops (scripts/replay_refs.py)")
a = ap.parse_args()
LAG = {"auto": (0.2, 2.5), "replay": (-0.3, 0.4)}  # clock start after the play: median ~1 s on the dev set; on
# replays the hand read and the clock coincide (±0.2 s, 2026-10-14) and the old window caught the NEXT play's clock
FRAME_OFFSETS = (0.6, 1.1)  # after the clock's last frame: the unit has landed and barely moved
BADGE_TILES = 3.0  # the badge is within this of the clock spot (flying units sit higher)
MIN_SIM = 0.4  # DINOv2 similarity to the card's approved slices (trial: the 3 wrong boxes of 21 were 0.31-0.36)
ALONE_S = {"auto": 2.0, "replay": 0.8}  # skip a play with another opponent play this close (two plays can share one
# clock); replay clocks sit within ±0.3 s of the play, so plays 0.8 s apart no longer share one
BELOW_BADGE = (0.3, 0.9)  # SAM point = badge centre + this many tiles (x, y): the unit's body
BOX_TILES = (0.6, 4.5)  # plausible unit box side, tiles
# Replay review round 2 (2026-10-14): 21/60 right; the wrong ones were mostly buildings and spirits, and 50 of 60 frames
# had no approved slice to check against. Replay plays now skip those cards and must pass the replay-reference check:
# the crop's nearest card among the references (V4's confident, truth-confirmed boxes from other replay matches) is the
# named card, by REF_MARGIN, at REF_SIM or more. Same-arena references lift every crop's similarity, hence the margin.
REPLAY_SKIP = {"Fire Spirit", "Ice Spirit", "Electro Spirit", "Heal Spirit"}
REF_SIM, REF_MARGIN, REF_MIN = 0.6, 0.0, 5  # on round 2's verdicts: kept ~10/21 right, 1/32 wrong

hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
check = json.loads(Path("config/v5_check_games.json").read_text())["matches"]
hold |= set(check)
hold |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])  # replay test set
al = AutoLabeler(out=Path(a.out), dets_table=a.table)
db = al.db
stats, rows = collections.Counter(), []
items = []
for ts, f in sorted((ts, f) for ts in a.truth.split(",") for f in Path("data/ground_truth").glob(f"opponent_truth_{ts}_*.json")):
    tr = json.loads(f.read_text())
    m = tr["matches"][0]
    if m in hold:
        continue
    prev = Path(a.only_new) / f.name if a.only_new else None
    seen = {(round(i["t_video"], 1), i["label"]) for i in json.loads(prev.read_text())["items"]} if prev and prev.exists() else set()
    for i in tr["items"]:
        if (round(i["t_video"], 1), i["label"]) in seen:
            continue
        card, form = i["label"], i.get("form")
        if card in ("Mirror", "__ability") or al.ctype.get(card) == "spell" or card in MULTI or UNITS.get(card, 1) > 1:
            stats["skip_card_kind"] += 1
            continue
        ent = FORM_CLASS.get(slug(card, form), slug(card, form))
        if ent not in al.cidx:
            stats["skip_no_class"] += 1
            continue
        if any(j is not i and abs(j["t_video"] - i["t_video"]) <= ALONE_S[ts] for j in tr["items"]):
            stats["skip_not_alone"] += 1
            continue
        if ts == "replay" and (card in REPLAY_SKIP or al.ctype.get(card) == "building"):
            stats["skip_replay_kind"] += 1
            continue
        items.append({"video_id": tr["video_id"], "match_id": m, "t_video": i["t_video"], "card": card, "entity": ent,
                      "team": "enemy", "source": "paired-clock", "lag": LAG[ts], "set": ts})
refs = np.load(a.refs) if "replay" in a.truth.split(",") and Path(a.refs).exists() else None


def ref_check(crop, entity, match_id):
    """(ok, sim, best other entity, its sim) against the replay references, leaving out the play's own match."""
    keep = refs["match_id"] != match_id
    ents, E = refs["entity"][keep], refs["emb"][keep]
    if (ents == entity).sum() < REF_MIN:
        return None
    q = al.anchor.emb([crop])[0]
    sims = E @ q
    best = {}
    for e, x in zip(ents, sims):
        best[e] = max(best.get(e, -1.0), float(x))
    mine = best.pop(entity)
    other, osim = max(best.items(), key=lambda kv: kv[1])
    return mine >= REF_SIM and mine - osim > REF_MARGIN, mine, other, osim


NON_UNIT = set(al.classes[:15])  # towers, bars, badges, clocks, emotes, text, elixir, selection
CONFIRMED_CONF = 0.7  # replay: V4's box of exactly the truth's entity at the clock is the unit (no SAM, still checked)
TRACK_S, TRACK_STEP = 5.0, 1.0  # replay: follow the unit this long after the first box, one frame per step
TRACK_TILES, TRACK_SIZE = 1.5, (0.6, 1.6)  # next box: centre this close (tiles), size ratio in this range


def centre(d):
    return ((d["x0"] + d["x1"]) / 2 - ax0) / tile, ((d["y0"] + d["y1"]) / 2 - ay0) / tile


def check(crop, it):
    """(ok, sim, other, other_sim, reason)."""
    rc = ref_check(crop, it["entity"], it["match_id"]) if refs is not None else None
    if rc is not None:
        return (*rc, "ref_reject")
    sim = al.anchor.score(crop, it["entity"])
    if refs is not None:  # replay: an unchecked box is not kept
        return sim is not None and sim >= REF_SIM, sim, None, None, "ref_none" if sim is None else "low_sim"
    return sim is None or sim >= MIN_SIM, sim, None, None, "low_sim"


def from_badges(f, s, t, c, it):
    """Every enemy badge near the clock: SAM below it, keep the crop the check likes best (round 2: the nearest badge
    was often another unit's)."""
    badges = [d for d in by_t[t] if d["cls"] == "bar-level" and d["team"] == "enemy" and d["conf"] >= 0.4]
    near = [d for d in badges if abs(centre(d)[0] - c.x) + abs(centre(d)[1] - c.y) <= BADGE_TILES
            and (d["y1"] - ay0) / tile <= c.y + 0.5]
    if not near:
        stats["no_badge"] += 1
        return None
    best, why = None, "sam_empty"
    for b in sorted(near, key=lambda d: abs(centre(d)[0] - c.x) + abs(centre(d)[1] - c.y))[:4]:
        px = ((b["x0"] + b["x1"]) / 2 + BELOW_BADGE[0] * tile) * s
        py = (b["y1"] + BELOW_BADGE[1] * tile) * s
        res = al.sam(f, points=[[px, py]], labels=[1], verbose=False)[0]
        if res.masks is None or not len(res.masks.data):
            continue
        ys, xs = np.where(res.masks.data[0].cpu().numpy())
        if len(xs) < 30:
            why = "sam_tiny"; continue
        bw, bh = (xs.max() - xs.min()) / (tile * s), (ys.max() - ys.min()) / (tile * s)
        if not (BOX_TILES[0] <= bw <= BOX_TILES[1] and BOX_TILES[0] <= bh <= BOX_TILES[1]):
            why = "sam_implausible"; continue
        box = (xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s)
        ok, sim, other, osim, why2 = check(f[ys.min():ys.max() + 1, xs.min():xs.max() + 1], it)
        if not ok:
            why = why2; continue
        if best is None or (sim or 0) > (best[1] or 0):
            best = (box, sim, other, osim, "clock+sam")
    if best is None:
        stats[why] += 1
    return best


def from_v4(f, s, t, c, it):
    """V4's own confident box of the truth's entity at the clock (the play it already named)."""
    hits = [d for d in by_t[t] if d["cls"] == it["entity"] and d["conf"] >= CONFIRMED_CONF
            and abs(centre(d)[0] - c.x) + abs(centre(d)[1] - c.y) <= 2.5]
    if not hits:
        return None
    d = max(hits, key=lambda d: d["conf"])
    box = (d["x0"], d["y0"], d["x1"], d["y1"])
    ok, sim, other, osim, why = check(f[int(d["y0"] * s):int(d["y1"] * s), int(d["x0"] * s):int(d["x1"] * s)], it)
    if not ok:
        stats[why] += 1
        return None
    return box, sim, other, osim, "v4-confirmed"


def follow(box, t):
    """V4's box (any class: the point is the ones it misnames) that continues `box` at the frame nearest t."""
    tt = min(times, key=lambda x: abs(x - t))
    if abs(tt - t) > 0.3:
        return None, tt
    bx, by = ((box[0] + box[2]) / 2 - ax0) / tile, ((box[1] + box[3]) / 2 - ay0) / tile
    area = (box[2] - box[0]) * (box[3] - box[1])
    cand = [d for d in by_t[tt] if d["cls"] not in NON_UNIT and d["conf"] >= 0.3
            and abs(centre(d)[0] - bx) + abs(centre(d)[1] - by) <= TRACK_TILES
            and TRACK_SIZE[0] <= (d["x1"] - d["x0"]) * (d["y1"] - d["y0"]) / area <= TRACK_SIZE[1]]
    if not cand:
        return None, tt
    d = min(cand, key=lambda d: abs(centre(d)[0] - bx) + abs(centre(d)[1] - by))
    return (d["x0"], d["y0"], d["x1"], d["y1"]), tt


def write(it, v, lay, t, f, s, found, c, named):
    box, sim, other, osim, how = found
    r = al._write(it, v, lay, t, box, it["entity"], how, sim, by_t[t])
    if r:
        r.update(clock_t=round(c.t, 2), named_by_detector=named, ref_other=other, ref_other_sim=osim)
        rows.append(r)
        return 1
    return 0


random.Random(0).shuffle(items)
for it in items:
    if a.limit and stats["plays_labelled"] >= a.limit:
        break
    replay = it["set"] == "replay"
    v = al.videos[it["video_id"]]
    lay = Layout.load(v["layout_id"])
    ax0, ay0, ax1, ay1 = lay.katacr_arena
    tile = (ax1 - ax0) / 18
    t0 = it["t_video"]
    dets = al.dets.search().where(f"match_id = '{it['match_id']}' AND t_video >= {t0 - 1} AND t_video <= {t0 + 6 + TRACK_S}") \
        .limit(10**6).to_arrow().to_pylist()
    cl = [c for c in clock_starts(dets, tile, (ax0, ay0), 0.3)
          if c.n >= 2 and c.y >= 1.5 and side_team(c, 1.0) == "enemy" and it["lag"][0] <= c.t - t0 <= it["lag"][1]]
    if len(cl) != 1:
        stats["skip_clocks_%s" % ("none" if not cl else "several")] += 1
        continue
    c = cl[0]
    named = any(class_to_card(d["cls"], al.card_names)[0] == it["card"] and d["conf"] >= 0.5 and c.t <= d["t_video"] <= c.t + 3
                and abs(((d["x0"] + d["x1"]) / 2 - ax0) / tile - c.x) + abs(((d["y0"] + d["y1"]) / 2 - ay0) / tile - c.y) <= 2.5
                for d in dets)
    if named and not (a.all or replay):  # replay: V4's named plays are labelled too, from its own box
        stats["skip_already_named"] += 1
        continue
    by_t = collections.defaultdict(list)
    for d in dets:
        by_t[round(d["t_video"], 3)].append(d)
    times = sorted(by_t)
    wrote, last = 0, None
    for off in FRAME_OFFSETS:
        t = min(times, key=lambda x: abs(x - (c.last_t + off)))
        f = read_frame(Path(v["path"]), t)
        s = lay.scale(f.shape[1])
        found = (from_v4(f, s, t, c, it) if replay else None) or from_badges(f, s, t, c, it)
        if found:
            wrote += write(it, v, lay, t, f, s, found, c, named)
            last = (found[0], t)
    if replay and last:  # follow the unit: more poses and spots per play, every box re-checked so a drift stops it
        box, t = last
        for k in range(1, int(TRACK_S / TRACK_STEP) + 1):
            box, t = follow(box, last[1] + k * TRACK_STEP)
            if box is None:
                stats["track_lost"] += 1
                break
            f = read_frame(Path(v["path"]), t)
            s = lay.scale(f.shape[1])
            ok, sim, other, osim, why = check(f[int(box[1] * s):int(box[3] * s), int(box[0] * s):int(box[2] * s)], it)
            if not ok:
                stats["track_" + why] += 1
                break
            n = write(it, v, lay, t, f, s, (box, sim, other, osim, "track"), c, named)
            wrote += n
            stats["track_frames"] += n
    stats["plays_labelled" if wrote else "plays_no_frame"] += 1
    stats["plays_named" if named else "plays_new"] += bool(wrote)
(Path(a.out) / "manifest.json").write_text(json.dumps(rows, indent=1, default=float))
print(dict(stats), "frames", len(rows), "by how:", collections.Counter(r["how"] for r in rows),
      "by card:", collections.Counter(r["card"] for r in rows).most_common(12))
