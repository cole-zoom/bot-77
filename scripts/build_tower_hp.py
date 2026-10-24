"""Tower HP read off the replay videos (docs/simulator.md S2) -> LanceDB `tower_hp`.

  uv run python scripts/build_tower_hp.py [--sources replay_tv,replay_bl,replay_bot] [--splits train,dev,heldback]
                                          [--games N] [--step 1.0]

The HP number is printed on each tower's bar (the screen; OCR with Apple Vision via ocrmac, on this Mac). The bars are
found by the unit detector (`tower-bar`, `king-tower-bar`, `dagger-duchess-tower-bar`; run v5l_e15). Frames every
`step` seconds come from ffmpeg (hardware decoding); each bar is cropped from the nearest detector frame's box; a crop is read again only when it changed. A value is accepted when it is not above the
last accepted value (tower HP only goes down) and the next reading is not above it. A princess tower's max is the
commonest princess reading in the first 30 s (towers start full); a king's is the princess max x 6144 / 3584 (game files: the King / Princess
ratio is the same at every level), since the king shows no number until damaged.

Rows (bottom player's frame; 'own' = bottom): match_id, t (video s), tower (own_king, own_l, own_r, opp_*), hp,
max_hp, frac, raw (the OCR text), accepted (bool). Each match is cached in data/cache/tower_hp/ (shards run in
parallel); `--merge` writes them all into the table.
"""
import argparse
import bisect
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ocrmac import ocrmac
from PIL import Image

from bot77.sim import store

KING_PER_PRINCESS = 6144 / 3584  # cr-api-data KingTower / PrincessTower hitpoints, every level
BARS = {"tower-bar": "princess", "dagger-duchess-tower-bar": "princess", "king-tower-bar": "king"}
TEAM = {"friendly": "own", "enemy": "opp"}


def read_number(img) -> tuple[int | None, str]:
    """The HP on a bar crop: the longest 3-5 digit group (the 1-2 digit level badge is skipped)."""
    c = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res = ocrmac.OCR(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), recognition_level="accurate").recognize()
    text = " ".join(r[0] for r in res)
    groups = [g for g in re.findall(r"\d+", text.replace("O", "0").replace("o", "0")) if 3 <= len(g) <= 5]
    return (int(max(groups, key=len)) if groups else None), text


def signature(img) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (48, 12)).astype(np.float32)


def bars_by_time(match_id: str) -> dict[float, list[dict]]:
    rows = store.read("detections", f"match_id = '{match_id}' AND run_id = 'v5l_e15' AND cls IN "
                      "('tower-bar', 'king-tower-bar', 'dagger-duchess-tower-bar')")
    out: dict[float, list[dict]] = {}
    for r in rows:
        out.setdefault(round(r["t"], 2), []).append(r)
    return out


def tower_name(bar: dict, width: int) -> str | None:
    team = TEAM.get(bar["team"])
    if team is None:
        return None
    if BARS[bar["cls"]] == "king":
        return f"{team}_king"
    return f"{team}_{'l' if (bar['x0'] + bar['x1']) / 2 < width / 2 else 'r'}"


def frames_1fps(path: str, width: int, height: int, step: float):
    """(t, frame) every `step` s, hardware-decoded by ffmpeg (VideoToolbox): much faster than seeking with OpenCV."""
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", path, "-vf", f"fps={1 / step}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        k = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield k * step, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1


def process(match: dict, video: dict, step: float) -> list[dict]:
    bars = bars_by_time(match["match_id"])
    if not bars:
        return []
    ts = sorted(bars)
    width, height = int(video["width"]), int(video["height"])
    last_sig: dict[str, np.ndarray] = {}
    last_read: dict[str, tuple] = {}
    raw = []
    for t, frame in frames_1fps(video["path"], width, height, step):
        if t < ts[0] - 1 or t > ts[-1] + 1 or (match.get("t_end") and t > match["t_end"] - 1.0):
            continue  # nothing before the bars appear, and nothing in the match-end animation
        i = bisect.bisect_left(ts, t)
        near = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))
        if abs(ts[near] - t) > 1.0:  # bars come from the detector frame nearest t (bars don't move)
            continue
        for b in bars[ts[near]]:
            name = tower_name(b, width)
            if name is None:
                continue
            x0, y0, x1, y1 = (int(round(b[k])) for k in ("x0", "y0", "x1", "y1"))
            crop = frame[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
            if crop.size == 0:
                continue
            sig = signature(crop)
            if name in last_sig and np.abs(sig - last_sig[name]).mean() < 4.0:
                hp, text = last_read[name]  # unchanged bar: same reading
            else:
                hp, text = read_number(crop)
                last_sig[name], last_read[name] = sig, (hp, text)
            raw.append({"t": t, "tower": name, "hp_raw": hp, "raw": text})
    return clean(match["match_id"], raw)


def clean(match_id: str, raw: list[dict]) -> list[dict]:
    """Accept a reading when it is <= the last accepted one and the tower's next reading is <= it: a dropped digit
    (480 for 4802) is followed by a higher read, a junk number is above the last accepted value. Under steady fire the
    number changes every read, so two equal reads in a row can't be required."""
    rows, accepted, maxes = [], {}, {}
    by_tower: dict[str, list[dict]] = {}
    for r in sorted(raw, key=lambda r: r["t"]):
        by_tower.setdefault(r["tower"], []).append(r)
    # the princess max: the commonest princess reading in the first 30 s (towers start full); caps every reading
    t_first = min((r["t"] for r in raw), default=0.0)
    early = [r["hp_raw"] for r in raw if r["hp_raw"] and not r["tower"].endswith("king") and r["t"] < t_first + 30]
    cap = Counter(early).most_common(1)[0][0] if early else 10**6
    for name in by_tower:
        if not name.endswith("king"):
            accepted[name] = cap
            maxes[name] = cap
    for name, rs in by_tower.items():
        for i, r in enumerate(rs):
            hp = r["hp_raw"]
            nxt = next((q["hp_raw"] for q in rs[i + 1:] if q["hp_raw"] is not None), None)
            ceiling = accepted.get(name, cap * KING_PER_PRINCESS + 1)
            ok = hp is not None and hp <= ceiling and (nxt is None or nxt <= hp)
            if ok:
                accepted[name] = hp
            rows.append({"match_id": match_id, "t": r["t"], "tower": name, "hp": accepted.get(name), "raw": r["raw"],
                         "accepted": ok})
    pmax = max(maxes.values()) if maxes else None
    for r in rows:
        mx = (pmax * KING_PER_PRINCESS) if r["tower"].endswith("king") and pmax else maxes.get(r["tower"], pmax)
        r["max_hp"] = float(mx) if mx else None
        r["frac"] = (r["hp"] / mx) if (mx and r["hp"] is not None) else None
    return sorted(rows, key=lambda r: (r["t"], r["tower"]))


CACHE = Path("data/cache/tower_hp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="replay_tv,replay_bl,replay_bot")
    ap.add_argument("--splits", default="train,dev,heldback")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th game")
    ap.add_argument("--merge", action="store_true", help="write every cached match into the tower_hp table")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rows = [r for f in sorted(CACHE.glob("*.json")) for r in json.loads(f.read_text())]
        store.write("tower_hp", rows, mode="overwrite")
        print(f"tower_hp: {len(rows)} rows from {len(list(CACHE.glob('*.json')))} matches")
        return
    splits, sources = args.splits.split(","), args.sources.split(",")
    assert not {"test", "heldout"} & set(splits), "never on test / held-out"
    q = " OR ".join(f"source = '{s}'" for s in sources)
    games = sorted((m for m in store.read("matches", f"({q}) AND pov_side = 'bottom'") if m["split"] in splits),
                   key=lambda m: m["match_id"])
    if args.games:
        games = games[: args.games]
    i_shard, n_shard = (int(x) for x in args.shard.split("/"))
    games = games[i_shard::n_shard]
    t0 = time.time()
    for i, m in enumerate(games):
        out = CACHE / f"{m['match_id']}.json"
        if out.exists():
            continue
        v = store.read("videos", f"video_id = '{m['video_id']}'")
        if not v:
            continue
        rows = process(m, v[0], args.step)
        out.write_text(json.dumps(rows))
        print(f"[{args.shard}] {i + 1}/{len(games)} {m['match_id']}: {len(rows)} readings, "
              f"{sum(r['accepted'] for r in rows)} accepted, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
