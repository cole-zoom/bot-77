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

FUTURE_S = 15.0  # a reading must not be contradicted by a higher one this soon after
REREAD_S = 5.0  # read a bar again at least this often, even if it looks unchanged (a misread must not stick)
KING_PER_PRINCESS = 6144 / 3584  # cr-api-data KingTower / PrincessTower hitpoints, every level
BARS = {"tower-bar": "princess", "dagger-duchess-tower-bar": "princess", "king-tower-bar": "king"}
TEAM = {"friendly": "own", "enemy": "opp"}


LOOKALIKE = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "!": "1", "S": "5", "g": "9", "Z": "2"})


def read_number(img) -> tuple[int | None, str]:
    """The HP on a bar crop (OCR text; the number is chosen in clean())."""
    c = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res = ocrmac.OCR(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)), recognition_level="accurate").recognize()
    text = " ".join(r[0] for r in res)
    groups = number_groups(text)
    return (max(groups) if groups else None), text  # clean() picks among them against the tower's current HP


def number_groups(text: str) -> list[int]:
    """Every 3-5 digit group in a bar's text, look-alike letters read as digits. The level badge with the crown
    icon reads as 3 digits too ("121", "131"), left or right of the number depending on the replay."""
    return [int(g) for g in re.findall(r"\d+", (text or "").translate(LOOKALIKE)) if 3 <= len(g) <= 5]


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
    last_ocr_t: dict[str, float] = {}
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
            fresh = not (name in last_sig and np.abs(sig - last_sig[name]).mean() < 4.0
                         and t - last_ocr_t[name] < REREAD_S)
            if fresh:
                hp, text = read_number(crop)
                last_sig[name], last_read[name], last_ocr_t[name] = sig, (hp, text), t
            else:
                hp, text = last_read[name]  # unchanged bar: same reading
            raw.append({"t": t, "tower": name, "hp_raw": hp, "raw": text, "fresh": fresh})
    return clean(match["match_id"], raw)


def clean(match_id: str, raw: list[dict]) -> list[dict]:
    """Accept a fresh reading when it is <= the last accepted one and no reading of that tower in the next 15 s is
    higher: a dropped digit (457 for 4574) is followed by a higher read, a junk number is above the last accepted value.
    A cached reading (unchanged bar) shares its fresh reading's verdict. Each side has its own max (the commonest
    princess reading of that side in the first 30 s): the two players' towers can be different levels."""
    rows = []
    by_tower: dict[str, list[dict]] = {}
    for r in sorted(raw, key=lambda r: r["t"]):
        by_tower.setdefault(r["tower"], []).append(r)
    t_first = min((r["t"] for r in raw), default=0.0)
    caps = {}
    for side in ("own", "opp"):
        early = [max(number_groups(r["raw"]), default=None) for r in raw if r["tower"].startswith(side)
                 and not r["tower"].endswith("king") and r["t"] < t_first + 30]
        early = [e for e in early if e]
        caps[side] = Counter(early).most_common(1)[0][0] if early else None
    for name, rs in by_tower.items():
        side = name[:3]
        cap = caps[side]
        ceil0 = (cap * KING_PER_PRINCESS if name.endswith("king") else cap) if cap else 10**6
        for r in rs:  # the HP: the largest group not above the max (the badge is a small number)
            gs = [g for g in number_groups(r["raw"]) if g <= ceil0]
            r["hp_raw"] = max(gs) if gs else None
        mx = (cap * KING_PER_PRINCESS if name.endswith("king") else cap) if cap else None
        accepted = cap if (cap and not name.endswith("king")) else None
        verdict = False
        for i, r in enumerate(rs):
            hp = r["hp_raw"]
            if r.get("fresh", True):
                # HP only goes down: a reading is wrong if any plausible reading in the next FUTURE_S is higher
                # (a dropped digit, "457" for 4574, can repeat for a few seconds before the true value is back)
                future = [q["hp_raw"] for q in rs[i + 1:] if q["hp_raw"] is not None and q["t"] - r["t"] <= FUTURE_S]
                ceiling = accepted if accepted is not None else (mx + 1 if mx else 10**6)
                verdict = hp is not None and hp <= ceiling and all(f <= hp for f in future)
            ok = verdict and hp is not None and hp <= (accepted if accepted is not None else 10**6)
            if ok:
                accepted = hp
            rows.append({"match_id": match_id, "t": r["t"], "tower": name, "hp": accepted, "raw": r["raw"],
                         "accepted": ok, "max_hp": float(mx) if mx else None,
                         "frac": (accepted / mx) if (mx and accepted is not None) else None})
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
    ap.add_argument("--reclean", action="store_true", help="re-parse the cached OCR text (no OCR) before --merge")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.reclean:  # re-parse every cached match's OCR text with the current rules (no OCR)
        for f in sorted(CACHE.glob("*.json")):
            rows = json.loads(f.read_text())
            raw = [{"t": r["t"], "tower": r["tower"], "raw": r["raw"], "hp_raw": None} for r in rows]
            f.write_text(json.dumps(clean(f.stem, raw)))
        print(f"re-cleaned {len(list(CACHE.glob('*.json')))} matches")
        return
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
