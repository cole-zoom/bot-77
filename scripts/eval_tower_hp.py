"""Score the live tower-HP bar reader (bot77.readers.tower_hp) against the HP numbers printed on the same bars.

  uv run --no-sync python scripts/eval_tower_hp.py [--games 20] [--fps 1] [--glob 'bot_2026-11-1[45]_*']

Each recorded live game (data/raw/live_games/<game>/game.mkv, layout a16_live) is decoded at --fps; every frame with
the battle clock gets the bar reader's raw fill per tower, and the number above / on the bar is OCR'd (Apple Vision,
offline only, read again only when the crop changed). Truth: a princess tower's fraction = its number / its side's
max (the commonest princess number in the first 30 s: towers start full); a King's = its number / its max, the King's
own highest number. Out: one JSON per game in --out (rows: t, tower, fill, present, hp), and the error summary.
"""
import argparse
import glob
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bot77.layout import Layout  # noqa: E402
from bot77.readers.clock import read_clock  # noqa: E402
from bot77.readers.tower_hp import BARS_A16, FIXED, TowerHP, read_bar  # noqa: E402

W, H = 1080, 1920
# OCR crops (x0, y0, x1, y1) around each tower's number, wide enough for every arena's offset
NUM = {"own_l": (225, 1222, 352, 1264), "own_r": (760, 1222, 890, 1264), "own_king": (480, 1462, 645, 1508),
       "opp_l": (225, 318, 352, 380), "opp_r": (760, 318, 890, 380), "opp_king": (480, 98, 645, 162)}


def frames(path: str, fps: float):
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", path, "-vf", f"fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        k = 0
        while True:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            k += 1


def ocr_number(img) -> int | None:
    from build_tower_hp import number_groups, read_number
    _, text = read_number(img)
    g = number_groups(text)
    return max(g) if g else None


def run_game(video: str, fps: float, layout: Layout, ocr: dict | None = None) -> list[dict]:
    """`ocr`: {(t, tower): hp} from an earlier run (the bar reader changed, the numbers didn't): no OCR."""
    rows, last_sig, last_hp = [], {}, {}
    tw = TowerHP(every=0.0)
    for t, im in frames(video, fps):
        c = read_clock(im, layout)
        if c.elapsed_s is None:
            continue
        state = tw.update(im, t, c.elapsed_s)
        for name in BARS_A16:
            band = tw.band.get(name)
            f, present = read_bar(im, name, band) if (band or name in FIXED) else (None, False)
            if ocr is not None:
                rows.append({"t": t, "elapsed": c.elapsed_s, "tower": name, "fill": f, "present": present,
                             "state": state[name], "hp": ocr.get((t, name))})
                continue
            x0, y0, x1, y1 = NUM[name]
            crop = im[y0:y1, x0:x1]
            sig = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 16)).astype(np.float32)
            if name not in last_sig or np.abs(sig - last_sig[name]).mean() >= 3.0:
                last_hp[name], last_sig[name] = ocr_number(crop), sig
            rows.append({"t": t, "elapsed": c.elapsed_s, "tower": name, "fill": f, "present": present,
                         "state": state[name], "hp": last_hp[name]})
    return rows


def truth(rows: list[dict]) -> dict:
    """tower -> max HP: a princess = its side's commonest number in the first 30 s (towers start full); a King = its
    highest number read on >= 2 frames (a one-frame OCR misread can't set it), not above 1.7x the princess max."""
    t0 = min((r["t"] for r in rows), default=0)
    out = {}
    for side in ("own", "opp"):
        early = [r["hp"] for r in rows if r["tower"].startswith(side) and not r["tower"].endswith("king")
                 and r["hp"] and r["t"] < t0 + 30]
        pmax = Counter(early).most_common(1)[0][0] if early else None
        out[f"{side}_l"] = out[f"{side}_r"] = pmax
        ks = Counter(r["hp"] for r in rows if r["tower"] == f"{side}_king" and r["hp"] and pmax
                     and r["hp"] <= 1.7 * pmax)
        out[f"{side}_king"] = max((k for k, n in ks.items() if n >= 2), default=None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="bot_2026-11-1[45]_*")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", default="/Volumes/LaCie/bot-77-data/tower_eval")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--reread", action="store_true", help="games already read: run the bar reader again, keep the OCR")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout.load("a16_live")
    games = sorted(glob.glob(str(ROOT / "data/raw/live_games" / a.glob / "game.mkv")))
    step = max(1, len(games) // a.games)
    games = games[::step][: a.games]
    errs: dict[str, list[float]] = {}
    gone = Counter()
    for v in games:
        gid = Path(v).parent.name
        f = out / f"{gid}.json"
        if a.reread and f.exists():
            ocr = {(r["t"], r["tower"]): r["hp"] for r in json.loads(f.read_text())}
            f.write_text(json.dumps(run_game(v, a.fps, layout, ocr)))
        elif not f.exists() and not a.summary_only:
            f.write_text(json.dumps(run_game(v, a.fps, layout)))
            print(gid, "read", flush=True)
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        mx = truth(rows)
        for r in rows:
            m = mx.get(r["tower"])
            if not m or r["hp"] is None or r["hp"] > m:
                continue
            true = r["hp"] / m
            errs.setdefault(r["tower"], []).append(abs(r["state"] - true))
        for name in ("own_l", "own_r", "opp_l", "opp_r"):  # destroyed: the reader's call vs the number's last frame
            rs = [r for r in rows if r["tower"] == name]
            t_gone = next((r["t"] for r in rs if r["state"] == 0.0), None)
            last_num = max((r["t"] for r in rs if r["hp"]), default=None)
            end = max((r["t"] for r in rs), default=0)
            if t_gone is not None:
                gone["called"] += 1
                gone["false"] += any(r["hp"] and r["t"] > t_gone + 2 for r in rs)
                gone["lag_s"] += t_gone - (last_num or t_gone)
            elif last_num is not None and last_num < end - 5:
                gone["missed?"] += 1  # the number vanished > 5 s before the end and the reader never called it
    for k, e in sorted(errs.items()):
        e = np.array(e)
        print(f"{k:9s} n {len(e):5d}  median |err| {np.median(e):.3f}  p90 {np.percentile(e, 90):.3f}  "
              f"<=0.05 {np.mean(e <= 0.05):.1%}  >0.2 {np.mean(e > 0.2):.1%}")
    print("princess destroyed:", dict(gone))


if __name__ == "__main__":
    main()
