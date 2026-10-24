"""Fill `matches.result` for the human replays and the bot's replays (docs/simulator.md S3: the win model needs it;
it was empty for all of them).

  uv run python scripts/fill_results.py [--dry-run] [--sheet N]

From the end screen first: "Winner!" is printed above the winner's name (OCR, Apple Vision on this Mac). Without
an end screen in the video, from the end of the match: which towers the detector still sees in the last seconds before `t_end` (v5l_e15,
tower classes at each tower's spot), so crowns per side; a destroyed king loses. Level on crowns (an overtime that ran
out): the side whose weakest standing tower has less HP (`tower_hp`, build_tower_hp.py) loses; equal or unknown -> draw
/ None. Both rows of a game get their side's result ('win' / 'loss' / 'draw').
`--sheet N` writes a contact sheet of N matches' last frames (local, gitignored) for a hand check.
"""
import argparse
import collections
from pathlib import Path

import cv2
import lancedb
import numpy as np
from ocrmac import ocrmac
from PIL import Image

from bot77.sim import store

PRINCESS = ("queen-tower", "cannoneer-tower", "dagger-duchess-tower")
TEAM = {"friendly": "bottom", "enemy": "top"}
END_WINDOW = (3.0, 0.5)  # seconds before t_end looked at
SHEET = Path("data/cache/results_sheet.jpg")


def final_towers(match_id: str, t_end: float) -> dict[str, bool] | None:
    """{'bottom_king': True, 'bottom_l': ..., 'top_r': ...}: alive at the end, or None without detections."""
    lo, hi = t_end - END_WINDOW[0], t_end - END_WINDOW[1]
    rows = store.read("detections", f"match_id = '{match_id}' AND run_id = 'v5l_e15' AND t >= {lo} AND t <= {hi} "
                      "AND (cls IN ('queen-tower', 'cannoneer-tower', 'dagger-duchess-tower', 'king-tower'))")
    frames = {round(r["t"], 2) for r in store.read(
        "detections", f"match_id = '{match_id}' AND run_id = 'v5l_e15' AND t >= {lo} AND t <= {hi}")}
    if not frames:
        return None
    seen = collections.Counter()
    for r in rows:
        side = TEAM.get(r["team"])
        if side is None:
            continue
        name = f"{side}_king" if r["cls"] == "king-tower" else f"{side}_{'l' if r['x'] < 9 else 'r'}"
        seen[(name, round(r["t"], 2))] = 1
    share = collections.Counter(name for name, _ in seen)
    names = [f"{s}_{k}" for s in ("bottom", "top") for k in ("king", "l", "r")]
    return {n: share[n] / len(frames) >= 0.3 for n in names}


def weakest(match_id: str, side: str, t_end: float, alive: dict) -> float | None:
    own = {"bottom": "own", "top": "opp"}[side]
    rows = store.read("tower_hp", f"match_id = '{match_id}' AND accepted = true AND t <= {t_end}")
    last = {}
    for r in sorted(rows, key=lambda r: r["t"]):
        last[r["tower"]] = r["hp"]
    hps = [last[f"{own}_{k}"] for k in ("l", "r", "king") if alive.get(f"{side}_{k}") and f"{own}_{k}" in last]
    return min(hps) if hps else None


def end_screen(m: dict) -> tuple[str, float] | None:
    """The bottom player's result from the end screen: 'Winner!' is printed above the winner's name, so above the
    'VS' when the top player won and below it when the bottom player did; 'Draw' for a draw. None if not found."""
    v = store.read("videos", f"video_id = '{m['video_id']}'")
    if not v:
        return None
    cap = cv2.VideoCapture(v[0]["path"])
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    t = max(0.0, m["t_end"] - 5.0)
    try:
        while t <= dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            t += 0.5
            if not ok:
                continue
            small = cv2.cvtColor(cv2.resize(f, (540, 960)), cv2.COLOR_BGR2RGB)
            words = ocrmac.OCR(Image.fromarray(small), recognition_level="accurate").recognize()
            win = [w for w in words if "winner" in w[0].lower()]
            vs = [w for w in words if w[0].strip().lower() == "vs"]
            if any("draw" in w[0].lower() for w in words) and vs:
                return "draw", t - 0.5
            if win and vs:  # Vision boxes: (x, y, w, h), y up from the bottom of the frame
                return ("loss" if win[0][2][1] > vs[0][2][1] else "win"), t - 0.5
    finally:
        cap.release()
    return None


def decide(m: dict) -> tuple[str | None, str]:
    """The bottom player's result and how it was decided."""
    res = end_screen(m)
    if res:
        m["_t_decided"] = res[1]
        return res[0], "end screen"
    alive = final_towers(m["match_id"], m["t_end"])
    if alive is None:
        return None, "no detections"
    if not alive["top_king"] and alive["bottom_king"]:
        return "win", "king"
    if not alive["bottom_king"] and alive["top_king"]:
        return "loss", "king"
    crowns_b = sum(not alive[f"top_{k}"] for k in ("l", "r"))
    crowns_t = sum(not alive[f"bottom_{k}"] for k in ("l", "r"))
    if crowns_b != crowns_t:
        return ("win" if crowns_b > crowns_t else "loss"), f"crowns {crowns_b}-{crowns_t}"
    wb, wt = weakest(m["match_id"], "bottom", m["t_end"], alive), weakest(m["match_id"], "top", m["t_end"], alive)
    if wb is None or wt is None:
        return None, f"crowns {crowns_b}-{crowns_t}, tower HP unknown"
    if wb == wt:
        return "draw", f"crowns {crowns_b}-{crowns_t}, weakest towers equal ({wb})"
    return ("win" if wb > wt else "loss"), f"crowns {crowns_b}-{crowns_t}, weakest tower {wb} vs {wt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--splits", default="train,dev,heldback,test,heldout")
    args = ap.parse_args()
    games = [m for m in store.read("matches", "source IN ('replay_tv', 'replay_bl', 'replay_bot') "
                                   "AND pov_side = 'bottom'") if m["split"] in args.splits.split(",")]
    out, how = {}, collections.Counter()
    for m in sorted(games, key=lambda m: m["match_id"]):
        res, why = decide(m)
        out[m["match_id"]] = (res, why, m)
        how[(res, why.split(",")[0].split(" ")[0])] += 1
    print(f"{len(games)} games:", dict(how))
    if args.sheet:
        rng = np.random.default_rng(77)
        pick = rng.choice(sorted(k for k, v in out.items() if v[0]), size=min(args.sheet, len(out)), replace=False)
        tiles = []
        for mid in pick:
            res, why, m = out[mid]
            v = store.read("videos", f"video_id = '{m['video_id']}'")[0]
            cap = cv2.VideoCapture(v["path"])
            cap.set(cv2.CAP_PROP_POS_MSEC, m.get("_t_decided", m["t_end"] - 0.7) * 1000)
            ok, f = cap.read()
            cap.release()
            if not ok:
                continue
            f = cv2.resize(f, (270, 480))
            cv2.putText(f, f"{mid[:22]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(f, f"bottom: {res}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(f, why[:34], (4, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            tiles.append(f)
            print(mid, res, why)
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        SHEET.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SHEET), sheet)
        print(f"sheet -> {SHEET}")
    if args.dry_run:
        return
    t = lancedb.connect(store.DB).open_table("matches")
    rows = []
    for mid, (res, why, m) in out.items():
        if res is None:
            continue
        other = {"win": "loss", "loss": "win", "draw": "draw"}[res]
        t.update(where=f"match_id = '{mid}'", values={"result": res})
        rows.append({"match_id": mid, "result": res, "how": why})
        if m["pair_match_id"]:
            t.update(where=f"match_id = '{m['pair_match_id']}'", values={"result": other})
            rows.append({"match_id": m["pair_match_id"], "result": other, "how": why})
    # its own table too: build_sources.py rebuilds `matches` from scratch and merges these back (2026-10-24: a rebuild
    # wiped the results once)
    store.write("match_results", rows, mode="overwrite")
    print(f"matches.result written for {sum(1 for v in out.values() if v[0])} games (both rows)")


if __name__ == "__main__":
    main()
