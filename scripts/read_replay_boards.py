"""Step 5 of the input rebuild (2026-11-16): the pro replays' boards read through the live path, cached, so the new
pro shards (scripts/build_bc_live.py) are built from what live play would have read. Runs on the Mac (mps) or a
Linux CUDA box (scripts/pod/), no LanceDB needed: everything per match comes from a manifest.

  python scripts/read_replay_boards.py --manifest pod_job/manifest.json --videos pod_job/videos --out pod_job/reads \
      [--workers 6] [--device cuda:0] [--matches bl_2026-10-12_r02_m01 ...] [--seconds 60]

Per match (one video each; scripts/pod/make_manifest.py writes the manifest from the matches / videos / hud_states
tables): every frame at 10 fps from t_start to t_end goes through LiveReader.read_board (bot77.live.state), the live
code itself: fast mode (the detector every 0.2 s), troop HP bars on (BOT77_LIVE_HP); tower HP too where the live
tower reader has the layout's bar geometry (not replay_bottom: its frames are dimmed, see tower_hp.py). The clock and our
elixir, which read_board needs only to know a battle is on and for the tower reader's clock, come from the match's
offline HUD reads (hud_states, 0.1 s; the live clock reader is Apple-Vision OCR, macOS only). The hand isn't read:
the shards take both hands from hud_states, as before.

Out per match, reads/<match_id>.npz:
  det_t [n]          the read's video time (each detector run, every 0.2 s)
  det_cls, det_card  index into `classes` / `cards` (-1: no card)
  det_team           0 friendly, 1 enemy, 2 other
  det_conf, det_box [n, 4] (frame pixels x0 y0 x1 y1), det_xy [n, 2] (arena tiles, box centre, as Unit.x / y),
  det_hp             the HP bar's fraction (nan: no bar read)
  run_t              every detector run's time (a run can find nothing)
  tw_t, tw [m, 6]    tower HP fractions at every read where they changed (own_king own_l own_r opp_king opp_l
                     opp_r; nan: not read)
  meta               json: match, video, layout, fps, frames read, seconds taken
Every raw detection is kept (bars, badges, towers, clocks too): the "detector misses" table joins them with the
replays' true plays later without a second GPU pass.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOWERS = ["own_king", "own_l", "own_r", "opp_king", "opp_l", "opp_r"]
TEAMS = {"friendly": 0, "enemy": 1}
_W: dict = {}


def board_reader(layout_id: str, cards: list[dict], det):
    """LiveReader without the hand reader (its card art isn't needed here): read_board only."""
    from bot77.layout import Layout
    from bot77.live.state import LiveReader
    r = LiveReader.__new__(LiveReader)
    r.layout, r.cards, r.detector = Layout.load(layout_id), cards, det
    r.names = [c["name"] for c in cards]
    r.fast, r._units, r.read_hp = True, None, True
    # tower HP: the live reader only where its bar geometry is measured (bot77.readers.tower_hp.BARS_BY_LAYOUT); the
    # replays' frames are dimmed under its brightness threshold, so their shards keep the OCR'd tower HP
    from bot77.readers.tower_hp import BARS_BY_LAYOUT, TowerHP
    r.tower_hp = TowerHP(bars=BARS_BY_LAYOUT[layout_id]) if layout_id in BARS_BY_LAYOUT else _NoTowers()
    return r


class _NoTowers:
    def update(self, *a, **k):
        return None


def init(weights: str, device: str, cards_path: str) -> None:
    os.chdir(ROOT)  # layouts / assets are relative paths
    if device:
        os.environ["BOT77_DETECT_DEVICE"] = device
    import torch
    torch.set_num_threads(2)
    from bot77.detect.yolo import Bot77Detector
    _W["det"] = Bot77Detector(Path(weights))
    _W["cards"] = json.loads(Path(cards_path).read_text())


def read_match(job: tuple) -> str:
    m, video, out, seconds = job
    from bot77.live.state import LiveState
    from bot77.video import iter_frames
    t0 = time.time()
    cards = _W["cards"]
    names = [c["name"] for c in cards]
    reader = board_reader(m["layout_id"], cards, _W["det"])
    hud_t = np.asarray(m["hud_t"], np.float64)
    hud_m, hud_e = m["hud_match"], m["hud_elixir"]
    classes: dict[str, int] = {}
    D = {k: [] for k in ("t", "cls", "card", "team", "conf", "box", "xy", "hp")}
    run_t, tw_t, tw = [], [], []
    last_tw = None
    start = float(m["t_start"])
    dur = float(m["t_end"]) - start if m["t_end"] else None
    if seconds:
        dur = min(dur or seconds, seconds)
    n = 0
    for t, im in iter_frames(Path(video), 10.0, start, dur):
        n += 1
        i = int(np.searchsorted(hud_t, t, side="right")) - 1
        if i < 0 or hud_m[i] is None:
            continue
        el = hud_m[i] + (t - hud_t[i])  # the clock counted forward from the last HUD read, as fast mode does
        st = LiveState(t, [None] * 4, None, hud_e[i] if hud_e[i] is not None else 5.0, None, el, 1,
                       "regulation" if el < 180 else "overtime")
        prev = reader._units
        st = reader.read_board(im, t, st)
        if st.towers is not None:
            v = [st.towers.get(k, np.nan) for k in TOWERS]
            if last_tw is None or not np.allclose(v, last_tw, equal_nan=True):
                tw_t.append(t); tw.append(v); last_tw = v
        if reader._units is not prev and reader._units is not None and reader._units[0] == t:  # a detector run
            run_t.append(t)
            for u in st.units:
                D["t"].append(t)
                D["cls"].append(classes.setdefault(u.cls, len(classes)))
                D["card"].append(names.index(u.card) if u.card in names else -1)
                D["team"].append(TEAMS.get(u.team, 2))
                D["conf"].append(u.conf)
                D["box"].append(u.box or (np.nan,) * 4)
                D["xy"].append((u.x, u.y))
                D["hp"].append(np.nan if u.hp_frac is None else u.hp_frac)
    meta = {"match": m["match_id"], "video": Path(video).name, "layout": m["layout_id"], "fps": 10.0, "frames": n,
            "runs": len(run_t), "dets": len(D["t"]), "s": round(time.time() - t0, 1),
            "classes": sorted(classes, key=classes.get), "cards": "cards.json order (index into it)"}
    np.savez_compressed(
        out, det_t=np.array(D["t"], np.float32), det_cls=np.array(D["cls"], np.int16),
        det_card=np.array(D["card"], np.int16), det_team=np.array(D["team"], np.int8),
        det_conf=np.array(D["conf"], np.float16), det_box=np.array(D["box"], np.float32).reshape(-1, 4),
        det_xy=np.array(D["xy"], np.float32).reshape(-1, 2), det_hp=np.array(D["hp"], np.float16),
        run_t=np.array(run_t, np.float32), tw_t=np.array(tw_t, np.float32),
        tw=np.array(tw, np.float32).reshape(-1, 6), meta=np.array(json.dumps(meta)))
    return json.dumps(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos", required=True, help="the folder holding each manifest entry's `video` file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards", default=None, help="cards.json (default: next to the manifest)")
    ap.add_argument("--weights", default="data/models/bot77_detector/v6l/best.pt")
    ap.add_argument("--device", default="", help="cuda:0 on the pod; empty: mps")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--seconds", type=float, default=0.0, help="read only this much of each match (tests)")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    cards_path = a.cards or str(Path(a.manifest).parent / "cards.json")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in man["matches"]:
        if a.matches and m["match_id"] not in a.matches:
            continue
        f = out / f"{m['match_id']}.npz"
        if f.exists():  # resumable: a restarted run skips what's done
            continue
        jobs.append((m, str(Path(a.videos).resolve() / m["video"]), str(f), a.seconds))
    jobs.sort(key=lambda j: -(j[0]["t_end"] - j[0]["t_start"]))  # longest first: no straggler at the end
    print(f"{len(jobs)} matches to read -> {out}", flush=True)
    t0 = time.time()
    weights = str(Path(a.weights).resolve())
    with get_context("spawn").Pool(a.workers, initializer=init, initargs=(weights, a.device, cards_path)) as pool:
        for k, msg in enumerate(pool.imap_unordered(read_match, jobs)):
            print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f} s {msg}", flush=True)


if __name__ == "__main__":
    main()
