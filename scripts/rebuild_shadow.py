"""Step 5 of the input rebuild (2026-11-16): Cole's shadow games re-read through the live path with tower / troop HP
and the new input block (bot77.live.extra_obs), so the rows match what live play now builds.

  uv run --no-sync python scripts/rebuild_shadow.py [--games 2026-11-13_175840 ...] [--no-hp] [--old-tracker]
  in:  data/cache/bc/shadow_t01{,_held}/cole_shadow_<gid>.npz, /Volumes/LaCie/bot-77-data/raw/shadow_frames/<gid>/
       <i:05d>.jpg (the screen at row i, saved by shadow_play.py from 2026-11-13)
  out: data/cache/bc/shadow_t02{,_held}/cole_shadow_<gid>.npz   (same split, rows, labels, weights; obs 13,977 + 437)

Per game, in row order: LiveReader (a16_live, our deck, fast mode, BOT77_LIVE_HP) reads frame i at the row's own time,
OppInputs (PlayTracker + SpellTracker) takes the read, the observation is built as SelfPlayPolicy._mirror builds it
(PlayTracker's plays for the cycle blocks, as live_play --play-tracker) and the block appended. Cole's label at row i
is his play: OppInputs.own_play(the card in that hand slot, the cell's centre) and LiveReader.played(slot), applied
before row i + 1 (live, his touch is drained on the frame after the decision it labels).

A row whose frame is missing (the frame writer drops frames when the drive falls behind) reuses the last read. Games
with no frames (11-10 .. 11-13 morning) keep their old base rows with a zero block (block-present 0).
--old-tracker --no-hp: the old live stack (OpponentTracker, no HP), to check the replay reproduces the saved rows.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))

from bot77.detect.yolo import Bot77Detector  # noqa: E402
from bot77.layout import Layout  # noqa: E402
from bot77.live.extra_obs import EXTRA_DIM, OppInputs  # noqa: E402
from bot77.live.rebuild import DECK, LiveObs, cards as load_cards, cell_to_ours, diff_sections  # noqa: E402
from bot77.live.state import LiveReader, OpponentTracker  # noqa: E402
from bot77.models import adopted_path  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402

FRAMES = Path("/Volumes/LaCie/bot-77-data/raw/shadow_frames")
SRC = {"shadow_t01": "shadow_t02", "shadow_t01_held": "shadow_t02_held"}


def rebuild(src: Path, frames: Path, cards: list[dict], det, hp: bool, old_tracker: bool) -> tuple[dict, dict]:
    z = dict(np.load(src, allow_pickle=True))
    old, card, cell, tt = z["obs"], z["card"], z["cell"], z["t"]
    by_name = {c["name"]: c["id"] for c in cards}
    reader = LiveReader(Layout.load("a16_live"), cards, det, deck=[by_name[n] for n in DECK.split(",")])
    reader.fast = True
    reader.read_hp = hp
    opp = OpponentTracker() if old_tracker else OppInputs(cards, DECK.split(","))
    lo = LiveObs(cards)
    rows, blocks = [], []
    st, n_missing, n_none, pending = None, 0, 0, None
    for i in range(len(tt)):
        t = float(tt[i])
        if pending is not None:  # Cole's play at the row before: his touch, drained now
            name, slot, x, y, t0 = pending
            if not old_tracker:
                opp.own_play(name, t, x, y)
            reader.played(slot, t0)
            pending = None
        im = cv2.imread(str(frames / f"{i:05d}.jpg"))
        if im is None:
            n_missing += 1
            if st is not None:
                st.t = t
        else:
            st = reader.read(im, t)
            if st.in_battle:
                opp.update(st)
        o = lo.obs(st, opp.plays) if st is not None and st.in_battle else None
        if o is None:  # the live loop never decided here; keep the old row's base so the rows line up
            n_none += 1
            o = old[i]
        rows.append(o)
        blocks.append(np.zeros(EXTRA_DIM, np.float32) if old_tracker else opp.block(t))
        if card[i] >= 0 and card[i] < 4:
            # the card in that slot as the saved row's hand said (what Cole's label indexed)
            e = hand_engine(old[i], int(card[i]))
            name = sc.engine_cards()[e]["name"] if e is not None else None
            x, y = cell_to_ours(int(cell[i])) if cell[i] >= 0 else (None, None)
            if name:
                pending = (name, int(card[i]), x, y, t)
    new = np.stack(rows).astype(np.float16)
    blk = np.stack(blocks).astype(np.float16)
    stats = {"rows": len(tt), "frames_missing": n_missing, "rows_not_decided": n_none,
             "base_diff_cells": diff_sections(old, new), "rows_any_base_diff": int((np.abs(
                 old.astype(np.float32) - new.astype(np.float32)) > 1e-3).any(1).sum()),
             "block_nonzero_per_row": round(float((blk != 0).sum(1).mean()), 1),
             "rows_with_spell_slot": int((blk[:, 377] > 0).sum()),
             "rows_hand_named": int((blk[:, 3] > 0).sum()),
             "opp_plays": len(opp.plays)}
    out = {k: v for k, v in z.items() if k != "obs"}
    out["obs"] = np.concatenate([new, blk], 1)
    if not old_tracker:
        out["opp_plays"] = np.array(json.dumps([{"t": round(p["t"], 2), "card": p["card"]} for p in opp.plays]))
    out["rebuilt"] = np.array(f"live path 2026-11-16: LiveReader a16_live fast, hp={hp}, "
                              f"{'OpponentTracker' if old_tracker else 'OppInputs'}, frames {frames}")
    return out, stats


def hand_engine(obs_row: np.ndarray, slot: int) -> int | None:
    from bot77.live.rebuild import HAND0, NCARD
    blk = obs_row[HAND0 + 5 + slot * NCARD: HAND0 + 5 + (slot + 1) * NCARD]
    nz = np.nonzero(blk)[0]
    return int(nz[0]) if len(nz) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", help="game ids (default: all)")
    ap.add_argument("--no-hp", action="store_true")
    ap.add_argument("--old-tracker", action="store_true")
    ap.add_argument("--out-suffix", default="", help="appended to the output dir names (tests)")
    ap.add_argument("--flat-cycle", action="store_true", help="BOT77_FLAT_OPP_CYCLE=1 (repro check)")
    a = ap.parse_args()
    if a.flat_cycle:
        os.environ["BOT77_FLAT_OPP_CYCLE"] = "1"
    cards = load_cards()
    det = Bot77Detector(Path(adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt")))
    summary = []
    for sdir, odir in SRC.items():
        out_dir = ROOT / "data/cache/bc" / (odir + a.out_suffix)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted((ROOT / "data/cache/bc" / sdir).glob("cole_shadow_*.npz")):
            gid = f.stem.replace("cole_shadow_", "")
            if a.games and gid not in a.games:
                continue
            frames = FRAMES / gid
            t0 = time.time()
            if frames.exists() and any(frames.iterdir()):
                out, stats = rebuild(f, frames, cards, det, not a.no_hp, a.old_tracker)
            else:  # no frames: the old rows, block-present 0
                z = dict(np.load(f, allow_pickle=True))
                out = {k: v for k, v in z.items() if k != "obs"}
                out["obs"] = np.concatenate([z["obs"], np.zeros((len(z["obs"]), EXTRA_DIM), np.float16)], 1)
                out["rebuilt"] = np.array("no frames: old rows, zero block (block-present 0)")
                stats = {"rows": len(z["obs"]), "no_frames": True}
            np.savez_compressed(out_dir / f.name, **out)
            stats.update(game=gid, split=sdir, s=round(time.time() - t0, 1))
            summary.append(stats)
            print(json.dumps(stats), flush=True)
        (out_dir / "rebuild_summary.json").write_text(json.dumps([s for s in summary if s["split"] == sdir], indent=1))


if __name__ == "__main__":
    main()
