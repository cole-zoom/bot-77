"""The live HP readers on recorded live games: troop HP bars found, a by-eye sheet, and the added time per frame.

  uv run --no-sync python scripts/eval_unit_hp.py [--games 4] [--fps 2] [--sheet 40]

Runs the live reader (LiveReader, a16_live, the adopted live detector, fast mode, read_hp on) over each recording
(data/raw/live_games/<game>/game.mkv). Per detector frame: every unit (not a bar / badge / tower) and whether a bar
was paired to it (hp_frac), every `bar` detection and whether it went to a unit. Timing: the tower reader's full read
(every TowerHP.every s) and attach_unit_hp per detector frame. --sheet: a contact sheet of that many units with a
bar (crop with the read fraction printed) for checking by eye, in --out.
"""
import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import cv2
import lancedb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECK = "Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="bot_2026-11-1[45]_*")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--sheet", type=int, default=40)
    ap.add_argument("--out", default="/Volumes/LaCie/bot-77-data/unit_hp_eval")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from bot77.detect.yolo import Bot77Detector
    from bot77.layout import Layout
    from bot77.live import state as S
    from bot77.models import adopted_path
    from bot77.readers.tower_hp import TowerHP
    from bot77.readers.unit_hp import bar_of
    from bot77.video import iter_frames

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = lancedb.connect(str(ROOT / "data/lancedb")).open_table("cards").search().where("kind = 'deck_card'") \
        .limit(1000).to_arrow().to_pylist()
    by_name = {c["name"]: c["id"] for c in cards}
    det = Bot77Detector(Path(adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt")))
    games = sorted(glob.glob(str(ROOT / "data/raw/live_games" / a.glob / "game.mkv")))
    random.seed(a.seed)
    games = random.sample(games, min(a.games, len(games)))
    t_unit, t_tower, n_units, n_hp, n_bars, n_bars_used = [], [], 0, 0, 0, 0
    by_card: dict[str, list[int]] = {}
    samples = []
    timed = S.attach_unit_hp

    def spy(frame, units):
        t0 = time.perf_counter()
        timed(frame, units)
        t_unit.append(time.perf_counter() - t0)

    S.attach_unit_hp = spy
    for v in games:
        game = Path(v).parent.name
        reader = S.LiveReader(Layout.load("a16_live"), cards, det, deck=[by_name[n] for n in DECK.split(",")])
        reader.fast, reader.read_hp = True, True
        last = None
        tw = TowerHP(every=0.0)
        for t, im in iter_frames(Path(v), a.fps):
            st = reader.read(im, t)
            if not st.in_battle:
                continue
            t0 = time.perf_counter()
            tw.update(im, t, st.elapsed_s)
            t_tower.append(time.perf_counter() - t0)
            if st.units is last or (last is not None and st.units == last):
                continue
            last = st.units
            bars = [u for u in st.units if u.cls == "bar"]
            body = [u for u in st.units if u.cls not in ("bar", "bar-level") and "tower" not in u.cls and u.card]
            n_bars += len(bars)
            got = sum(u.hp_frac is not None for u in body)
            n_units += len(body)
            n_hp += got
            n_bars_used += got
            for u in body:
                by_card.setdefault(f"{u.card} ({u.team})", [0, 0])[0] += 1
                by_card[f"{u.card} ({u.team})"][1] += u.hp_frac is not None
            picked = [u for u in body if u.hp_frac is not None and random.random() < 0.05]
            if picked:  # the bar each was read from: the same pairing again on this frame
                from bot77.readers.unit_hp import attach_bar_hp
                bd = [{"cls": b.cls, "team": b.team, "box": b.box} for b in st.units if b.cls in ("bar", "bar-level")]
                again = attach_bar_hp(im, [{"box": u.box, "team": u.team} for u in body], bd)
                for u, g in zip(body, again):
                    if u not in picked or "hp_bar" not in g:
                        continue
                    (x0, y0, x1, y1), badge = g["hp_bar"]
                    x0, y0, x1, y1 = (int(round(c)) for c in (x0, y0, x1, y1))
                    bx0 = int(round(badge[0])) if badge else x0
                    crop = im[max(0, y0 - 6): y1 + 6, max(0, bx0 - 6): x1 + 8].copy()
                    samples.append({"game": game, "t": round(t, 2), "card": u.card, "team": u.team,
                                    "hp": round(g["hp_frac"], 2), "crop": crop})
        print(game, f"units {n_units} with bar {n_hp}; bars {n_bars}", flush=True)
    random.shuffle(samples)
    samples = samples[: a.sheet]
    tiles = []
    for i, s in enumerate(samples):  # each bar (with its badge) 4x, the read fraction under it
        c = cv2.resize(s["crop"], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)[:, :540]
        c = cv2.copyMakeBorder(c, 0, 0, 0, 540 - c.shape[1], cv2.BORDER_CONSTANT)
        c = cv2.copyMakeBorder(c[:90], 0, max(0, 90 - c[:90].shape[0]) + 34, 0, 10, cv2.BORDER_CONSTANT)
        cv2.putText(c, f"#{i} read {s['hp']:.2f}", (4, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(c)
    rows = [np.hstack(tiles[k: k + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[k: k + 3])))
            for k in range(0, len(tiles), 3)]
    for k in range(0, len(rows), 6):
        cv2.imwrite(str(out / f"sheet_{k // 6}.jpg"), np.vstack(rows[k: k + 6]))
    (out / "sheet.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "crop"} | {"i": i}
                                                for i, s in enumerate(samples)], indent=1))
    ms = lambda xs: f"median {1000 * np.median(xs):.2f} ms, p95 {1000 * np.percentile(xs, 95):.2f} ms (n {len(xs)})"  # noqa: E731
    print(f"units {n_units}, with a bar read {n_hp} ({n_hp / max(1, n_units):.1%}); bar detections {n_bars}, "
          f"paired to a unit {n_bars_used} ({n_bars_used / max(1, n_bars):.1%})")
    for k, (n, h) in sorted(by_card.items(), key=lambda kv: -kv[1][0])[:16]:
        print(f"  {k:28s} {n:5d}  bar {h / n:.0%}")
    print("tower read (all six, uncached):", ms(t_tower))
    print("attach_unit_hp per detector frame:", ms(t_unit))


if __name__ == "__main__":
    main()
