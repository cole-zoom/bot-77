"""Cut an in-game hand-art override (data/cards/art_overrides/<Card>__<form>.png, bot77.readers.hand) from a frame
where the reader can't read a card, then check it on other frames (decisions.md, 2026-10-15: the replay viewer draws
Mirror, Lightning and the new Bandit art differently from the API icons, so the opponent hand missed them).

The art sits where readable normal cards match in the same slot, so the box is taken from the best normal-card match
in a readable slot of the same frame (the lowest one: a selected card is drawn raised).

  uv run python scripts/hand_art_override.py cut <match_id> <t_video> <slot 1-4> "<Card>" [--form normal]
  uv run python scripts/hand_art_override.py check <match_id> <t_video> <slot 1-4>   # best reads with current templates
"""
import argparse
from pathlib import Path

import cv2
import lancedb

from bot77.layout import Layout
from bot77.readers import hand as H
from bot77.video import iter_frames


def frame_and_reader(match_id, t):
    db = lancedb.connect("data/lancedb")
    m = db.open_table("matches").search().where(f"match_id = '{match_id}'").to_arrow().to_pylist()[0]
    v = db.open_table("videos").search().where(f"video_id = '{m['video_id']}'").to_arrow().to_pylist()[0]
    lay = Layout.load(v["layout_id"])
    cards = db.open_table("cards").search().limit(5000).to_arrow().to_pylist()
    _, f = next(iter(iter_frames(Path(v["path"]), 1, start=t, duration=0.5)))
    return f, lay, H.HandReader.from_cards(lay, cards)


def slot_gray(f, lay, region):
    s = lay.scale(f.shape[1])
    crop = lay.crop(f, region)
    if abs(s - 1.0) > 1e-3:
        crop = cv2.resize(crop, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def ranked(reader, gray, size="slot"):
    g = H.normalise(gray, H.LCN_SIGMA * reader.ui_scale)
    best = {}
    for t in reader.templates:
        for p in (t.slot,) if size == "slot" else t.next:
            if p.shape[0] > g.shape[0] or p.shape[1] > g.shape[1]:
                continue
            _, sc, _, loc = cv2.minMaxLoc(cv2.matchTemplate(g, p, cv2.TM_CCOEFF_NORMED))
            k = (t.name, t.form)
            if k not in best or sc > best[k][0]:
                best[k] = (sc, loc, p.shape)
    return sorted(best.items(), key=lambda kv: -kv[1][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cut", "check"]); ap.add_argument("match_id"); ap.add_argument("t", type=float)
    ap.add_argument("slot", type=int); ap.add_argument("card", nargs="?"); ap.add_argument("--form", default="normal")
    a = ap.parse_args()
    f, lay, reader = frame_and_reader(a.match_id, a.t)
    gray = slot_gray(f, lay, f"slot{a.slot}")
    if a.cmd == "check":
        for (name, form), (sc, loc, shp) in ranked(reader, gray)[:4]:
            print(f"  {name} ({form}) {sc:.2f}")
        return
    # the art box: where the best normal card matches in another, readable slot of this frame
    box = None
    for s in range(1, 5):
        if s == a.slot:
            continue
        (name, form), (sc, loc, shp) = ranked(reader, slot_gray(f, lay, f"slot{s}"))[0]
        # the lowest box: a selected card is drawn raised
        if form == "normal" and sc >= 0.6 and (box is None or loc[1] > box[1][1]):
            box = (sc, loc, shp, name)
    if box is None:
        raise SystemExit("no readable normal card in another slot of this frame to place the art box")
    _, (x, y), (h, w), ref = box
    patch = gray[y:y + h, x:x + w]
    u = reader.ui_scale
    patch = cv2.resize(patch, None, fx=1 / u, fy=1 / u, interpolation=cv2.INTER_AREA)  # stored at ui_scale 1
    out = H.ART_OVERRIDES / f"{a.card}__{a.form}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), patch)
    print(f"wrote {out} ({patch.shape[1]}x{patch.shape[0]}), box from {ref} at {x},{y}")


if __name__ == "__main__":
    main()
