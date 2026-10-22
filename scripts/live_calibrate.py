"""B v0: derive the BlueStacks battle layout (run while a battle is on screen). Captures ~15 s of screenshots over
ADB, writes them as a short video, derives the layout with bot77.autolayout (the same fit used for creators' videos),
then checks it: the hand, elixir and clock read on the captured frames, and an image with the regions drawn.

  uv run python scripts/live_calibrate.py [--device 127.0.0.1:5555] [--layout bluestacks_live]
"""
import argparse, json, time
from pathlib import Path

import cv2
import lancedb

from bot77.autolayout import derive_layout
from bot77.detect.yolo import Bot77Detector
from bot77.layout import LAYOUT_DIR, Layout
from bot77.live.arena import fit_arena, tower_centres
from bot77.live.device import Device
from bot77.live.state import LiveReader

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="127.0.0.1:5555"); ap.add_argument("--layout", default="bluestacks_live")
ap.add_argument("--seconds", type=int, default=15)
a = ap.parse_args()
out = Path("data/raw/live_calib"); out.mkdir(parents=True, exist_ok=True)
dev = Device(a.device)
frames = []
for _ in range(a.seconds):
    im = dev.cap()
    if im is not None:
        frames.append(im)
    time.sleep(1.0)
video = out / f"{a.layout}.mp4"
w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (frames[0].shape[1], frames[0].shape[0]))
for f in frames:
    w.write(f)
w.release()
d = derive_layout(video, a.layout, [t + 0.5 for t in range(1, len(frames) - 1, 3)], description="BlueStacks live battle (B v0)")
print("layout written:", a.layout, "frame", d["frame_size"], "rough arena", d["katacr_arena"])
# the arena crop from the HUD fit is only ~50 px close: fit it to the princess towers (bot77.live.arena)
det = Bot77Detector(Path("data/models/bot77_detector/v6l/best.pt"))
towers = tower_centres(frames, Layout.load(a.layout), det)
if {"enemy_L", "enemy_R"} <= set(towers):
    arena, err = fit_arena(towers, frames[0].shape[1], d["frame_size"][0])
    d["katacr_arena"] = arena
    d.setdefault("notes", {})["arena_fit"] = {"towers": {k: [round(v, 1) for v in p] for k, p in towers.items()},
                                              "check_error_tiles": round(err, 3)}
    (LAYOUT_DIR / f"{a.layout}.json").write_text(json.dumps(d, indent=2))
    print(f"arena fitted to the towers: {arena} (our towers off by {err:.2f} tiles; > 0.3 means check by eye)")
else:
    print("WARNING: enemy princess towers not found; arena left from the HUD fit:", sorted(towers))
cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
names = {c["id"]: c["name"] for c in cards}
layout = Layout.load(a.layout)
r = LiveReader(layout, cards)
for k, f in enumerate(frames):
    st = r.read(f, float(k))
    print(f"  frame {k}: hand {[names.get(c) for c in st.hand]} next {names.get(st.next)} elixir {st.elixir} "
          f"clock {st.time_left_s} {st.phase} x{st.multiplier}")
check = frames[len(frames) // 2].copy()
for n, (x0, y0, x1, y1) in layout.regions.items():
    cv2.rectangle(check, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(check, n, (x0, y0 - 4), 0, 0.6, (0, 255, 255), 2)
ax0, ay0, ax1, ay1 = layout.katacr_arena
cv2.rectangle(check, (ax0, ay0), (ax1, ay1), (255, 0, 255), 3)
cv2.imwrite(str(out / f"{a.layout}_check.jpg"), check)
print("check image:", out / f"{a.layout}_check.jpg")
