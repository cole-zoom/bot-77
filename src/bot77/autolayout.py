"""Derive a layout for a new capture by finding Alina's HUD in it.

Patches of Alina's calibrated HUD (timer box, tower HP bars, chat button, "Next:" label, elixir
drop) are template-matched into sample frames of the new video over a range of scales. The game UI
keeps its proportions but phones differ in safe-area insets, so the top HUD, the hand bar and the
arena between them shift independently. Fit: one horizontal scale + offset for everything, a vertical
offset for the top HUD, one for the hand bar, and a linear map for the arena in between. This is
exactly what was done by hand for Cole's iPhone (decisions.md, 2026-09-29).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bot77.calibrate import read_frame
from bot77.layout import LAYOUT_DIR
from bot77.video import probe

REF_VIDEO = Path("data/raw/uz4VVzGlOjE.webm")
REF_T = 36.5
REF_LAYOUT = "alina_portrait"
# name -> (x0, y0, x1, y1) in Alina's frame, and which band it anchors
ANCHORS = {
    "timer_box": ((905, 85, 1075, 195), "top"),
    "opp_left_bar": ((165, 395, 300, 420), "arena_top"),
    "my_left_bar": ((165, 1450, 300, 1478), "arena_bottom"),
    "chat": ((30, 2010, 150, 2110), "hand"),
    "next_label": ((40, 2155, 150, 2195), "hand"),
    "elixir_drop": ((250, 2240, 300, 2320), "hand"),
}
# where each anchor may be, as fractions of the game area (x0, y0, x1, y1): the tower HP bars repeat
# four times, so the opponent's left one must be searched for in the top-left only
WINDOWS = {
    "timer_box": (0.55, 0.0, 1.0, 0.2), "opp_left_bar": (0.0, 0.08, 0.5, 0.35),
    "my_left_bar": (0.0, 0.45, 0.5, 0.8), "chat": (0.0, 0.7, 0.35, 1.0), "next_label": (0.0, 0.75, 0.35, 1.0), "elixir_drop": (0.0, 0.8, 0.5, 1.0),
}
TOP = {"opp_name", "opp_clan", "opp_rating", "timer_label", "timer", "multiplier", "opp_king_hp"}
HAND = {"ability_left", "ability_right", "next", "elixir_number", "elixir_bar"} | \
       {f"slot{i}" for i in range(1, 5)} | {f"slot{i}_tab" for i in range(1, 5)}
MIN_SCORE = 0.6


def _match(gray: np.ndarray, patch: np.ndarray, scales) -> tuple[float, float, tuple[int, int]]:
    best = (-1.0, 1.0, (0, 0))
    for s in scales:
        t = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1]:
            continue
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED))
        if mx > best[0]:
            best = (mx, s, loc)
    return best


def find_anchors(video: Path, times: list[float], scales=np.arange(0.4, 1.15, 0.02),
                 game_box: tuple[int, int, int, int] | None = None) -> dict:
    """`game_box` (x0, y0, x1, y1): where the game sits in the frame (default: the whole frame)."""
    ref = cv2.cvtColor(read_frame(REF_VIDEO, REF_T), cv2.COLOR_BGR2GRAY)
    found: dict[str, list] = {}
    for t in times:
        gray_full = cv2.cvtColor(read_frame(video, t), cv2.COLOR_BGR2GRAY)
        gx0, gy0, gx1, gy1 = game_box or (0, 0, gray_full.shape[1], gray_full.shape[0])
        # only in-game frames: the match timer must be on screen (menus fool the other anchors)
        (tx0, ty0, tx1, ty1), _ = ANCHORS["timer_box"]
        wx0, wy0, wx1, wy1 = WINDOWS["timer_box"]
        tw = gray_full[int(gy0 + wy0 * (gy1 - gy0)):int(gy0 + wy1 * (gy1 - gy0)), int(gx0 + wx0 * (gx1 - gx0)):int(gx0 + wx1 * (gx1 - gx0))]
        if _match(tw, ref[ty0:ty1, tx0:tx1], scales)[0] < MIN_SCORE:
            continue
        for name, ((x0, y0, x1, y1), _) in ANCHORS.items():
            wx0, wy0, wx1, wy1 = WINDOWS[name]
            ox0, oy0 = int(gx0 + wx0 * (gx1 - gx0)), int(gy0 + wy0 * (gy1 - gy0))
            gray = gray_full[oy0:int(gy0 + wy1 * (gy1 - gy0)), ox0:int(gx0 + wx1 * (gx1 - gx0))]
            score, s, (lx, ly) = _match(gray, ref[y0:y1, x0:x1], scales)
            lx, ly = lx + ox0, ly + oy0
            if score >= MIN_SCORE:
                found.setdefault(name, []).append(((x0 + x1) / 2, (y0 + y1) / 2,
                                                   lx + (x1 - x0) * s / 2, ly + (y1 - y0) * s / 2, s, score))
    # median over frames per anchor
    return {n: tuple(np.median(np.array(v), axis=0)) for n, v in found.items()}


UNIQUE = ("timer_box", "chat", "next_label", "elixir_drop")  # appear once on screen
# Tower HP bars as colour, not templates (templates depend on the arena skin): the opponent's
# left princess tower bar is a long red bar top-left, ours a long blue bar lower-left.
BARS = {"opp_left_bar": ((232, 408), "red", (0.0, 0.08, 0.5, 0.35)),
        "my_left_bar": ((232, 1464), "blue", (0.0, 0.45, 0.5, 0.8))}


def find_bars(video: Path, times: list[float], scale: float, game_box=None, expect: dict | None = None) -> dict:
    """`expect`: name -> rough (x, y) predicted from the HUD fit; the nearest bar-shaped candidate wins
    (red arenas and pink name text also produce red bar-shaped blobs)."""
    found: dict[str, list] = {}
    for t in times:
        f = read_frame(video, t)
        gx0, gy0, gx1, gy1 = game_box or (0, 0, f.shape[1], f.shape[0])
        for name, ((rx, ry), colour, (wx0, wy0, wx1, wy1)) in BARS.items():
            ox, oy = int(gx0 + wx0 * (gx1 - gx0)), int(gy0 + wy0 * (gy1 - gy0))
            hsv = cv2.cvtColor(f[oy:int(gy0 + wy1 * (gy1 - gy0)), ox:int(gx0 + wx1 * (gx1 - gx0))], cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            hue = ((h < 8) | (h > 168)) if colour == "red" else ((h >= 95) & (h <= 118))
            n, _, st, cen = cv2.connectedComponentsWithStats((hue & (s > 120) & (v > 120)).astype(np.uint8))
            bars = [(st[i][2], cen[i]) for i in range(1, n)
                    if 55 * scale <= st[i][2] <= 140 * scale and 6 * scale <= st[i][3] <= 30 * scale and st[i][2] > 3 * st[i][3]]
            if bars:
                if expect and name in expect:
                        ex, ey = expect[name]
                        w, (cx, cy) = min(bars, key=lambda b: abs(b[1][0] + ox - ex) + abs(b[1][1] + oy - ey))
                else:
                    w, (cx, cy) = max(bars, key=lambda b: b[0])
                # the bar's right part empties with damage: use its left end + the full-bar half width
                left = cx - w / 2 + ox
                found.setdefault(name, []).append((rx, ry, left + 60 * scale, cy + oy, scale, 1.0))
    return {n: tuple(np.median(np.array(v), axis=0)) for n, v in found.items() if len(v) >= 2}


def derive_layout(video: Path, layout_id: str, times: list[float], description: str = "",
                  game_box: tuple[int, int, int, int] | None = None) -> dict:
    # pass 1: the UI scale from anchors that appear once; pass 2: the repeated tower HP bar,
    # searched only near that scale (it otherwise matches tiny look-alikes)
    a = {k: v for k, v in find_anchors(video, times, game_box=game_box).items() if k in UNIQUE}
    if not a:
        raise RuntimeError(f"no HUD anchors found in {video.name}")
    s0 = float(np.median([v[4] for v in a.values()]))
    # rough bar positions from the HUD fit: the opponent's bar hangs off the top HUD, ours off the hand bar
    t_dy = a["timer_box"][3] - a["timer_box"][1] * s0
    h = [a[n] for n in ("chat", "next_label", "elixir_drop") if n in a]
    h_dy = float(np.median([v[3] - v[1] * s0 for v in h])) if h else t_dy
    x_off = float(np.median([v[2] - v[0] * s0 for v in a.values()]))
    expect = {n: (x_off + rx * s0, (t_dy if n == "opp_left_bar" else h_dy) + ry * s0) for n, ((rx, ry), _, _) in BARS.items()}
    a.update(find_bars(video, times, s0, game_box, expect))
    need = {"timer_box", "opp_left_bar", "chat"}
    if not need <= set(a):
        raise RuntimeError(f"anchors not found in {video.name}: {sorted(need - set(a))} (found {sorted(a)})")
    base = json.loads((LAYOUT_DIR / f"{REF_LAYOUT}.json").read_text())
    # horizontal: scale from the anchors' median template scale, offset from their x positions
    sx = float(np.median([v[4] for v in a.values()]))
    ox = float(np.median([v[2] - v[0] * sx for v in a.values()]))
    ty = a["timer_box"][3] - a["timer_box"][1] * sx
    hand = [a[n] for n in ("chat", "next_label", "elixir_drop") if n in a]
    hy = float(np.median([v[3] - v[1] * sx for v in hand]))
    low = a["my_left_bar"] if "my_left_bar" in a else a["chat"]  # arena spans the two tower HP bars
    (_, y_a, _, y_a2, _, _), (_, y_h, _, y_h2, _, _) = a["opp_left_bar"], low
    ka = (y_h2 - y_a2) / (y_h - y_a)
    ba = y_a2 - ka * y_a
    X = lambda x: round(ox + sx * x)
    regions = {}
    for n, (x0, y0, x1, y1) in base["regions"].items():
        if n in TOP:
            Y = lambda y: round(ty + sx * y)
        elif n in HAND:
            Y = lambda y: round(hy + sx * y)
        else:
            Y = lambda y: round(ba + ka * y)
        regions[n] = [X(x0), Y(y0), X(x1), Y(y1)]
    kx0, ky0, kx1, ky1 = base["katacr_arena"]
    info = probe(video)
    d = {
        "layout_id": layout_id, "description": description or f"Auto-derived from {video.name}",
        "frame_size": [info["width"], info["height"]], "regions": regions,
        "hand": {"ui_scale": round(sx, 4)},
        "katacr_arena": [X(kx0), round(ba + ka * ky0), X(kx1), round(ba + ka * ky1)],
        "notes": {"autolayout": {"anchors": {n: [round(v, 3) for v in vals] for n, vals in a.items()},
                                 "x": [round(sx, 4), round(ox, 1)], "top_dy": round(ty, 1), "hand_dy": round(hy, 1),
                                 "arena": [round(ka, 4), round(ba, 1)]}},
    }
    (LAYOUT_DIR / f"{layout_id}.json").write_text(json.dumps(d, indent=2))
    return d
