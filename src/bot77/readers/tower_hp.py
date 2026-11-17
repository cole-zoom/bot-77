"""Tower HP fractions read live off the six towers' HP bars (2026-11-16), by fill colour, no OCR.

The bars sit at fixed screen spots (layout a16_live, the phone at 1080x1920). Each bar is a bright fill from the left
(blue ours, pink theirs) over a dark track; the HP number is printed over (ours) or above (theirs) it, so the fill is
read on rows the digits don't reach. A column is fill when half its rows in the bar's band are bright in the team's
hue (S >= 80, V >= 170) or white (a just-hit flash); the fill runs from the bar's left end to where its dark track
starts. Their princess bars sit 15-20 px lower in some arenas: their band is located once per game in a taller
window. TowerHP keeps the per-game state (HP only goes down, destroyed towers, Kings woken). Measured 2026-11-16 on 20
recorded games against the printed numbers (scripts/eval_tower_hp.py; decisions.md).

King towers show no bar until damaged / woken: no bar = full. A princess tower without its bar for GONE_S is
destroyed (sticky until the next game). Geometry in source pixels of a 1080-wide frame; other widths scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# name -> (x_zero, x0, x1, y0, y1): the fill is read from x0 (the level badge's right edge) to x1 (its last column
# when full); x_zero is where an empty fill would end (under the badge). Fitted 2026-11-16 against the printed numbers
# on 20 recorded games (fill end vs HP / max: median residual 0.001-0.004). y0, y1: the rows read (ours, fixed) or
# the window their band is looked for in (their bars sit 15-20 px lower in some arenas).
BARS_A16 = {
    "own_l": (230, 234, 344, 1228, 1234),
    "own_r": (766, 770, 881, 1228, 1234),
    "own_king": (480, 486, 634, 1474, 1481),
    "opp_l": (230, 234, 344, 325, 405),
    "opp_r": (767, 770, 880, 325, 405),
    "opp_king": (480, 486, 634, 105, 185),
}
# BlueStacks (layout bluestacks_live, the bot's games up to 11-13; 2026-11-16, step 5): the same bars measured on its
# frames (princess bars at 20 s of bot_2026-10-31_013343, the Kings' fill rows on bot_2026-11-02_122958 36 s; x
# scales 1.05 from the phone's). Their windows as the phone's: the band is found in them.
BARS_BLUESTACKS = {
    "own_l": (214, 218, 333, 1190, 1196),
    "own_r": (780, 784, 899, 1190, 1196),
    "own_king": (477, 484, 640, 1448, 1454),
    "opp_l": (212, 216, 334, 232, 312),
    "opp_r": (778, 782, 900, 232, 312),
    "opp_king": (479, 486, 641, 30, 110),
}
# Not the replays (layout replay_bottom): their frames are dimmed (their pink fill reads V ~153, under the bright
# threshold), so the live tower reader doesn't apply; the pro shards keep the replays' OCR'd tower HP.
BARS_BY_LAYOUT = {"a16_live": BARS_A16, "bluestacks_live": BARS_BLUESTACKS}
FIXED = {"own_l", "own_r", "own_king"}
# OpenCV hue ranges. Their bar is pink 160-178 (fill) / 162-165 (track); not through red: some arenas' ground is
# orange-red (hue 4-8, S 190) and read as their bar (bot_2026-11-15_012028: 0.09 for 0.91; bot_2026-11-15_030845
# 135 s: a destroyed tower's empty band read 1.0), and so does the red pulse of the last seconds (hue 0-3)
TEAM_HUE = {"own": (90, 115), "opp": (155, 179)}
EMPTY = 0.01  # a princess bar with no fill left: standing, at a few HP
GAP = 3       # a King: non-fill columns allowed inside the fill run (antialiasing)
OCCLUDE = 40  # a princess: columns neither fill nor track (something drawn over the bar) skipped inside the run
GONE_S = 1.5  # a princess bar unseen this long: the tower is destroyed...
GONE_BELOW = 0.6  # ...if it was last read at or below this (bars vanish for 1-3 s in the start animation)
DROP_TOL = 0.45  # ...and a fall of more than this in one read (no card does that in 0.5 s) only on a second read
WAKE = 0.5  # a King's first bar must read at least this (before it wakes its band can catch a unit's bar)
DROP_S = 1.0  # ...held this long (2026-11-16: BlueStacks' red "No card selected" over their bar read 0.04 for 0.5 s,
              # and the full bar coming back counted as a rise, then as a destroyed tower, bot_2026-10-31_012417)
UP_TOL, UP_S = 0.05, 2.0  # HP only goes down: a read this much above the last counts only after UP_S of such reads
                          # (a destroyed tower's spot flashed pink for 5 s, read 0.41 -> 1.0, bot_2026-11-14_172646)
BAND_COVER = 0.85  # a row of their bar: this share of its columns fill or track
START_S = 4.0  # no reads in the first seconds of the clock: the bars draw in from the left (step 5, 2026-11-16: on
               # BlueStacks the half-drawn opp_r bar read 0.04 twice at 0-0.5 s and the tower was called destroyed for
               # the whole game, bot_2026-10-31_012417); nothing can damage a tower that early
DIM_FILL = 0.83  # a dark arena draws the bars dimmer (BlueStacks night arena, 2026-10-31 16:24 on: their full fill V ~153,
                 # track ~77, read all track = 0.01 for 0.76): the bright threshold per team is min(170, this x the
                 # princess fill's median V at its first read, when every tower is full); 223 on a normal arena -> 170
LEARN_BY_S = 20.0
BAND_MIN = 9  # rows: their bars are 12-16 tall; a 5-row band locked in the start animation read 0.05 for 0.97


def _masks(crop: np.ndarray, team: str, vmin: float = 170) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fill, team-coloured fill, track) per pixel. Fill is bright: the team's hue, or white (a bar flashes white
    for a frame when hit: bot_2026-11-14_150329 17 s read 0.07 for 0.63 before white counted)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = TEAM_HUE[team]
    hue = ((h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)) & (s >= 80)
    bright = v >= vmin
    return bright & (hue | (s < 70)), hue & bright, hue & ~bright


def _px(name: str, frame_w: int, bars: dict) -> tuple[int, int, int, int]:
    s = frame_w / 1080.0
    _, x0, x1, y0, y1 = bars[name]
    return int(round(x0 * s)), int(round(x1 * s)) + 1, int(round(y0 * s)), int(round(y1 * s))


def find_band(frame: np.ndarray, name: str, bars: dict = BARS_A16, vmin: float = 170) -> tuple[int, int] | None:
    """The rows of their bar in its search window (frame pixels): the tallest run of rows where >= BAND_COVER of the
    columns are the bar's colour (fill or track; a King's track is grey, so its fill alone), trimmed a row each side.
    None when the bar isn't there (a King not woken yet)."""
    X0, X1, Y0, Y1 = _px(name, frame.shape[1], bars)
    _, colour, track = _masks(frame[Y0:Y1, X0:X1], name[:3], vmin)
    cover = (colour | track).mean(1) if not name.endswith("king") else colour.mean(1)
    ok = cover >= (BAND_COVER if not name.endswith("king") else 0.5)
    best, run, start = (0, 0), 0, 0
    for i, v in enumerate(list(ok) + [False]):
        if v:
            start, run = (i, run + 1) if run == 0 else (start, run + 1)
        elif run:
            if run > best[1] - best[0]:
                best = (start, start + run)
            run = 0
    if best[1] - best[0] < BAND_MIN * frame.shape[1] / 1080.0:  # a bar half drawn (the start animation): not yet
        return None
    return Y0 + best[0] + 1, Y0 + best[1] - 1


def read_bar(frame: np.ndarray, name: str, band: tuple[int, int] | None = None,
             bars: dict = BARS_A16, vmin: float = 170) -> tuple[float | None, bool]:
    """(fill fraction or None if no fill, whether the bar — fill or dark track — is there) on rows `band` (frame
    pixels; default the bar's fixed rows)."""
    X0, X1, Y0, Y1 = _px(name, frame.shape[1], bars)
    if band is not None:
        Y0, Y1 = band
    crop = frame[Y0:Y1, X0:X1]
    if crop.size == 0:
        return None, False
    fill_px, _, track_px = _masks(crop, name[:3], vmin)
    half = max(1, crop.shape[0] // 2)
    fill = fill_px.sum(0) >= half
    track = track_px.sum(0) >= half
    # a King's track is grey (not the team's colour): its bar is there when any of it is fill (white when just hit)
    present = bool(fill.mean() + track.mean() >= 0.6) if not name.endswith("king") else bool(fill.any())
    head = np.flatnonzero(fill[: max(3, len(fill) // 8)])
    if not present:
        return None, False
    if len(head) == 0:
        # a princess bar that is all track stands at a few HP (bot_2026-11-15_002049 112-141 s: 63 HP, 30 s); a King
        # without fill isn't there
        return (EMPTY if not name.endswith("king") and track.mean() >= 0.8 else None), present
    end, gap, run_track = head[0], 0, 0
    for i in range(head[0], len(fill)):
        if fill[i]:
            end, gap, run_track = i, 0, 0
            continue
        gap += 1
        if name.endswith("king"):  # grey track: the fill ends at the first real gap
            if gap > GAP:
                break
            continue
        # a princess bar ends where its dark track starts; anything else (a unit's badge or bar drawn over it,
        # bot_2026-11-15_012028 38 s read 0.1 for 0.97) is skipped up to OCCLUDE columns
        run_track = run_track + 1 if track[i] else 0
        if run_track >= 3 or gap > OCCLUDE:
            break
    s = frame.shape[1] / 1080.0
    zero = bars[name][0] * s
    return float(min(1.0, (X0 + end + 1 - zero) / (X1 - zero))), present


@dataclass
class TowerHP:
    """Per-game tower state, read every `every` s (towers change slowly); `fracs()` is mirror()'s `towers`."""
    every: float = 0.5
    bars: dict = None

    def __post_init__(self):
        self.bars = self.bars or BARS_A16
        self.reset()

    def reset(self) -> None:
        self.frac = {k: 1.0 for k in self.bars}
        self.band: dict[str, tuple[int, int] | None] = {k: None for k in self.bars}
        self.gone_since: dict[str, float | None] = {k: None for k in self.bars}
        self.up_since: dict[str, float | None] = {k: None for k in self.bars}
        self.drop_seen: dict[str, tuple[float, float] | None] = {k: None for k in self.bars}
        self.woken: set[str] = set()
        self.vmin: dict[str, float] = {}  # team -> the bright threshold (DIM_FILL)
        self.destroyed: set[str] = set()
        self.last_t = -1e9
        self.last_elapsed: float | None = None

    def update(self, frame: np.ndarray, t: float, elapsed_s: float | None = None) -> dict[str, float]:
        if elapsed_s is not None and self.last_elapsed is not None and elapsed_s < self.last_elapsed - 10:
            self.reset()  # the clock went back: a new game
        if elapsed_s is not None:
            self.last_elapsed = elapsed_s
        if t - self.last_t < self.every - 1e-6 or (elapsed_s is not None and elapsed_s < START_S):
            return self.fracs()
        self.last_t = t
        for team in ("own", "opp"):
            # learnt only early (every tower full); a reader started later keeps 170
            if team not in self.vmin and elapsed_s is not None and elapsed_s <= LEARN_BY_S:
                self._learn_vmin(frame, team)
        for name in self.bars:
            if name in self.destroyed:
                continue
            vmin = self.vmin.get(name[:3], 170)
            band = None
            if name not in FIXED:
                band = self.band[name] = self.band[name] or find_band(frame, name, self.bars, vmin)
                if band is None:  # their bar not located yet (a King not woken, or the start of the game)
                    continue
            f, present = read_bar(frame, name, band, self.bars, vmin)
            absent = f is None and not present  # (fill None but present: their bar under the red pulse, or no fill)
            if f is not None and name.endswith("king") and name not in self.woken:
                if f < WAKE:
                    f = None
                else:
                    self.woken.add(name)
            if f is not None and f > self.frac[name] + UP_TOL:  # higher than before: wait for it to hold
                self.up_since[name] = self.up_since[name] if self.up_since[name] is not None else t
                if t - self.up_since[name] < UP_S:
                    f, absent = None, True  # a destroyed tower's spot can flash pink (bot_2026-11-14_172646 58 s)
            else:
                self.up_since[name] = None
            if f is not None and f < self.frac[name] - DROP_TOL:  # a fall too big for one read: hold it DROP_S
                d = self.drop_seen[name]
                if d is None or abs(d[0] - f) > 0.1:
                    self.drop_seen[name], f = (f, t), None
                elif t - d[1] < DROP_S - 1e-6:
                    f = None
            else:
                self.drop_seen[name] = None
            if f is not None:
                self.frac[name], self.gone_since[name] = f, None
                continue
            if name.endswith("king"):
                continue  # no bar: not woken yet (full), or a flicker; a King's fall ends the game
            if not absent:
                self.gone_since[name] = None
                continue
            # no bar (or none believable): gone this long, from low enough, is destroyed
            self.gone_since[name] = self.gone_since[name] if self.gone_since[name] is not None else t
            if t - self.gone_since[name] >= GONE_S and self.frac[name] <= GONE_BELOW:
                self.destroyed.add(name)
                self.frac[name] = 0.0
        return self.fracs()

    def _learn_vmin(self, frame: np.ndarray, team: str) -> None:
        """The team's bright threshold from its princess bars at their first read (all full: the clock is past
        START_S): the median V of the bar's hue pixels (the fill) x DIM_FILL, at most 170."""
        vs = []
        for name in (f"{team}_l", f"{team}_r"):
            if name not in self.bars:
                continue
            X0, X1, Y0, Y1 = _px(name, frame.shape[1], self.bars)
            if name not in FIXED:
                band = find_band(frame, name, self.bars, 0)  # fill or track, whatever their brightness
                if band is None:
                    return
                Y0, Y1 = band
            hsv = cv2.cvtColor(frame[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
            lo, hi = TEAM_HUE[team]
            m = (hsv[:, 0] >= lo) & (hsv[:, 0] <= hi) & (hsv[:, 1] >= 80)
            if m.mean() < 0.5:
                return  # something over the bar: try again next read
            vs.append(float(np.median(hsv[m, 2])))
        if vs:
            self.vmin[team] = min(170.0, DIM_FILL * min(vs))

    def fracs(self) -> dict[str, float]:
        return dict(self.frac)
