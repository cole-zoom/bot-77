"""One screen frame -> the bot's view of the game, with the same readers the offline pipeline uses (so a layout that
works for recordings works here), plus an online tracker of the opponent's plays from the unit detector."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from bot77.detect.names import class_to_card
from bot77.layout import Layout
from bot77.readers.clock import ClockRead, read_clock
from bot77.readers.elixir import read_elixir
from bot77.readers.hand import HandReader, infer_deck

TILES_W = 18  # the arena crop is 18 tiles wide (bot77.spawns)
RIVER_Y = 14.9


@dataclass
class Unit:
    card: str | None
    cls: str
    team: str
    x: float  # arena tiles
    y: float
    conf: float


@dataclass
class LiveState:
    t: float
    hand: list[int | None]  # 4 card ids (None: empty / unreadable slot)
    next: int | None
    elixir: float | None
    time_left_s: int | None
    elapsed_s: int | None
    multiplier: int
    phase: str | None  # regulation | overtime | None: no clock on screen (not in a battle)
    units: list[Unit] = field(default_factory=list)

    @property
    def in_battle(self) -> bool:
        return self.phase is not None and self.elixir is not None

    def threats(self) -> list[Unit]:
        """Enemy units on our half of the arena."""
        return [u for u in self.units if u.team == "enemy" and u.y > RIVER_Y and u.card]


class LiveReader:
    def __init__(self, layout: Layout, cards: list[dict], detector=None):
        self.layout, self.cards, self.detector = layout, cards, detector
        self.names = [c["name"] for c in cards]
        self.full = HandReader.from_cards(layout, cards)
        self.reader, self.reads, self.deck = self.full, [], []
        self.last_hand: list[int | None] = [None] * 4
        self.unknown_streak = 0
        # Fast mode (2026-10-28, decisions every 0.1 s): the clock is read at most every CLOCK_EVERY s and counted
        # forward in between (it is real time); the hand only when its pixels change; units every DETECT_EVERY s
        # (the replays the agent learned from had units 4 times a second). Off by default (fast = False).
        self.fast = False
        self._clock: tuple[float, ClockRead] | None = None
        self._hand: tuple[np.ndarray, dict] | None = None
        self._units: tuple[float, list] | None = None

    def tile_px(self, frame_w: int) -> tuple[float, float, float]:
        """(tile size, arena x0, arena y0) in frame pixels."""
        s = self.layout.scale(frame_w)
        ax0, ay0, ax1, _ = self.layout.katacr_arena
        return (ax1 - ax0) / TILES_W * s, ax0 * s, ay0 * s

    CLOCK_EVERY = 0.5
    DETECT_EVERY = 0.2

    def _read_clock(self, frame: np.ndarray, t: float) -> ClockRead:
        if self.fast and self._clock and t - self._clock[0] < self.CLOCK_EVERY and self._clock[1].elapsed_s is not None:
            t0, c = self._clock
            dt = t - t0
            return ClockRead(c.phase, None if c.time_left_s is None else c.time_left_s - dt, c.multiplier,
                             c.elapsed_s + dt)
        c = read_clock(frame, self.layout)
        self._clock = (t, c)
        return c

    def _read_hand(self, frame: np.ndarray) -> dict:
        if not self.fast:
            return self.reader.read(frame)
        # the four slots and Next only (the elixir bar below them changes every frame)
        crop = np.hstack([cv2.resize(cv2.cvtColor(self.layout.crop(frame, r), cv2.COLOR_BGR2GRAY), (24, 32))
                          for r in ("slot1", "slot2", "slot3", "slot4", "next")])
        if self._hand is not None and float(np.abs(crop.astype(np.int16) - self._hand[0]).mean()) < 3.0:
            return self._hand[1]  # nothing in the hand moved: the last read stands
        hand = self.reader.read(frame)
        self._hand = (crop.astype(np.int16), hand)
        return hand

    def read(self, frame: np.ndarray, t: float) -> LiveState:
        clock = self._read_clock(frame, t)
        hand = self._read_hand(frame)
        n_unknown = sum(v is None or v.state == "unknown" for k, v in hand.items() if k.startswith("slot"))
        # with a 7-card deck (one card the reader has no art for, 2026-10-25) one unreadable slot is expected
        # only in battle (the clock reads): the hand leaves the screen at the end, and that wiped the learned deck
        # before the next game could start from it (2026-10-27)
        stuck = clock.elapsed_s is not None and n_unknown > (1 if len(self.deck) == 7 else 0)
        self.unknown_streak = self.unknown_streak + 1 if stuck else 0
        if self.deck and self.unknown_streak >= 15:  # a slot unreadable for ~15 reads: a card outside the known
            self.deck, self.reader, self.reads = [], self.full, []  # deck (swapped in): relearn from all cards
            self.unknown_streak = 0
        if len(self.deck) < 8:  # learn our deck from the hand, then read against it only (faster, fewer mistakes)
            self.reads.append(hand)
            deck = infer_deck(self.reads[-90:])
            if len(deck) == 8:
                self.deck, self.reader = deck, self.full.restrict(set(deck))
            elif len(deck) == 7 and len(self.reads) >= 90 and not self.deck:
                # 90 reads and only 7 cards: the 8th has no art here. Read against the 7 (the full set took 1.2 s a
                # read and slowed the agent to a decision every ~4.5 s); the 8th stays unknown. (30 reads was too
                # few at a read a second: a card at the back of the cycle hadn't shown yet, 2026-10-27)
                self.deck, self.reader = deck, self.full.restrict(set(deck))
        slots = [hand.get(f"slot{k}") for k in range(1, 5)]
        nxt = hand.get("next")
        el = read_elixir(frame, self.layout)
        # a slot briefly unreadable (the card animating in after a play) keeps its last card; empty stays empty
        cur = [s.card_id if s and s.state == "card" else (None if s and s.state == "empty" else self.last_hand[k])
               for k, s in enumerate(slots)]
        self.last_hand = cur
        st = LiveState(t, cur,
                       nxt.card_id if nxt and nxt.state == "card" else None, el.elixir, clock.time_left_s, clock.elapsed_s,
                       clock.multiplier or 1, clock.phase)
        if self.detector is not None and st.in_battle:
            if self.fast and self._units and t - self._units[0] < self.DETECT_EVERY:
                st.units = list(self._units[1])
                return st
            tile, ox, oy = self.tile_px(frame.shape[1])
            for d in self.detector.detect(frame, self.layout):
                card = class_to_card(d.name, self.names)[0]
                cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                st.units.append(Unit(card, d.name, d.team, (cx - ox) / tile, (cy - oy) / tile, d.conf))
            self._units = (t, list(st.units))
        return st


class OpponentTracker:
    """The opponent's plays, online: an enemy unit of a card that wasn't on the field in the last few seconds is a new
    play. Coarser than the offline rules (bot77.opponent), but it only feeds A's opponent history, which A v1 barely
    relies on (C's history cost it 0-2 points, docs/move-predictor-spec.md)."""

    def __init__(self, gap_s: float = 6.0, min_seen: int = 2):
        self.gap_s, self.min_seen = gap_s, min_seen
        self.last_seen: dict[str, float] = {}
        self.pending: Counter = Counter()
        self.plays: list[dict] = []

    def update(self, st: LiveState) -> list[dict]:
        new = []
        cards = {u.card for u in st.units if u.team == "enemy" and u.card and u.conf >= 0.5}
        for c in cards:
            fresh = st.t - self.last_seen.get(c, -1e9) > self.gap_s
            if fresh:
                self.pending[c] += 1
                if self.pending[c] >= self.min_seen:  # seen on consecutive reads: a real unit, not a flicker
                    p = {"t": st.t, "card": c}
                    self.plays.append(p); new.append(p)
                    self.last_seen[c] = st.t; self.pending[c] = 0
            else:
                self.last_seen[c] = st.t
        for c in list(self.pending):
            if c not in cards:
                self.pending[c] = 0
        return new
