"""Our own card cycle, tracked (2026-11-02). Cole: "Can we not programmatically keep track of our deck? like its
deterministic in what order the deck will come in."

The hand is 4 cards and a queue of 4: a played card goes to the back of the queue and the front (Next) takes its slot.
So once the order is seen, every slot and every card's return is known. The reader is the eyes, this is the memory:

  - A confident read that matches the tracker: nothing to do.
  - A legal play: a slot now reads the card the tracker has as Next (or, while Next is unknown, one of the cards not
    seen yet), held for PLAY_HOLD_S, with our elixir dropping by about the old card's cost. The old card goes to the
    back of the queue. The tracker moves on what the screen shows, never on the bot's taps: a tap that fails changes
    nothing on screen.
  - Anything else (a card that isn't Next, one already in another slot, a change with no elixir drop): ignored,
    unless the screen keeps showing it for OVERRIDE_HOLD_S. Then the screen wins, the tracker re-syncs and the
    mismatch is logged. Misreads are one-frame events (2 flickers in 85,293 live reads; decisions.md 2026-11-02).

`hand` never reads None once started, so a card the reader can't make out (a card sliding in, the 10-28..11-02
lock-out bug) is still known. `queue` is what's coming, front first; positions not seen yet are None.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PLAY_HOLD_S = 0.2  # a legal play: the new card held this long (2 reads at 0.1 s)
OVERRIDE_HOLD_S = 0.5  # an unexpected read held this long: the screen is right, the tracker re-syncs
NEXT_AHEAD_HOLD_S = 3.0  # Next showing a card from further back: wait for the slot to settle first
START_HOLD_S = 0.5  # the opening hand and Next read the same this long before the tracker starts
ELIXIR_WINDOW_S = 3.0  # the drop is measured from the bar's top in the 3 s before the change
ELIXIR_SLACK = 1.5  # regen while the new card slides in (~1.2 s at 2x) and read noise


@dataclass
class Event:
    t: float
    kind: str  # start | play | next_learned | override | next_override
    slot: int | None = None
    old: int | None = None
    new: int | None = None
    note: str = ""


@dataclass
class CycleTracker:
    deck: list[int]
    cost: dict[int, float]
    hand: list[int | None] = field(default_factory=lambda: [None] * 4)
    queue: list[int | None] = field(default_factory=lambda: [None] * 4)
    started: bool = False
    events: list[Event] = field(default_factory=list)
    _elixir: list[tuple[float, float]] = field(default_factory=list)
    _seen: dict[int, tuple[int, float]] = field(default_factory=dict)  # slot -> (card, first t) of a pending change
    _next_seen: tuple[int, float] | None = None
    _start: tuple[tuple, float] | None = None

    @property
    def unseen(self) -> set[int]:
        """Deck cards whose queue position isn't known."""
        return set(self.deck) - {c for c in self.hand + self.queue if c is not None}

    @property
    def next(self) -> int | None:
        return self.queue[0]

    def update(self, t: float, slots: list[int | None], nxt: int | None, elixir: float | None) -> list[int | None]:
        """One read: the four slots and Next as confident card ids (None: unsure, empty, or not a card of ours)."""
        slots = [c if c in self.deck else None for c in slots]
        nxt = nxt if nxt in self.deck else None
        if elixir is not None:
            self._elixir.append((t, float(elixir)))
            self._elixir = [(a, e) for a, e in self._elixir if t - a <= ELIXIR_WINDOW_S + 1.0]
        if not self.started:
            self._try_start(t, slots, nxt)
            return list(self.hand)
        for k, c in enumerate(slots):
            if c is None or c == self.hand[k]:
                self._seen.pop(k, None)
                continue
            first = self._seen.get(k)
            if first is None or first[0] != c:
                self._seen[k] = (c, t)
                first = (c, t)
            held = t - first[1]
            legal = c == self.queue[0] or (self.queue[0] is None and c in self.unseen)
            # a chain: the slot settled on a card further back in the queue, so the ones in front of it came into
            # this slot and were played at once (the bot plays a card the moment it can read it: 185958, 0:45, Giant
            # then the Mini P.E.K.K.A that replaced it inside 0.2 s)
            j = self.queue.index(c) if c in self.queue[1:] else 0
            if legal and held >= PLAY_HOLD_S - 1e-9 and self._elixir_dropped(first[1], [self.hand[k]]):
                self._play(t, k, c, "")
            elif j and None not in self.queue[:j] and held >= PLAY_HOLD_S - 1e-9 \
                    and self._elixir_dropped(first[1], [self.hand[k]] + self.queue[:j]):
                self._chain(t, k, j, "")
            elif held >= OVERRIDE_HOLD_S - 1e-9:
                if legal:  # Next arrived and stayed, but the bar didn't show the cost (an elixir misread): a play
                    self._play(t, k, c, "no elixir drop seen")
                elif j and None not in self.queue[:j]:
                    self._chain(t, k, j, "no elixir drop seen")
                else:
                    self._override(t, k, c)
        self._learn_next(t, nxt)
        return list(self.hand)

    # --- internals ---
    def _try_start(self, t: float, slots, nxt) -> None:
        key = tuple(slots) + (nxt,)
        if None in key or len(set(key)) != 5:
            self._start = None
            return
        if self._start is None or self._start[0] != key:
            self._start = (key, t)
            return
        if t - self._start[1] >= START_HOLD_S - 1e-9:
            self.hand, self.queue, self.started = list(slots), [nxt, None, None, None], True
            self.events.append(Event(t, "start", note=f"hand {slots}, next {nxt}"))

    def _elixir_dropped(self, t0: float, cards: list[int | None]) -> bool:
        """Did the bar fall by about these cards' cost (each with ELIXIR_SLACK for regen and read noise)?"""
        before = [e for a, e in self._elixir if t0 - ELIXIR_WINDOW_S <= a <= t0]
        after = [e for a, e in self._elixir if a >= t0 - ELIXIR_WINDOW_S]
        if not before or None in cards:
            return False
        return max(before) - min(after) >= sum(self.cost.get(c, 3.0) - ELIXIR_SLACK for c in cards)

    def _play(self, t: float, k: int, c: int, note: str) -> None:
        old = self.hand[k]
        self.hand[k] = c
        # c came from the front of the queue (Next, known or not); the played card goes to the back
        self.queue = [None if q == c else q for q in self.queue[1:]] + [old]
        self._fill_last_unknown()
        self._seen.pop(k, None)
        self.events.append(Event(t, "play", k, old, c, note))

    def _chain(self, t: float, k: int, j: int, note: str) -> None:
        """The slot went through queue[0..j-1], each played as it arrived, and now holds queue[j]."""
        for _ in range(j + 1):
            self._play(t, k, self.queue[0], ("chained" + (f"; {note}" if note else "")))

    def _override(self, t: float, k: int, c: int) -> None:
        old = self.hand[k]
        self.hand[k] = c
        for j, q in enumerate(self.queue):  # the card can't also be in the queue; where `old` went is unknown
            if q == c:
                self.queue[j] = None
        for j in range(4):
            if j != k and self.hand[j] == c:
                self.hand[j] = None
        self._fill_last_unknown()
        self._seen.pop(k, None)
        self.events.append(Event(t, "override", k, old, c, "the screen held another card: re-synced"))

    def _learn_next(self, t: float, nxt: int | None) -> None:
        if nxt is None or nxt in self.hand:
            self._next_seen = None
            return
        if self.queue[0] == nxt:
            self._next_seen = None
            return
        if self._next_seen is None or self._next_seen[0] != nxt:
            self._next_seen = (nxt, t)
            return
        held = t - self._next_seen[1]
        if self.queue[0] is None and nxt in self.unseen and held >= PLAY_HOLD_S - 1e-9:
            self.queue[0] = nxt
            self._fill_last_unknown()
            self.events.append(Event(t, "next_learned", new=nxt))
            self._next_seen = None
        elif held >= (NEXT_AHEAD_HOLD_S if nxt in self.queue[1:] else OVERRIDE_HOLD_S) - 1e-9:
            # Next showing a card further back in the queue is a play the slots haven't shown yet (a card sliding
            # in): the slot settles and chains it; only if nothing settles for NEXT_AHEAD_HOLD_S is it a mismatch
            old = self.queue[0]
            self.queue = [nxt] + [None if q == nxt else q for q in self.queue[1:]]
            self._fill_last_unknown()
            self.events.append(Event(t, "next_override", old=old, new=nxt, note="Next held another card: re-synced"))
            self._next_seen = None

    def _fill_last_unknown(self) -> None:
        """One unknown position and one unplaced card: it goes there."""
        holes = [j for j, q in enumerate(self.queue) if q is None]
        left = self.unseen
        if len(holes) == 1 and len(left) == 1:
            self.queue[holes[0]] = left.pop()

    def mismatches(self) -> int:
        return sum(e.kind in ("override", "next_override") for e in self.events)
