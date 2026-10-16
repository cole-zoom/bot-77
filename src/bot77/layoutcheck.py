"""Score how well a layout reads a video: clock, elixir *and* hand, on sampled in-game frames.

Batch prep used to pick layouts by clock reads alone. That let 20 of 129 videos through with a layout that read the
clock but not the elixir (Ryley/Mohamed: elixir 0 all game) or not the hand (Ian77's older split overlay), so
almost none of their plays could be confirmed (decisions.md, 2026-10-02)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np

from bot77.calibrate import read_frame
from bot77.layout import Layout
from bot77.readers.clock import read_clock
from bot77.readers.elixir import read_elixir
from bot77.readers.hand import HandReader
from bot77.video import probe

_CARDS = None


def _cards() -> list[dict]:
    global _CARDS
    if _CARDS is None:
        _CARDS = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'") \
            .limit(1000).to_arrow().to_pylist()
    return _CARDS


@dataclass
class LayoutScore:
    layout_id: str
    clock: float  # share of sampled frames where the clock reads
    elixir: float  # share of in-game frames with a readable, consistent elixir value
    hand: float  # mean share of the 4 slots read as a card with a clear margin, on in-game frames
    next: float  # share of in-game frames where the Next card reads (Ian's old overlay: never)
    total: float

    def __str__(self):
        return (f"{self.layout_id}: clock {self.clock:.0%} elixir {self.elixir:.0%} hand {self.hand:.0%} "
                f"next {self.next:.0%} -> {self.total:.2f}")


def score_layout(video: Path, layout_id: str, n: int = 12, min_margin: float = 0.05) -> LayoutScore:
    lay = Layout.load(layout_id)
    du = probe(video)["duration"]
    hand = HandReader.from_cards(lay, _cards())
    clock = elix = nxt = 0
    slots: list[float] = []
    for q in np.linspace(0.08, 0.92, n):
        try:
            f = read_frame(video, du * q)
        except Exception:  # noqa: BLE001 - unreadable frame: counts as a miss
            continue
        if f is None or read_clock(f, lay).elapsed_s is None:
            continue
        clock += 1
        e = read_elixir(f, lay)
        # a readable, non-trivial value: a wrong layout can read a "consistent" 0 or nothing at all
        elix += int(e.elixir is not None and e.readable_frac > 0.8 and e.elixir > 0.2)
        reads = hand.read(f)
        slots.append(np.mean([r.state == "card" and r.margin >= min_margin for k, r in reads.items() if k.startswith("slot")]))
        nxt += int(reads.get("next") is not None and reads["next"].state == "card" and reads["next"].margin >= min_margin)
    c = clock / n
    el, nx = (elix / clock, nxt / clock) if clock else (0.0, 0.0)
    h = float(np.mean(slots)) if slots else 0.0
    return LayoutScore(layout_id, c, el, h, nx, c * el * h * nx)  # every reader has to work: a product
