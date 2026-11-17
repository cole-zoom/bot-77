"""Step 5 of the input rebuild (2026-11-16, decisions.md): training rows rebuilt through the live code path, so the
data a net learns from is as noisy as what it reads in a game.

Shared by scripts/rebuild_shadow.py, rebuild_review.py, rebuild_sim_block.py and build_bc_live.py:

- LiveObs: the engine observation exactly as live play builds it (SelfPlayPolicy._mirror: the screen read -> an
  engine), without loading a net.
- cell_to_ours: a label's board cell -> the ground point in our frame (the own play fed to OppInputs).
- cards(): the deck-card rows of the cards table (names, ids, types, costs) the live stack reads with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FORK = ROOT / "third_party" / "clash-royale-env"
DECK = "Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows"  # bot-77's deck
BOARD_W = 18


def cards(db_path: str | Path = ROOT / "data/lancedb") -> list[dict]:
    import lancedb
    return lancedb.connect(str(db_path)).open_table("cards").search().where("kind = 'deck_card'").limit(1000) \
        .to_arrow().to_pylist()


def cell_to_ours(cell: int) -> tuple[float, float]:
    """The engine cell's centre (the fork's binning, row * 18 + col) -> our frame."""
    from bot77.sim.geometry import engine_to_ours
    return engine_to_ours(cell % BOARD_W + 0.5, cell // BOARD_W + 0.5)


class LiveObs:
    """SelfPlayPolicy._mirror without the net: the 13,977-float observation live play would hand its net."""

    def __init__(self, cards_: list[dict]):
        if str(FORK) not in sys.path:
            sys.path.insert(0, str(FORK))
        from bot77.live.selfplay import SelfPlayPolicy
        self.p = SelfPlayPolicy.__new__(SelfPlayPolicy)
        self.p.cards = {c["id"]: c for c in cards_}
        self.p.deck = []
        self.p.blind = []
        self.sim = None

    def obs(self, st, opp_plays: list[dict]) -> np.ndarray | None:
        self.sim = self.p._mirror(st, opp_plays)
        if self.sim is None:
            return None
        return np.asarray(self.sim.env.get_observation_for_team(0), dtype=np.float16)


# observation sections (scripts/dump_live_inputs.py decode): where the rebuilt rows differ from the old ones
NCH, W, H, NCARD = 21, 18, 34, 185
SPATIAL = NCH * W * H
HAND0 = SPATIAL  # elixir, 4 costs, 4 x 185 hand
SCAL0 = HAND0 + 5 + 4 * NCARD
CYC0 = SCAL0 + 10


def section_of(col: int) -> str:
    if col < SPATIAL:
        ch = col // (W * H)
        return "board HP ch0-7" if ch < 8 else ("board units ch9-10" if ch in (9, 10) else
                                                "board river ch8" if ch == 8 else "board attrs ch11-20")
    if col < SCAL0:
        return "own elixir+hand"
    if col < CYC0:
        k = col - SCAL0
        return {0: "scalar time", 1: "scalar own spent", 2: "scalar opp spent", 9: "scalar multiplier"}.get(
            k, "scalar tower HP")
    if col < CYC0 + 2 * NCARD:
        return "opp cycle"
    return "tail"


def diff_sections(old: np.ndarray, new: np.ndarray, tol: float = 1e-3) -> dict[str, int]:
    """Rows x columns that differ by more than `tol`, counted per section of the observation."""
    d = np.abs(old.astype(np.float32) - new.astype(np.float32)) > tol
    cols = np.nonzero(d.any(0))[0]
    out: dict[str, int] = {}
    per_col = d.sum(0)
    for c in cols:
        s = section_of(int(c))
        out[s] = out.get(s, 0) + int(per_col[c])
    return out
