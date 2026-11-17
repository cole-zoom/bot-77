"""The brain's new input block (2026-11-16, step 4 of the input rebuild; decisions.md 2026-11-16): the opponent's
elixir, their likely hand, and spells in flight, appended after the engine's 13,977-float observation.

The engine observation is unchanged and stays first, so every old dataset and checkpoint still works: an old net
reads the first 13,977 floats (the fork locates every section by a forward offset), and a new net
(bot77.rl.extra_net.ExtraObsNet) reads the block on top. A missing block is all zeros.

Layout v1 (EXTRA_DIM = 436 floats, offsets from the block's start):

  0     their elixir estimate / 10 (PlayTracker.elixir(): 6 at 3:00, 1/2.8 s x the phase multiplier, minus each
        counted play's cost; held-out MAE 2.3, reads high when plays are missed)
  1     seconds since their last counted play / 30, capped at 1 (1 before any): how stale the estimate is
  2     their cards known / 8
  3     slots of their likely hand named / 4
  4     1 when all 8 of their cards are known (the cycle rule is exact then: 3.9 of 4 right on true plays, 2.5 on the
        tracker's)
  5     plays counted / 30, capped at 1
  6     can_hold[185]  (engine card ids) 1 for each card of their likely hand (PlayTracker.hand(): their known cards
        outside the last 4 played; longest-unplayed first, 4 at most). Cards they haven't shown stay 0: slot 3 says
        how many of the 4 are unknown
  191   comes_back[185] 1 for the card that returns to their hand at their next play: the oldest of their last 4
        different plays (only once 4 are known)
  376   3 spell slots x 20, soonest impact first (bot77.live.spells.SpellTracker):
          +0   present
          +1   type one-hot over SPELL_TYPES (11: Fireball, Rocket, Arrows, The Log, Zap, Goblin Barrel,
               Barbarian Barrel, Giant Snowball, Lightning, Poison, other)
          +12  theirs, +13 ours
          +14  where it is now: engine x / 18, engine y / 34 (the acting player's frame, as the board channels)
          +16  where it should land: engine x / 18, engine y / 34
          +18  time to impact / 3 s, capped at 1
          +19  theirs and in their likely hand (1), theirs and their 8 not all known yet (0.5), else 0

The same code builds it live and offline: OppInputs is fed every board read (LiveState, or the cached detections'
(card, team, x, y, conf) rows) and our plays, and block(t) reads it out.
"""
from __future__ import annotations

import numpy as np

from bot77.live.opponent import PlayTracker
from bot77.live.spells import SPELL_TYPES, SpellTracker

BASE_DIM = 13977
NCARD = 185
N_SCALARS = 6
SLOTS, SLOT_DIM = 3, 20
CAN_HOLD = N_SCALARS
COMES_BACK = CAN_HOLD + NCARD
SPELLS = COMES_BACK + NCARD
EXTRA_DIM = SPELLS + SLOTS * SLOT_DIM  # 436
LAYOUT = "v1"
SCALAR_NAMES = ["their elixir / 10", "s since their last play / 30", "their cards known / 8", "likely hand named / 4",
                "all 8 known", "plays counted / 30"]


def _engine_id(name: str) -> int | None:
    from bot77.sim import cards as sc
    try:
        return sc.engine_id(name)
    except Exception:  # noqa: BLE001
        return None


def _to_engine(x: float, y: float) -> tuple[float, float]:
    from bot77.sim.geometry import ours_to_engine
    return ours_to_engine(x, y)


class OppInputs:
    """PlayTracker + SpellTracker behind one interface. `plays` stands in for OpponentTracker.plays (the cycle blocks
    of the engine observation read it), `block(t)` is the new input block."""

    def __init__(self, cards: list[dict], our_deck: list[str]):
        self.trk = PlayTracker.from_cards(cards, our_deck=our_deck)
        self.spells = SpellTracker(self.trk.card_type, our_deck)
        self.t = 0.0

    @property
    def plays(self) -> list[dict]:
        return self.trk.plays

    def own_play(self, card: str, t: float, x: float | None = None, y: float | None = None) -> None:
        self.trk.own_play(card, t, x, y)

    def update(self, st) -> list[dict]:
        """bot77.live.state.LiveState in."""
        return self.update_units(st.t, st.units, st.elapsed_s)

    def update_units(self, t: float, units, elapsed_s: float | None = None) -> list[dict]:
        self.t = t
        new = self.trk.update(t, units, elapsed_s)
        deck = self.trk.deck()
        self.spells.update(t, units, self.trk.own, set(deck) if len(deck) == 8 else None)
        return new

    def block(self, t: float | None = None) -> np.ndarray:
        return build_block(self.trk, self.spells, self.t if t is None else t)


def build_block(trk: PlayTracker, spells: SpellTracker, t: float) -> np.ndarray:
    out = np.zeros(EXTRA_DIM, np.float32)
    hand, sure = trk.hand()
    deck = trk.deck()
    out[0] = trk.elixir() / 10.0
    out[1] = 1.0 if not trk.plays else min(1.0, max(0.0, t - trk.plays[-1]["t"]) / 30.0)
    out[2] = len(deck) / 8.0
    out[3] = sum(c is not None for c in hand) / 4.0
    out[4] = float(sure)
    out[5] = min(1.0, len(trk.plays) / 30.0)
    for c in hand:
        e = _engine_id(c) if c else None
        if e is not None:
            out[CAN_HOLD + e] = 1.0
    recent = []
    for q in reversed(trk.plays):
        if q["card"] in deck and q["card"] not in recent:
            recent.append(q["card"])
        if len(recent) == 4:
            break
    if len(recent) == 4:
        e = _engine_id(recent[3])
        if e is not None:
            out[COMES_BACK + e] = 1.0
    for k, tr in enumerate(spells.slots(SLOTS)):
        o = SPELLS + k * SLOT_DIM
        out[o] = 1.0
        out[o + 1 + (SPELL_TYPES.index(tr.card) if tr.card in SPELL_TYPES else len(SPELL_TYPES) - 1)] = 1.0
        out[o + 12 + (tr.owner != "opp")] = 1.0
        ex, ey = _to_engine(tr.x, tr.y)
        lx, ly = _to_engine(*(tr.land or (tr.x, tr.y)))
        out[o + 14: o + 18] = [ex / 18.0, ey / 34.0, lx / 18.0, ly / 34.0]
        # time to impact as of now (the tracker last aimed it at its last update)
        out[o + 18] = min(1.0, max(0.0, tr.tti - max(0.0, t - spells.t)) / 3.0)
        if tr.owner == "opp":
            out[o + 19] = 1.0 if tr.card in hand else 0.5 if not sure else 0.0
    return out


def decode_block(b: np.ndarray) -> dict:
    """The block back into words (scripts/dump_live_inputs.py decode)."""
    from bot77.sim import cards as sc

    def name(e):
        info = sc.engine_cards().get(int(e))
        return info["name"] if info else f"#{e}"

    out = {k: round(float(b[i]), 3) for i, k in enumerate(SCALAR_NAMES)}
    out["their elixir"] = round(float(b[0]) * 10, 2)
    out["can_hold"] = [name(e) for e in np.nonzero(b[CAN_HOLD:CAN_HOLD + NCARD] > 0)[0]]
    out["comes_back_next"] = [name(e) for e in np.nonzero(b[COMES_BACK:COMES_BACK + NCARD] > 0)[0]]
    sl = []
    for k in range(SLOTS):
        s = b[SPELLS + k * SLOT_DIM: SPELLS + (k + 1) * SLOT_DIM]
        if s[0] < 0.5:
            continue
        sl.append({"type": SPELL_TYPES[int(np.argmax(s[1:12]))], "side": "theirs" if s[12] > 0.5 else "ours",
                   "now_engine": [round(float(s[14]) * 18, 1), round(float(s[15]) * 34, 1)],
                   "lands_engine": [round(float(s[16]) * 18, 1), round(float(s[17]) * 34, 1)],
                   "impact_in_s": round(float(s[18]) * 3, 2), "in_their_hand": round(float(s[19]), 2)})
    out["spells_in_flight"] = sl
    return out


def fit_obs(obs: np.ndarray, width: int) -> np.ndarray:
    """Rows cut or zero-padded to `width` columns (old data for a new net: an all-zero block; new data for an old
    net: the engine part only)."""
    if obs.shape[-1] == width:
        return obs
    if obs.shape[-1] > width:
        return obs[..., :width]
    pad = np.zeros(obs.shape[:-1] + (width - obs.shape[-1],), obs.dtype)
    return np.concatenate([obs, pad], axis=-1)
