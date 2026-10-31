"""One player's view of a moment, mirrored into an engine, for the self-play network's observation (2026-10-24, option 3:
imitation from pro replays, then reinforcement learning). The same function builds the imitation data from replays
and the live agent's input from the screen, so training and play see the world the same way.

All positions are in the acting player's frame (bottom = them, y down the screen), feet (not box centres). The acting
player is engine team 0.
"""
from __future__ import annotations

import numpy as np

from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.sim.scenario import place_units


def mirror(own_deck: list[int], opp_deck: list[int], t_match: float, hand: list[int], elixir_own: float,
           elixir_opp: float, units: list[dict], towers: dict[str, float] | None = None,
           opp_played: list | None = None) -> Sim | None:
    """Engine ids throughout. units: card (our name), cls, team own / opp, x, y (feet, acting player's frame).
    towers: HP fractions {own_king, own_l, ...} (0 = destroyed; missing = full). opp_played: the opponent's plays so
    far, oldest first, each (engine id, seconds ago) (the observation's card-cycle blocks: which of their cards they
    have, and how recently each was played; a bare id counts as played now). None if the hand can't be set.

    Each play is stamped at its own match time (2026-10-31, Cole's review: "knowing their cycle, and what cards they
    have in their hand"): note_played_card stamps the engine's CURRENT tick, and the mirror is rebuilt per decision, so
    noting every play at t_match made every card read "just played" (recency 1.0), in the pro imitation data
    (scripts/build_bc_dataset.py) and live alike; the bot never saw which of their cards were cycled away."""
    sim = Sim.new(own_deck, opp_deck or [sc.engine_id("Knight")])
    sim.reset()
    sim.set_clock(float(t_match))
    if not sim.set_hand("own", hand):
        return None
    sim.set_elixir("own", float(np.clip(elixir_own, 0, 10)))
    sim.set_elixir("opp", float(np.clip(elixir_opp, 0, 10)))
    for key, frac in (towers or {}).items():
        team, slot = key.split("_")
        if frac is None or frac >= 0.999:
            continue
        if not key.endswith("king") and frac <= 0:
            sim.set_tower(team, slot, 0.0)
        elif frac > 0:
            sim.set_tower(team, slot, frac)
    place_units(sim, units)
    for e in opp_played or []:
        e, ago = e if isinstance(e, tuple) else (e, 0.0)
        sim.set_clock(max(0.0, float(t_match) - float(ago)))
        sim.env.note_played_card(1, e)
    sim.set_clock(float(t_match))
    sim.step(1)  # commit the injected units
    return sim


def observation(sim: Sim) -> np.ndarray:
    return np.asarray(sim.env.get_observation_for_team(0), dtype=np.float32)
