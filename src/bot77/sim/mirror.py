"""One player's view of a moment, mirrored into an engine, for the self-play network's observation (2026-10-24, option 3:
imitation from pro replays, then reinforcement learning). The same function builds the imitation data from replays
and the live agent's input from the screen, so training and play see the world the same way.

All positions are in the acting player's frame (bottom = them, y down the screen), feet (not box centres). The acting
player is engine team 0.
"""
from __future__ import annotations

import numpy as np

from bot77.sim import cards as sc
from bot77.sim.engine import Sim, sync_spent_to_bars
from bot77.sim.scenario import place_units


def mirror(own_deck: list[int], opp_deck: list[int], t_match: float, hand: list[int], elixir_own: float,
           elixir_opp: float, units: list[dict], towers: dict[str, float] | None = None,
           opp_played: list[int] | None = None) -> Sim | None:
    """Engine ids throughout. units: card (our name), cls, team own / opp, x, y (feet, acting player's frame).
    towers: HP fractions {own_king, own_l, ...} (0 = destroyed; missing = full). opp_played: the opponent's plays so
    far, oldest first (the observation's card-cycle blocks). None if the hand can't be set."""
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
        sim.env.note_played_card(1, e)
    # plan P5: the spend that makes the engine's opponent-elixir estimate (observation scalar 10) read elixir_opp, and
    # our own read elixir_own (a no-op on an engine without it)
    sync_spent_to_bars(sim.env)
    sim.step(1)  # commit the injected units
    return sim


def observation(sim: Sim) -> np.ndarray:
    return np.asarray(sim.env.get_observation_for_team(0), dtype=np.float32)
