"""Search for the best play at a moment (docs/simulator.md S3).

From a replay moment: the board as the scenario sets it up (bot77.sim.scenario), towers from tower_hp. Candidates are
each card in hand x a grid of legal spots x a few timings (now, +0.5, +1, +2, +3 s), plus waiting; the best spots are
then refined on a finer grid. Each candidate is a snapshot of the scenario, played, then run HORIZON_S seconds with the
opponent's actual next plays at their real times and spots (a replay of the same game: both hands are exact). Its
score is the change in win chance (bot77.sim.value) from the moment to the end of the window.

Caveats, all recorded with the results: the opponent's plays answered what really happened, not the candidate; our own
later plays are left out (only the candidate is played); unit HP at the start is full.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from bot77.features.frames import FEET_DY
from bot77.sim import cards as sc
from bot77.sim.engine import Sim, _nearest_legal
from bot77.sim.geometry import Y_SCALE, engine_to_ours, ours_to_engine
from bot77.sim.scenario import _rows, scenario_from_moment
from bot77.sim.spots import recover
from bot77.sim.value import WinModel, sim_features

HORIZON_S = 10.0
TIMINGS = (0.0, 0.5, 1.0, 2.0, 3.0)
GRID = 1.5  # coarse grid step, our tiles (x) / our units (y)
REFINE = (0.5, 3)  # step and half-width (in steps) of the fine grid around the best coarse spots
REFINE_TOP = 3


@dataclass
class Candidate:
    card: int | None  # engine id; None = wait
    x: float | None
    y: float | None
    delay: float
    score: float = 0.0
    p_end: float = 0.0
    note: str = ""


@dataclass
class SearchResult:
    moment_id: str
    p0: float
    candidates: list[Candidate]
    notes: dict = field(default_factory=dict)

    def best(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.score)

    def best_per_card(self) -> dict:
        """card (None = wait) -> its best candidate."""
        out: dict = {}
        for c in self.candidates:
            if c.card not in out or c.score > out[c.card].score:
                out[c.card] = c
        return out


def opponent_plays(game_id: str, side: str, t_match: float, horizon: float) -> tuple[list[dict], int]:
    """The opponent's plays in (t, t + horizon] with a spot (our frame of the deciding player, feet), and how many
    had none (left out)."""
    opp = "top" if side == "bottom" else "bottom"
    rows = _rows("plays", f"game_id = '{game_id}' AND source = 'hand' AND side = '{opp}' AND kind = 'play'",
                 ["t", "t_match", "card", "card_id", "form", "side", "x", "y"])
    out, missing = [], 0
    for p in sorted(rows, key=lambda p: p["t_match"]):
        if not (t_match < p["t_match"] <= t_match + horizon) or not p["card_id"]:
            continue
        e = sc.engine_id(p["card_id"], p["form"] or "normal") or sc.engine_id(p["card_id"])
        xy = recover(game_id, p) if p["card"] else None  # plays x / y, else from the detections
        if e is None or xy is None:
            missing += 1
            continue
        spell = sc.engine_cards()[e]["is_spell"]
        x, y = xy[0], xy[1] if spell else xy[1] + FEET_DY  # bottom frame, box centres -> ground
        if side == "top":  # into the top player's frame (spells about the river, units about FLIP_Y)
            from bot77.features.frames import FLIP_Y, RIVER_Y
            x, y = 18.0 - x, (2 * RIVER_Y if spell else FLIP_Y + 2 * FEET_DY) - y
        out.append({"t_match": p["t_match"], "card": e, "x": x, "y": y})
    return out, missing


def spot_grid(sim: Sim, card: int, step: float = GRID) -> list[tuple[float, float]]:
    """Legal spots for `card` on a grid over the arena, our frame."""
    out = []
    for x in np.arange(0.5, 18.0, step):
        for y in np.arange(0.5, 28.0, step):
            ex, ey = ours_to_engine(x, y)
            if sim.env.is_valid_placement(card, ex, ey, 0):
                out.append((float(x), float(y)))
    return out


def rollout(base: Sim, model: WinModel, opp: list[dict], cand: Candidate, t0: float) -> float:
    sim = base.snapshot()
    end = t0 + HORIZON_S
    queue = list(opp)
    played = cand.card is None
    while sim.t < end - 1e-6:
        if not played and sim.t >= t0 + cand.delay - 1e-6:
            r = sim.play(cand.card, cand.x, cand.y, "own", force=False)
            if not r.played:
                cand.note = (cand.note + ": " if cand.note else "") + (r.reason or "refused")
                return -1.0  # not playable as given (elixir, spot): never the best
            played = True
            continue
        if queue and sim.t >= queue[0]["t_match"] - 1e-6:
            p = queue.pop(0)
            sim.play(p["card"], p["x"], p["y"], "opp")  # forced: their elixir is an estimate
            continue
        nxt = min([end] + ([queue[0]["t_match"]] if queue else []) + ([t0 + cand.delay] if not played else []))
        if sim.step(max(1, round((nxt - sim.t) * 10))):
            break
    return model.p(sim_features(sim))


def search(moment_id: str, model: WinModel, tower_hp: dict | None = None, timings=TIMINGS,
           extra: list[Candidate] | None = None) -> SearchResult:
    """`extra`: candidates to score as well (e.g. the bot's actual play)."""
    scen = scenario_from_moment(moment_id, tower_hp=tower_hp)
    base, m = scen.sim, scen.moment
    t0 = base.t
    p0 = model.p(sim_features(base))
    opp, missing = opponent_plays(m["game_id"], m["side"], m["t_match"], HORIZON_S)
    hand = base.hand("own")
    cands = [Candidate(None, None, None, 0.0, note="wait")]
    for card in hand:
        for d in timings:
            cands += [Candidate(card, x, y, d) for x, y in spot_grid(base, card)]
    for c in cands:
        c.p_end = rollout(base, model, opp, c, t0)
        c.score = c.p_end - p0 if c.p_end >= 0 else -1.0
    # refine around the best few coarse candidates of each card
    fine = []
    for card in hand:
        top = sorted((c for c in cands if c.card == card and c.score > -1), key=lambda c: -c.score)[:REFINE_TOP]
        for c in top:
            step, n = REFINE
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    x, y = c.x + i * step, c.y + j * step
                    ex, ey = ours_to_engine(x, y)
                    if (i or j) and base.env.is_valid_placement(card, ex, ey, 0):
                        fine.append(Candidate(card, x, y, c.delay))
    for c in extra or []:  # the deck may hold the card's Evolution / Hero variant: use the hand's id for that card
        if c.card is not None and c.card not in hand:
            c.card = next((h for h in hand if sc.our_name(h) == sc.our_name(c.card)), c.card)
        if c.card is not None:  # a tap the engine calls illegal (e.g. just behind a tower): the game places the card
            ex, ey = ours_to_engine(c.x, c.y)  # at the nearest free spot, so does this
            if not base.env.is_valid_placement(c.card, ex, ey, 0):
                spot = _nearest_legal(base.env, c.card, ex, ey, 0)
                if spot is not None:
                    c.x, c.y = engine_to_ours(*spot)
    for c in fine + (extra or []):
        c.p_end = rollout(base, model, opp, c, t0)
        c.score = c.p_end - p0 if c.p_end >= 0 else -1.0
    notes = {**scen.notes, "opponent_plays": len(opp), "opponent_plays_without_spot": missing,
             "candidates": len(cands) + len(fine), "y_scale": Y_SCALE}
    return SearchResult(moment_id, p0, cands + fine + (extra or []), notes)
