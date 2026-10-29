"""A recorded lab battle (bot77.lab, docs/plan-sim-1to1.md P0.4) against the simulator running the same placements.

The replay comes through the usual pipeline (`plays` source hand: exact card and time for both sides; `detections`
v5l_e15; `tower_hp`), loaded with bot77.sim.validate.load_game. The BlueStacks account is the replay's bottom player
(engine team 0, 'own'), Cole the top ('opp').

The sim plays the whole battle from 0:00 with every real play at its real time: the test's scripted placements on their
scripted tile, anything else (cycling) on the spot read from the replay (bot77.sim.spots). It is compared with the
replay at every detector frame (0.25 s):
  - landing: where each scripted card's units first show vs its tile (engine tiles), and how late it was played;
  - bodies: per card and team, how many are on the board over time (real counts smoothed over 3 frames) and when
    each body goes (the "who died when" table);
  - towers: HP fraction over time (tower_hp vs the sim);
  - positions: each scripted unit's bodies vs the sim's at play + 1, 2, 4, 6, 8 s (engine tiles, Hungarian).
"""
from __future__ import annotations

import bisect
import collections
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot77 import lab
from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.sim.geometry import engine_to_ours, ours_to_engine
from bot77.sim.validate import DET_DT, Game, _play_spot, tower_frac

SIDE_OF = {"bluestacks": "bottom", "cole": "top"}
TEAM_OF = {"bluestacks": "own", "cole": "opp", "bottom": "own", "top": "opp"}
MATCH_DT = 25.0  # a scripted placement matches a real play of the same card and side within this many seconds
CHECKPOINTS = (1.0, 2.0, 4.0, 6.0, 8.0)
TOWERS = ("own_l", "own_r", "own_king", "opp_l", "opp_r", "opp_king")


def match_plays(test: lab.LabTest, g: Game) -> list[dict]:
    """Scripted placement -> the real play (same side and card, nearest in time, each used once)."""
    used, out = set(), []
    plays = [p for p in g.plays if p["kind"] == "play" and p["t_match"] is not None]
    for pl in sorted(test.placements, key=lambda p: (p.t, p.i)):
        cands = [(abs(p["t_match"] - pl.t), k) for k, p in enumerate(plays)
                 if k not in used and p["side"] == SIDE_OF[pl.side] and p["card"] == pl.card
                 and abs(p["t_match"] - pl.t) <= MATCH_DT]
        if cands:
            _, k = min(cands)
            used.add(k)
            out.append({"placement": pl, "play": plays[k], "late_s": plays[k]["t_match"] - pl.t})
        else:
            out.append({"placement": pl, "play": None, "late_s": None})
    return out


def score(test: lab.LabTest, g: Game) -> float:
    """How well a replay fits a test: the share of scripted placements found within 12 s."""
    m = match_plays(test, g)
    return sum(r["play"] is not None and abs(r["late_s"]) <= 12 for r in m) / max(1, len(m))


@dataclass
class Frame:
    t: float  # game time (s since the clock started)
    vt: float  # video time
    sim: list[dict]  # units (our frame, feet)
    real: list[dict]
    towers_sim: dict[str, float]
    towers_real: dict[str, float | None]


@dataclass
class LabRun:
    test: lab.LabTest
    game: Game
    matched: list[dict]
    frames: list[Frame] = field(default_factory=list)
    plays: list[dict] = field(default_factory=list)  # every play given to the sim: card, side, t, spot, scripted, result
    skipped: list[dict] = field(default_factory=list)


def run(test: lab.LabTest, g: Game, towers: str = "liquipedia_l11", t_end: float | None = None) -> LabRun:
    matched = match_plays(test, g)
    scripted = {id(r["play"]): r["placement"] for r in matched if r["play"] is not None}
    sim = Sim.new(g.decks["bottom"], g.decks["top"], towers=towers)
    sim.reset()
    for team in ("own", "opp"):
        sim.set_elixir(team, 10.0)
    lr = LabRun(test, g, matched)
    queue = []
    for p in sorted((p for p in g.plays if p["kind"] == "play" and p["t_match"] is not None), key=lambda p: p["t"]):
        card = p.get("engine_card")
        pl = scripted.get(id(p))
        if pl is not None:
            spot = engine_to_ours(float(pl.tile[0]), float(pl.tile[1]))
        else:
            spot = _play_spot(p) if card is not None else None
        if card is None or spot is None:
            lr.skipped.append({"card": p["card"], "side": p["side"], "t": p["t_match"],
                               "why": "no engine card" if card is None else "no spot"})
            continue
        queue.append((round(p["t_match"] * 10), p, card, spot, pl))
    queue = collections.deque(queue)
    if t_end is None:
        t_end = (g.frame_ts[-1] - g.offset) if g.frame_ts else test.end_t
    fticks = {}
    for vt in g.frame_ts:
        fticks.setdefault(round((vt - g.offset) * 10), vt)
    end_tick = round(t_end * 10)
    while sim.tick <= end_tick:
        if sim.tick in fticks:
            vt = fticks[sim.tick]
            lr.frames.append(Frame(sim.t, vt, [u for u in sim.units() if not u["spell"] and not u["tower"]],
                                   g.frames[vt], {k: v["frac"] for k, v in sim.towers().items()},
                                   {k: tower_frac(g, k, vt) for k in TOWERS}))
        if queue and queue[0][0] <= sim.tick:
            tick, p, card, spot, pl = queue.popleft()
            res = sim.play(card, spot[0], spot[1], TEAM_OF[p["side"]])  # advances one tick
            lr.plays.append({"card": p["card"], "side": p["side"], "t": p["t_match"], "x": spot[0], "y": spot[1],
                             "scripted": pl is not None, "played": res.played, "moved": res.moved,
                             "elixir_added": res.elixir_added, "reason": res.reason})
        elif sim.step(1):
            break
    return lr


# --- the comparisons ---

def _eng(x: float, y: float) -> tuple[float, float]:
    return ours_to_engine(x, y)


def window(lr: LabRun) -> tuple[float, float]:
    """The test's stretch of the battle: 2 s before its first scripted play to `tail_s` after its last."""
    ts = [r["play"]["t_match"] for r in lr.matched if r["play"] is not None] or [p.t for p in lr.test.placements]
    return min(ts) - 2.0, max(ts) + lr.test.tail_s


def _clean_tower_flags(lr: LabRun) -> None:
    """A princess tower read as destroyed (alive flag off, bot77.sim.validate.tower_frac -> 0) that shows HP again
    later was a detector miss: keep its last reading instead."""
    for k in TOWERS:
        vals = [f.towers_real[k] for f in lr.frames]
        later_alive = [False] * len(vals)
        seen = False
        for i in range(len(vals) - 1, -1, -1):
            later_alive[i] = seen
            seen = seen or (vals[i] is not None and vals[i] > 0)
        prev = None
        for i, f in enumerate(lr.frames):
            if vals[i] == 0.0 and later_alive[i]:
                f.towers_real[k] = prev
            prev = f.towers_real[k]


def landing(lr: LabRun) -> list[dict]:
    """Scripted card -> where its bodies first show in the replay (engine tiles; squads: their centre) vs the tile."""
    out = []
    for r in lr.matched:
        pl, p = r["placement"], r["play"]
        row = {"side": pl.side, "card": pl.card, "tile": list(pl.tile), "t_plan": pl.t,
               "t_real": None if p is None else round(p["t_match"], 2),
               "late_s": None if p is None else round(r["late_s"], 2)}
        if p is not None and not pl.spell:
            team = TEAM_OF[pl.side]
            for f in lr.frames:
                if f.t < p["t_match"] - 0.1 or f.t > p["t_match"] + 2.5:
                    continue
                bodies = [u for u in f.real if u["card"] == pl.card and u["team"] == team]
                if bodies:
                    ex, ey = _eng(np.mean([u["x"] for u in bodies]), np.mean([u["y"] for u in bodies]))
                    row.update({"first_seen_t": round(f.t, 2), "real_tile": [round(float(ex), 2), round(float(ey), 2)],
                                "dx": round(float(ex) - pl.tile[0], 2), "dy": round(float(ey) - pl.tile[1], 2),
                                "bodies": len(bodies)})
                    break
        out.append(row)
    return out


def _smooth(counts: list[int], k: int = 3) -> list[int]:
    h = k // 2
    return [int(np.median(counts[max(0, i - h):i + h + 1])) for i in range(len(counts))]


def bodies(lr: LabRun) -> list[dict]:
    """Per (card, team) seen in either: first seen, max bodies, and when each body went (count drops, held >= 3
    frames), sim vs real. Covers spawned bodies too (Spear Goblins from the Hut, the Battle Ram's Barbarians)."""
    w0, w1 = window(lr)
    frames = [f for f in lr.frames if w0 <= f.t <= w1]
    keys = sorted({(u["card"], u["team"]) for f in frames for u in f.sim + f.real})
    ts = [f.t for f in frames]
    out = []
    for card, team in keys:
        sim_c = [sum(u["card"] == card and u["team"] == team for u in f.sim) for f in frames]
        real_c = _smooth([sum(u["card"] == card and u["team"] == team for u in f.real) for f in frames])
        e = {"card": card, "team": team, "sim": _events(ts, sim_c), "real": _events(ts, real_c)}
        if e["sim"]["first"] is not None or e["real"]["first"] is not None:  # one-frame detections smooth away
            out.append(e)
    return out


def _events(ts: list[float], c: list[int], hold: int = 6) -> dict:
    """First seen, max count, and the times the count went down (each drop held for `hold` frames: 1.5 s, so a
    detector gap of a second isn't a death)."""
    if not any(c):
        return {"first": None, "max": 0, "drops": [], "last": None}
    first = ts[next(i for i, v in enumerate(c) if v)]
    last = ts[max(i for i, v in enumerate(c) if v)]
    drops = []
    for i in range(1, len(c)):
        if c[i] < c[i - 1] and all(v <= c[i] for v in c[i:i + hold]):
            drops += [round(ts[i], 2)] * (c[i - 1] - c[i])
    return {"first": round(first, 2), "max": max(c), "drops": drops, "last": round(last, 2)}


def towers(lr: LabRun, every: float = 1.0) -> list[dict]:
    """Tower HP fractions every `every` s where either side's value changed since the last row (plus the last)."""
    out, prev, next_t = [], None, -1.0
    w0, w1 = window(lr)
    frames = [f for f in lr.frames if w0 <= f.t <= w1]
    for f in frames:
        if f.t < next_t:
            continue
        next_t = f.t + every - 1e-6
        row = {"t": round(f.t, 2)}
        for k in TOWERS:
            row[k] = {"real": None if f.towers_real[k] is None else round(f.towers_real[k], 3),
                      "sim": round(f.towers_sim[k], 3)}
        sig = tuple((row[k]["real"], row[k]["sim"]) for k in TOWERS)
        if sig != prev:
            out.append(row)
            prev = sig
    if frames and (not out or out[-1]["t"] != round(frames[-1].t, 2)):
        f = frames[-1]
        out.append({"t": round(f.t, 2), **{k: {"real": None if f.towers_real[k] is None else round(f.towers_real[k], 3),
                                                "sim": round(f.towers_sim[k], 3)} for k in TOWERS}})
    return out


def positions(lr: LabRun) -> list[dict]:
    """Each scripted troop / building: its bodies (sim vs real, same card and team, Hungarian) at the checkpoints,
    mean distance in engine tiles (x tiles, y rows)."""
    out = []
    ts = [f.t for f in lr.frames]
    for r in lr.matched:
        pl, p = r["placement"], r["play"]
        if p is None or pl.spell:
            continue
        team = TEAM_OF[pl.side]
        for dt in CHECKPOINTS:
            t = p["t_match"] + dt
            i = bisect.bisect_left(ts, t)
            j = min((k for k in (i - 1, i) if 0 <= k < len(ts)), key=lambda k: abs(ts[k] - t), default=None)
            if j is None or abs(ts[j] - t) > DET_DT:
                continue
            f = lr.frames[j]
            a = [_eng(u["x"], u["y"]) for u in f.sim if u["card"] == pl.card and u["team"] == team]
            b = [_eng(u["x"], u["y"]) for u in f.real if u["card"] == pl.card and u["team"] == team]
            row = {"card": pl.card, "side": pl.side, "dt": dt, "n_sim": len(a), "n_real": len(b)}
            if a and b:
                cost = np.array([[np.hypot(p0[0] - q[0], p0[1] - q[1]) for q in b] for p0 in a])
                ri, ci = linear_sum_assignment(cost)
                row["err"] = round(float(cost[ri, ci].mean()), 2)
                row["sim"] = [round(float(v), 1) for v in np.mean([a[k] for k in ri], axis=0)]
                row["real"] = [round(float(v), 1) for v in np.mean([b[k] for k in ci], axis=0)]
            out.append(row)
    return out


def report(lr: LabRun) -> dict:
    _clean_tower_flags(lr)
    w0, w1 = window(lr)
    return {"window": [round(w0, 2), round(w1, 2)], "test": lr.test.id, "name": lr.test.name, "match_id": lr.game.game_id, "landing": landing(lr),
            "bodies": bodies(lr), "towers": towers(lr), "positions": positions(lr),
            "plays": lr.plays, "skipped": lr.skipped, "engine": _engine_tag()}


def _engine_tag() -> str:
    try:
        from bot77.sim.store import engine_commit
        return engine_commit()
    except Exception:  # noqa: BLE001
        return "?"


# --- the sim board, drawn like the play page (scripts/situation_play.py) ---

def draw_board(f: Frame, cell: int = 15):
    """The sim's units at a frame on the 18 x 34 engine grid (BlueStacks at the bottom), filled; the replay's
    detections at the same moment as rings. Returns a BGR image."""
    import cv2
    W, H = lab.ENGINE_W * cell, lab.ENGINE_H * cell
    im = np.full((H, W, 3), (214, 230, 210), np.uint8)
    for y in range(lab.ENGINE_H):
        if y % 2:
            im[H - (y + 1) * cell:H - y * cell] = (200, 220, 196)
    im[H - round(17.5 * cell):H - round(15.5 * cell)] = (226, 196, 140)
    for bx in (2, 14):
        im[H - round(17.5 * cell):H - round(15.5 * cell), round((bx - 0.5) * cell):round((bx + 1.5) * cell)] = (95, 141, 176)
    px = lambda ex, ey: (round((ex + 0.5) * cell), round(H - (ey + 0.5) * cell))  # noqa: E731
    colour = {"own": (181, 111, 47), "opp": (31, 122, 224)}  # BGR: BlueStacks blue, Cole orange
    for (tx, ty, team) in ((3, 6, "own"), (14, 6, "own"), (8.5, 2.5, "own"), (3, 27, "opp"), (14, 27, "opp"),
                           (8.5, 30.5, "opp")):
        key = f"{team}_{'king' if tx == 8.5 else ('l' if tx == 3 else 'r')}"
        s = 2 if tx == 8.5 else 1.5
        c = px(tx, ty)
        frac = f.towers_sim.get(key, 1.0)
        cv2.rectangle(im, (c[0] - round(s * cell), c[1] - round(s * cell)), (c[0] + round(s * cell), c[1] + round(s * cell)),
                      colour[team], 2 if frac > 0 else 1)
        real = f.towers_real.get(key)
        txt = f"{frac:.2f}" + ("" if real is None else f"/{real:.2f}")
        cv2.putText(im, txt, (c[0] - 26, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1, cv2.LINE_AA)
    for u in f.sim:
        ex, ey = _eng(u["x"], u["y"])
        c = px(ex, ey)
        r = round(0.45 * cell) if not u["building"] else round(0.9 * cell)
        cv2.circle(im, c, r, colour[u["team"]], -1)
        if u["flying"]:
            cv2.circle(im, c, r, (255, 255, 255), 1)
        cv2.putText(im, _abbr(u["card"]), (c[0] - 9, c[1] - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (20, 20, 20), 1,
                    cv2.LINE_AA)
    for u in f.real:
        ex, ey = _eng(u["x"], u["y"])
        cv2.circle(im, px(ex, ey), round(0.6 * cell), colour[u["team"]], 1, cv2.LINE_AA)
    return im


ABBR = {"Mini P.E.K.K.A": "MP", "Mega Minion": "MM", "Goblin Hut": "Hut", "Battle Ram": "Ram", "Hog Rider": "Hog",
        "Spear Goblins": "spg", "Barbarians": "barb", "Skeletons": "sk", "Goblins": "gob", "Ice Spirit": "ice"}


def _abbr(name: str) -> str:
    return ABBR.get(name, (name or "?")[:4])


def synthetic_test(g: Game, t0: float = 0.0, n: int = 4, window: float = 60.0) -> lab.LabTest:
    """A pseudo lab test from a replay's own plays (the self-check before the session): its first `n` plays after
    t0 that have a spot and an engine card, each 'scripted' on the engine tile nearest its spot, at its real time."""
    ps = []
    for p in sorted(g.plays, key=lambda p: p["t"]):
        if p["kind"] != "play" or p["t_match"] is None or p["t_match"] < t0 or p["t_match"] > t0 + window:
            continue
        spot = _play_spot(p) if p.get("engine_card") is not None else None
        if spot is None:
            continue
        ex, ey = ours_to_engine(*spot)
        ex, ey = min(max(ex, 0), lab.ENGINE_W - 1), min(max(ey, 0), lab.ENGINE_H - 1)
        side = "bluestacks" if p["side"] == "bottom" else "cole"
        ps.append(lab.Placement(side, p["card"], (int(round(ex)), int(round(ey))), round(p["t_match"], 2), "", len(ps)))
        if len(ps) == n:
            break
    names = sc.our_cards()
    decks = {s: [names.get(c, str(c)) for c in v] for s, v in (("bluestacks", []), ("cole", []))}
    return lab.LabTest("SELF", f"self-check on {g.game_id}", "pipeline check", "the compare runs end to end",
                       "left", decks, ps, tail_s=20.0)
