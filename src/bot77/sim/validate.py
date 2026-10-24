"""The simulator against the replays (docs/simulator.md S2, 2026-10-23).

A window: a play on an empty board in a human replay, then every play of both players (`plays`, source hand: exact
card and time; the spot is the box centre of the unit's first detection) for up to WINDOW_S seconds, ending early at
the first play with no spot, no engine card, or no card read. The sim starts PRE_S before the first play and gets the
same plays at the same times; every DET_DT (0.25 s) its units are compared with the replay's detections (v5l_e15):
per card and team, matched by distance (Hungarian).

Only troops and buildings are compared (spells leave no body in either). Towers are set from the alive flags; tower
HP comes later (`tower_hp`). Windows need every play spotted, which keeps 51% of troop plays but only 21% of spells, so
the windows lean toward troop fights.
"""
from __future__ import annotations

import bisect
import collections
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot77.features.frames import FEET_DY
from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.sim.scenario import SPELL_CLASSES, _engine_deck, _rows, _squad_size, place_units, side_decks

RUN = "v5l_e15"
WINDOW_S = 12.0
MIN_WINDOW_S = 6.0
PRE_S = 0.5
DET_DT = 0.25
EMPTY_BEFORE_S = (1.0, 0.25)  # the board must be (near-)empty in these frames before the first play
MAX_START_UNITS = 2  # bodies already on the board at the start: injected (full HP, deployed); 0 = empty boards only
TEAM = {"friendly": "own", "enemy": "opp"}
SIDE = {"bottom": "own", "top": "opp"}
DT_BINS = (0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5)


def _is_unit(cls: str | None, card: str | None) -> bool:
    if not card or not cls:
        return False
    return not ("tower" in cls or "bar" in cls or cls in SPELL_CLASSES
                or any(cls.startswith(s) for s in SPELL_CLASSES))


@dataclass
class Game:
    game_id: str
    split: str
    source: str
    plays: list[dict]  # both sides, sorted by video t
    frames: dict[float, list[dict]]  # video t -> units (card, team own/opp, x, y feet)
    frame_ts: list[float]
    towers: list[tuple[float, dict]]  # (t_match, alive flags in the bottom frame)
    offset: float  # video t - game clock
    decks: dict[str, list[int]] = field(default_factory=dict)  # side -> engine ids
    card_map: dict[str, dict[int, int]] = field(default_factory=dict)  # side -> our card id -> the deck's engine id
    missing: list[str] = field(default_factory=list)


def load_game(game_id: str, split: str, source: str) -> Game:
    plays = _rows("plays", f"game_id = '{game_id}' AND source = 'hand'",
                  ["side", "t", "t_match", "card", "card_id", "form", "kind", "elixir_after", "x", "y"])
    plays.sort(key=lambda p: p["t"])
    dets = _rows("detections", f"match_id = '{game_id}' AND run_id = '{RUN}'", ["t", "card", "cls", "team", "x", "y"],
                 limit=10**7)
    frames: dict[float, list[dict]] = collections.defaultdict(list)
    king_bar: dict[str, list[float]] = {"own": [], "opp": []}  # the king's HP bar shows once it has been damaged
    for d in dets:
        if d["cls"] == "king-tower-bar" and d["team"] in TEAM:
            king_bar[TEAM[d["team"]]].append(d["t"])
        t = round(d["t"], 2)
        frames.setdefault(t, [])
        if _is_unit(d["cls"], d["card"]) and d["team"] in TEAM:
            frames[t].append({"card": d["card"], "cls": d["cls"], "team": TEAM[d["team"]], "x": d["x"],
                              "y": d["y"] + FEET_DY})
    tw = _rows("moments", f"game_id = '{game_id}' AND side = 'bottom'", ["t_match", "towers"])
    towers = sorted(((m["t_match"], m["towers"]) for m in tw if m["towers"]), key=lambda x: x[0])
    offs = [p["t"] - p["t_match"] for p in plays if p["t_match"] is not None]
    decks, card_map, missing = {}, {}, []
    sides = side_decks(game_id)
    for side in ("bottom", "top"):
        deck, ids, miss = _engine_deck(sides.get(side, []), [p for p in plays if p["side"] == side])
        decks[side], card_map[side] = deck, ids
        missing += miss
    g = Game(game_id, split, source, plays, dict(frames), sorted(frames), towers,
             float(np.median(offs)) if offs else 0.0, decks, card_map, missing)
    # first time the bar is seen in a run of >= 4 frames (1 s; one-frame detections are noise)
    g.king_awake = {k: _first_run(v, 4) for k, v in king_bar.items()}
    for p in plays:  # an Evolution card plays as 'normal' and 'evo' in turn: always the deck's engine card
        p["engine_card"] = card_map[p["side"]].get(p["card_id"]) if p["card_id"] else None
    return g


def _frame(g: Game, t: float) -> list[dict] | None:
    """The replay frame nearest t (within half a frame), or None."""
    i = bisect.bisect_left(g.frame_ts, t)
    near = [j for j in (i - 1, i) if 0 <= j < len(g.frame_ts)]
    j = min(near, key=lambda j: abs(g.frame_ts[j] - t), default=None)
    if j is not None and abs(g.frame_ts[j] - t) <= DET_DT / 2:
        return g.frames[g.frame_ts[j]]
    return None


def _play_spot(p: dict) -> tuple[float, float] | None:
    if p["x"] is None:
        return None
    spell = sc.engine_cards().get(_engine_card(p) or -1, {}).get("is_spell")
    return (p["x"], p["y"]) if spell else (p["x"], p["y"] + FEET_DY)


def _engine_card(p: dict) -> int | None:
    return p.get("engine_card")


def windows(g: Game, max_start: int = MAX_START_UNITS) -> list[dict]:
    """Candidate windows: {'t0': video t of the first play, 't1': end, 'plays': [...]}. Windows don't overlap."""
    out, busy_until = [], -1.0
    plays = [p for p in g.plays if p["kind"] == "play"]
    for i, p in enumerate(plays):
        t0 = p["t"]
        if t0 < busy_until:
            continue
        before = [_frame(g, t0 - dt) for dt in EMPTY_BEFORE_S]
        if any(f is None or len(f) > max_start for f in before):
            continue
        t1, ws = t0 + WINDOW_S, []
        for q in plays[i:]:
            if q["t"] >= t1:
                break
            if _engine_card(q) is None or _play_spot(q) is None:
                t1 = q["t"]
                break
            ws.append(q)
        if t1 - t0 >= MIN_WINDOW_S and ws:
            out.append({"t0": t0, "t1": t1, "plays": ws, "start_units": len(_frame(g, t0 - PRE_S) or [])})
            busy_until = t1
    return out


def _towers_at(g: Game, t_match: float) -> dict | None:
    i = bisect.bisect_right([t for t, _ in g.towers], t_match) - 1
    return g.towers[i][1] if i >= 0 else None


def run_window(g: Game, w: dict, towers: str = "engine", wake_kings: bool = True) -> dict:
    """Play the window in the sim. -> {'frames': [(dt, sim units, replay units)], 'plays': [PlayResult], 'sim'}.
    Sim tick = (video t - offset) x 10; a replay frame is compared with the sim at its nearest tick."""
    sim = Sim.new(g.decks["bottom"], g.decks["top"], towers=towers)
    sim.reset()
    tick = lambda t: round((t - g.offset) * 10)  # noqa: E731
    sim.env.set_current_tick(tick(w["t0"] - PRE_S))
    for team in ("own", "opp"):
        sim.set_elixir(team, 10.0)
    tw = _towers_at(g, sim.t) or {}
    for key, team, slot in (("own_l", "own", "l"), ("own_r", "own", "r"), ("opp_l", "opp", "l"), ("opp_r", "opp", "r")):
        if tw.get(key) is False:
            sim.set_tower(team, slot, 0.0)
    if wake_kings:
        for team in ("own", "opp"):
            t_awake = getattr(g, "king_awake", {}).get(team)
            if t_awake is not None and t_awake < w["t0"]:
                sim.wake_king(team)
    start_units = _frame(g, w["t0"] - PRE_S) or []
    injected, _ = place_units(sim, start_units)
    plays = collections.deque(sorted(w["plays"], key=lambda p: p["t"]))
    dets = {}
    for t in g.frame_ts[bisect.bisect_left(g.frame_ts, w["t0"] - PRE_S):bisect.bisect_right(g.frame_ts, w["t1"])]:
        dets.setdefault(tick(t), (t, g.frames[t]))
    frames, results = [], []
    end = tick(w["t1"])
    while sim.tick <= end:
        if sim.tick in dets:
            t, du = dets[sim.tick]
            frames.append((t - w["t0"], [u for u in sim.units() if not u["spell"] and not u["tower"]], du))
        if plays and tick(plays[0]["t"]) <= sim.tick:
            p = plays.popleft()
            x, y = _play_spot(p)
            results.append(sim.play(_engine_card(p), x, y, SIDE[p["side"]]))  # advances one tick
        elif sim.step(1):
            break
    return {"frames": frames, "plays": results, "sim": sim, "injected": injected}


def match_frame(sim_units: list[dict], det_units: list[dict]) -> list[tuple[dict | None, dict | None, float]]:
    """Per (card, team): Hungarian on distance. -> (sim unit, det unit, distance); unmatched have None on one side."""
    out = []
    keys = {(u["card"], u["team"]) for u in sim_units} | {(u["card"], u["team"]) for u in det_units}
    for k in keys:
        a = [u for u in sim_units if (u["card"], u["team"]) == k]
        b = [u for u in det_units if (u["card"], u["team"]) == k]
        if a and b:
            cost = np.array([[np.hypot(p["x"] - q["x"], p["y"] - q["y"]) for q in b] for p in a])
            ri, ci = linear_sum_assignment(cost)
            out += [(a[i], b[j], float(cost[i, j])) for i, j in zip(ri, ci)]
            out += [(a[i], None, np.nan) for i in set(range(len(a))) - set(ri)]
            out += [(None, b[j], np.nan) for j in set(range(len(b))) - set(ci)]
        else:
            out += [(u, None, np.nan) for u in a] + [(None, u, np.nan) for u in b]
    return out


def value(units: list[dict]) -> dict[str, float]:
    """Elixir on the board per team: each body counts its card's cost / bodies the card spawns."""
    out = {"own": 0.0, "opp": 0.0}
    for u in units:
        e = sc.engine_id(u["card"])
        if e is not None:
            out[u["team"]] += sc.cost(e) / _squad_size(e)
    return out


def _first_run(ts: list[float], min_run: int) -> float | None:
    ts, n, prev = sorted(ts), 0, None
    for t in ts:
        n = n + 1 if prev is not None and t - prev <= DET_DT * 1.5 else 1
        if n >= min_run:
            return t
        prev = t
    return None


MIN_RUN = 2  # a replay unit counts as there only in runs of this many frames (one-frame flickers and mislabels)


def _last_run(dts: list[float], min_run: int) -> float | None:
    """The end of the last run of >= min_run consecutive frames (DET_DT apart) in sorted times."""
    dts = sorted(dts)
    best, run_start, prev, n = None, None, None, 0
    for t in dts:
        if prev is not None and t - prev <= DET_DT * 1.5:
            n += 1
        else:
            run_start, n = t, 1
        if n >= min_run:
            best = t
        prev = t
    return best


def metrics(g: Game, w: dict, r: dict) -> list[dict]:
    """One row per (metric, card, dt bin) for this window."""
    rows = []
    base = {"match_id": g.game_id, "split": g.split, "source": g.source, "t0": w["t0"], "t1": w["t1"],
            "n_plays": len(w["plays"]), "start_units": w["start_units"]}
    errs = collections.defaultdict(list)  # (card, bin) -> distances
    dx_dy = collections.defaultdict(list)
    count_err = collections.defaultdict(list)
    lane_same = collections.defaultdict(list)
    present = collections.defaultdict(lambda: {"sim": [], "det": []})  # (card, team) -> dts present
    for dt, su, du in r["frames"]:
        b = next(i for i in range(len(DT_BINS) - 1) if DT_BINS[i] <= dt < DT_BINS[i + 1]) if dt >= 0 else None
        for a, d, dist in match_frame(su, du):
            k = (a or d)["card"], (a or d)["team"]
            if a is not None and d is not None and b is not None:
                errs[(k[0], b)].append(dist)
                dx_dy[b].append((a["x"] - d["x"], a["y"] - d["y"]))
                lane_same[b].append((a["x"] < 9) == (d["x"] < 9))
        by = collections.Counter((u["card"], u["team"]) for u in su), collections.Counter((u["card"], u["team"]) for u in du)
        for k in set(by[0]) | set(by[1]):
            if b is not None:
                count_err[b].append(abs(by[0][k] - by[1][k]))
            if by[0][k]:
                present[k]["sim"].append(dt)
            if by[1][k]:
                present[k]["det"].append(dt)
    for (card, b), v in errs.items():
        rows.append({**base, "metric": "pos_err", "card": card, "dt_bin": b, "value": float(np.median(v)), "n": len(v)})
    for b, v in dx_dy.items():
        v = np.array(v)
        rows.append({**base, "metric": "dx", "card": "", "dt_bin": b, "value": float(np.median(v[:, 0])), "n": len(v)})
        rows.append({**base, "metric": "dy", "card": "", "dt_bin": b, "value": float(np.median(v[:, 1])), "n": len(v)})
    for b, v in lane_same.items():
        rows.append({**base, "metric": "same_lane", "card": "", "dt_bin": b, "value": float(np.mean(v)), "n": len(v)})
    for b, v in count_err.items():
        rows.append({**base, "metric": "count_err", "card": "", "dt_bin": b, "value": float(np.mean(v)), "n": len(v)})
    end = r["frames"][-1][0] if r["frames"] else 0.0
    played = {(p["card"], SIDE[p["side"]]) for p in w["plays"]}
    for (card, team), p in present.items():
        if (card, team) not in played:  # only units played in the window: start units' HP is unknown
            continue
        ls, ld = _last_run(p["sim"], 1), _last_run(p["det"], MIN_RUN)
        if ls is None or ld is None:
            continue
        if min(ls, ld) < end - 1.0:  # at least one side lost it before the window ended
            rows.append({**base, "metric": "life_diff", "card": card, "dt_bin": -1, "value": ls - ld, "n": 1})
    if r["frames"]:
        last = [f for f in r["frames"] if f[0] >= end - 1.0]  # the last second, averaged (detector flicker)
        vs = {k: np.mean([value(f[1])[k] for f in last]) for k in ("own", "opp")}
        vd = {k: np.mean([value(f[2])[k] for f in last]) for k in ("own", "opp")}
        ws = np.sign(round(vs["own"] - vs["opp"]))
        wd = np.sign(round(vd["own"] - vd["opp"]))
        both = any(p["side"] == "bottom" for p in w["plays"]) and any(p["side"] == "top" for p in w["plays"])
        rows.append({**base, "metric": "winner_agree" if both else "winner_agree_onesided", "card": "", "dt_bin": -1,
                     "value": float(ws == wd), "n": 1})
        rows.append({**base, "metric": "value_diff_own", "card": "", "dt_bin": -1, "value": vs["own"] - vd["own"], "n": 1})
        rows.append({**base, "metric": "value_diff_opp", "card": "", "dt_bin": -1, "value": vs["opp"] - vd["opp"], "n": 1})
    pr = r["plays"]
    rows.append({**base, "metric": "plays_refused", "card": "", "dt_bin": -1, "value": float(sum(not p.played for p in pr)),
                 "n": len(pr)})
    rows.append({**base, "metric": "spot_moved", "card": "", "dt_bin": -1,
                 "value": float(np.mean([p.moved for p in pr])) if pr else 0.0, "n": len(pr)})
    return rows
