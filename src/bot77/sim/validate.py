"""The simulator against the replays (docs/simulator.md S2, 2026-10-23).

A window: a play on an empty board in a human replay, then every play of both players (`plays`, source hand: exact
card and time; the spot is the box centre of the unit's first detection) for up to WINDOW_S seconds, ending early at
the first play with no spot (`plays` or recovered from the detections, bot77.sim.spots), no engine card, or no card
read. The sim starts PRE_S before the first play and gets the
same plays at the same times; every DET_DT (0.25 s) its units are compared with the replay's detections (v5l_e15):
per card and team, matched by distance (Hungarian).

Only troops and buildings are compared (spells leave no body in either). Towers are set from the alive flags; tower
HP comes later (`tower_hp`). Windows need every play spotted, which keeps 51% of troop plays but only 21% of spells, so
the windows lean toward troop fights.
"""
from __future__ import annotations

import bisect
from pathlib import Path
import collections
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot77.features.frames import FEET_DY
from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.sim.spots import recover
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


def load_game(game_id: str, split: str, source: str, recover_spots: bool = True) -> Game:
    plays = _rows("plays", f"game_id = '{game_id}' AND source = 'hand'",
                  ["side", "t", "t_match", "card", "card_id", "form", "kind", "elixir_after", "x", "y"])
    plays.sort(key=lambda p: p["t"])
    dets = _rows("detections", f"match_id = '{game_id}' AND run_id = '{RUN}'",
                 ["t", "card", "cls", "team", "x", "y", "x0", "y0", "x1", "y1"], limit=10**7)
    frames: dict[float, list[dict]] = collections.defaultdict(list)
    bars: dict[float, list[dict]] = collections.defaultdict(list)  # unit HP bars and level badges, frame pixels
    king_bar: dict[str, list[float]] = {"own": [], "opp": []}  # the king's HP bar shows once it has been damaged
    for d in dets:
        if d["cls"] == "king-tower-bar" and d["team"] in TEAM:
            king_bar[TEAM[d["team"]]].append(d["t"])
        t = round(d["t"], 2)
        frames.setdefault(t, [])
        if _is_unit(d["cls"], d["card"]) and d["team"] in TEAM:
            frames[t].append({"card": d["card"], "cls": d["cls"], "team": TEAM[d["team"]], "x": d["x"],
                              "y": d["y"] + FEET_DY, "box": (d["x0"], d["y0"], d["x1"], d["y1"])})
        elif d["cls"] in ("bar", "bar-level") and d["team"] in TEAM:
            bars[t].append({"cls": d["cls"], "team": TEAM[d["team"]], "box": (d["x0"], d["y0"], d["x1"], d["y1"])})
    tw = _rows("moments", f"game_id = '{game_id}' AND side = 'bottom'", ["t_match", "towers"])
    towers = _steady_flags(sorted(((m["t_match"], m["towers"]) for m in tw if m["towers"]), key=lambda x: x[0]))
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
    g.bars = dict(bars)
    g.tower_hp = load_tower_hp(game_id)
    g.towers = _standing_while_read(g.towers, g.tower_hp, g.offset)
    g.levels, g.tower_max = load_levels(game_id, decks)
    for p in plays:  # an Evolution card plays as 'normal' and 'evo' in turn: always the deck's engine card
        p["engine_card"] = card_map[p["side"]].get(p["card_id"]) if p["card_id"] else None
        if p["x"] is None and p["card"] and p["kind"] == "play" and recover_spots:
            p["spot"] = recover(game_id, p)
    return g


DOWN_SHARE = 0.8  # a "destroyed" flag counts only if this share of the tower's later flags agree


def _steady_flags(towers: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Alive flags with the detector's dropouts removed: a princess tower read destroyed counts as destroyed only if
    at least DOWN_SHARE of its flags from then to the end of the game say so too (a destroyed tower never comes
    back). Found 2026-10-29: flags drop to False for seconds while the tower's HP bar still reads 0.93
    (tv_2026-10-13_r06 at 27-33 s), which read as a whole tower lost in the window and, at a window's start, put
    the sim's tower at 0."""
    out = [(t, dict(f)) for t, f in towers]
    keys = {k for _, f in towers for k in f}
    for k in keys:
        down = later = 0
        for i in range(len(out) - 1, -1, -1):
            v = out[i][1].get(k)
            if v is None:
                continue
            later += 1
            down += v is False
            if v is False and down < DOWN_SHARE * later:
                out[i][1][k] = True
    return out


def _standing_while_read(towers: list[tuple[float, dict]], tower_hp: dict[str, list[tuple[float, float]]],
                         offset: float) -> list[tuple[float, dict]]:
    """A princess tower whose HP bar is still read is standing: its flags before its last reading > 0 are True.
    Found 2026-11-15 on the bot's replays: the flag drops to False from a tower's first hits, 4-10 s before it falls
    (bl_2026-10-29_r42 own_r: False from 24.7 s match time, bar read 0.93 ... 0.09 until 31 s). Every later flag is
    False too, so _steady_flags keeps them, and the window counted a whole tower lost. tower_hp is in video time, flags in match time."""
    last = {k: max(t for t, f in v if f > 0) - offset for k, v in tower_hp.items()
            if not k.endswith("king") and any(f > 0 for _, f in v)}
    out = []
    for t, f in towers:
        f = dict(f)
        for k, t_last in last.items():
            if f.get(k) is False and t <= t_last:
                f[k] = True
        out.append((t, f))
    return out


TOWER_KEYS = {"own_king": ("own", "king"), "own_l": ("own", "l"), "own_r": ("own", "r"),
              "opp_king": ("opp", "king"), "opp_l": ("opp", "l"), "opp_r": ("opp", "r")}
# A king's HP in princess-tower units. Was 6144 / 3584 = 1.714 (cr-api-data's level-1 ratio, 2400 / 1400) until
# 2026-10-29; the replays' own HP bars say 1.59: the highest king reading per princess max is 7728 / 4858, 7032 / 4424,
# 5832 / 3668 (1.59) and 4824 / 3052 (1.58, Liquipedia's level 11). The king grows slower per level than a princess
# tower (globals HITPOINT_INCREASE_PERCENT_PER_KING_LEVEL 7 vs _PER_TOWER_LEVEL 8). tower_hp's king max_hp was built
# with 1.714 (scripts/build_tower_hp.py), so a full king read 0.93: load_tower_hp re-scales it.
KING_PER_PRINCESS = 1.59
_KING_RATIO_IN_TABLE = 6144 / 3584


def load_tower_hp(game_id: str) -> dict[str, list[tuple[float, float]]]:
    """tower -> [(video t, HP fraction)] from `tower_hp` (accepted readings); {} if the match wasn't read."""
    try:
        rows = _rows("tower_hp", f"match_id = '{game_id}' AND accepted = true", ["t", "tower", "frac"])
    except Exception:
        return {}
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["t"]):
        if r["frac"] is not None:
            f = r["frac"] * (_KING_RATIO_IN_TABLE / KING_PER_PRINCESS if r["tower"].endswith("king") else 1.0)
            out[r["tower"]].append((r["t"], min(f, 1.0)))
    return dict(out)


def tower_frac(g: Game, tower: str, t: float) -> float | None:
    """The tower's HP fraction at video time t: the last accepted reading at or before t (None if none yet)."""
    if not tower.endswith("king"):  # a destroyed tower's bar is gone: the alive flags say 0
        flags = _towers_at(g, t - g.offset) or {}
        if flags.get(tower) is False:
            return 0.0
    hp = getattr(g, "tower_hp", {})
    v = hp.get(tower)
    if tower.endswith("king") and hp and (not v or v[0][0] > t):
        return 1.0  # a king shows no number until it is damaged: no reading yet = full
    if not v:
        return None
    i = bisect.bisect_right([x[0] for x in v], t) - 1
    return v[i][1] if i >= 0 else None


def _frame(g: Game, t: float) -> list[dict] | None:
    """The replay frame nearest t (within half a frame), or None."""
    i = bisect.bisect_left(g.frame_ts, t)
    near = [j for j in (i - 1, i) if 0 <= j < len(g.frame_ts)]
    j = min(near, key=lambda j: abs(g.frame_ts[j] - t), default=None)
    if j is not None and abs(g.frame_ts[j] - t) <= DET_DT / 2:
        return g.frames[g.frame_ts[j]]
    return None


def _play_spot(p: dict) -> tuple[float, float] | None:
    """Ground spot (bottom frame): `plays` x / y, else recovered from the detections (bot77.sim.spots)."""
    xy = (p["x"], p["y"]) if p["x"] is not None else p.get("spot")
    if xy is None:
        return None
    spell = sc.engine_cards().get(_engine_card(p) or -1, {}).get("is_spell")
    return xy if spell else (xy[0], xy[1] + FEET_DY)


def _engine_card(p: dict) -> int | None:
    return p.get("engine_card")


FALLING_S, FALLING_DROP = 3.0, 0.02  # a tower losing this share of its HP in the last FALLING_S before a window


def towers_falling(g: Game, t0: float) -> bool:
    """Whether a crown tower lost HP in the FALLING_S before video time t0. A unit hitting a tower from behind it is
    drawn behind the tower and not detected, so the board read empty while the tower fell (2026-11-15:
    bl_2026-10-29_r26 at 94.6 s, our left tower 1.0 -> 0.25 to a Knight the sim never had; the worst windows of the
    'sim deals too little to our towers' gap were these)."""
    for key in TOWER_KEYS:
        a, b = tower_frac(g, key, t0 - FALLING_S), tower_frac(g, key, t0)
        if a is not None and b is not None and a - b >= FALLING_DROP and b > 0:
            return True
    return False


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
        if towers_falling(g, t0):  # something unseen is already hitting a tower
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


def _badge_of(bar: dict, bars: list[dict]) -> dict | None:
    cy = (bar["box"][1] + bar["box"][3]) / 2
    for q in bars:
        if q["cls"] == "bar-level" and abs((q["box"][1] + q["box"][3]) / 2 - cy) < 15 \
                and q["box"][0] <= bar["box"][0] + 4 and q["box"][2] > bar["box"][0]:
            return q
    return None


def _bar_team(bar: dict, bars: list[dict]) -> str:
    """A bar's team is its level badge's when it has one: the detector calls a bar's team by its fill colour, and a
    just-hit bar is white (bl_2026-10-23_r25 50.5 s: our Giant's bar read 'opp', its badge 'own')."""
    badge = _badge_of(bar, bars)
    return badge["team"] if badge else bar["team"]


def bar_of(unit_box: tuple, team: str, bars: list[dict]) -> tuple[tuple, tuple | None] | None:
    """The HP bar (and its level badge) over a unit: same team, the bar's centre within the unit box's width, its
    bottom from 45 px above the box's top to 30 px into it (the bar hangs over the unit's head)."""
    x0, y0, x1, _ = unit_box
    best, bd = None, 1e9
    for b in bars:
        if b["cls"] != "bar" or _bar_team(b, bars) != team:
            continue
        bx0, by0, bx1, by1 = b["box"]
        cx = (bx0 + bx1) / 2
        if x0 - 10 <= cx <= x1 + 10 and y0 - 45 <= by1 <= y0 + 30:
            d = abs(cx - (x0 + x1) / 2) + abs(by1 - y0)
            if d < bd:
                best, bd = b, d
    if best is None:
        return None
    badge = _badge_of(best, bars)
    return best["box"], (badge["box"] if badge else None)


def with_bar_hp(g: Game, t: float, units: list[dict]) -> list[dict]:
    """The units of the frame nearest t, each with hp_frac read off its HP bar (none: full, the bar shows only once
    a unit is damaged; 2026-11-15)."""
    from bot77.calibrate import read_frame
    from bot77.readers.unit_hp import bar_fill
    i = bisect.bisect_left(g.frame_ts, t)
    ts = [g.frame_ts[j] for j in (i - 1, i) if 0 <= j < len(g.frame_ts)]
    if not ts:
        return units
    tf = min(ts, key=lambda x: abs(x - t))
    bars = getattr(g, "bars", {}).get(tf, [])
    if not any(b["cls"] == "bar" for b in bars):
        return units
    video = Path("data/raw/replay_links") / f"{g.game_id.rsplit('_m', 1)[0]}.mp4"
    try:
        frame = read_frame(video, tf)
    except Exception:  # noqa: BLE001  (video not on disk)
        return units
    out, free = [dict(u) for u in units], list(bars)
    # each bar to one unit, closest pairs first (overlapping units would otherwise share one)
    pairs = []
    for i, u in enumerate(out):
        hit = bar_of(u["box"], u["team"], free) if u.get("box") else None
        if hit:
            pairs.append((abs((hit[0][0] + hit[0][2]) / 2 - (u["box"][0] + u["box"][2]) / 2) + abs(hit[0][3] - u["box"][1]), i))
    for _, i in sorted(pairs):
        u = out[i]
        hit = bar_of(u["box"], u["team"], free)
        if not hit:
            continue
        free = [b for b in free if b["box"] != hit[0]]
        f = bar_fill(frame, hit[0], hit[1])
        if f is not None:
            u["hp_frac"] = f
    return out


def _towers_at(g: Game, t_match: float) -> dict | None:
    i = bisect.bisect_right([t for t, _ in g.towers], t_match) - 1
    return g.towers[i][1] if i >= 0 else None


def load_levels(game_id: str, decks: dict[str, list[int]]) -> tuple[dict[str, dict[int, int]], dict[str, float]]:
    """Card levels per side {side: {engine card id: level}} from `card_levels` (the Battle Log entry; 2026-11-15) and
    each side's princess tower max HP from `tower_hp` (the replay's own HP bars). Empty where not read."""
    vid = game_id.rsplit("_m", 1)[0].removesuffix("_top")
    try:
        rows = _rows("card_levels", f"video_id = '{vid}'", ["side", "card", "level"])
    except Exception:  # noqa: BLE001  (table not built yet)
        rows = []
    by = {(r["side"], r["card"]): r["level"] for r in rows if r["card"] and r["level"]}
    levels = {}
    for side, deck in decks.items():
        levels[side] = {e: by[(side, sc.our_name(e))] for e in deck if (side, sc.our_name(e)) in by}
    mx = collections.defaultdict(list)
    try:
        for r in _rows("tower_hp", f"match_id = '{game_id}' AND accepted = true", ["tower", "max_hp"]):
            if r["max_hp"] and not r["tower"].endswith("king"):
                mx[r["tower"].split("_")[0]].append(r["max_hp"])
    except Exception:  # noqa: BLE001
        pass
    tower_max = {k: float(np.median(v)) for k, v in mx.items() if v}
    return levels, tower_max


def run_window(g: Game, w: dict, towers: str = "liquipedia_l11", wake_kings: bool = True,
               levels: bool = False, per_body: bool = True) -> dict:
    """Play the window in the sim. -> {'frames': [(dt, sim units, replay units)], 'plays': [PlayResult], 'sim'}.
    Sim tick = (video t - offset) x 10; a replay frame is compared with the sim at its nearest tick."""
    sim = Sim.new(g.decks["bottom"], g.decks["top"], towers=towers)
    if levels:  # the replay's card levels; unread cards stay at the engine's 11
        for side, team in (("bottom", "own"), ("top", "opp")):
            sim.set_levels(team, getattr(g, "levels", {}).get(side, {}))
    sim.reset()
    if levels:  # towers at the replay's level: its princess max HP against the preset's
        from bot77.sim.engine import TOWER_PRESETS
        preset = TOWER_PRESETS[towers]
        for team, mxhp in getattr(g, "tower_max", {}).items():
            if preset and mxhp:
                sim.scale_towers(team, mxhp / preset[1][0])
    tick = lambda t: round((t - g.offset) * 10)  # noqa: E731
    sim.env.set_current_tick(tick(w["t0"] - PRE_S))
    for team in ("own", "opp"):
        sim.set_elixir(team, 10.0)
    tw = _towers_at(g, sim.t) or {}
    for key, team, slot in (("own_l", "own", "l"), ("own_r", "own", "r"), ("opp_l", "opp", "l"), ("opp_r", "opp", "r")):
        if tw.get(key) is False:
            sim.set_tower(team, slot, 0.0)
    for key, (team, slot) in TOWER_KEYS.items():  # measured HP (tower_hp) where read; full otherwise
        f = tower_frac(g, key, w["t0"] - PRE_S)
        if f is not None and 0 < f < 0.999 and sim.towers()[key]["hp"] > 0:
            sim.set_tower(team, slot, f)
    if wake_kings:
        for team in ("own", "opp"):
            t_awake = getattr(g, "king_awake", {}).get(team)
            if t_awake is not None and t_awake < w["t0"]:
                sim.wake_king(team)
    towers0 = {k: v["frac"] for k, v in sim.towers().items()}
    start_units = _frame(g, w["t0"] - PRE_S) or []
    if per_body and start_units:
        start_units = with_bar_hp(g, w["t0"] - PRE_S, start_units)
    injected, _ = place_units(sim, start_units, per_body=per_body)  # per_body: one detected Goblin is one Goblin
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
    return {"frames": frames, "plays": results, "sim": sim, "injected": injected, "towers0": towers0}


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
# ...where frames up to this far apart count as one run (2026-10-29): the detector misses units for a frame or
# three mid-fight (Valkyrie seen in 6 of 21 frames of its life), and a strict run of consecutive frames read their
# death seconds early. Moved the held-back life bias -0.26 s (engine 5e7711b, 1.82 -> 1.56 mean).
RUN_GAP_S = 1.0
# Cards the detector loses while they are alive: Tesla under the ground between targets (its HP bar stays),
# Royal Ghost and Suspicious Bush invisible. Their replay "death" is not one; left out of life_diff.
HIDES = {"Tesla", "Royal Ghost", "Suspicious Bush"}
FOREIGN_FRAMES = 6


def _last_run(dts: list[float], min_run: int, gap: float = DET_DT * 1.5) -> float | None:
    """The end of the last run of >= min_run frames in sorted times, frames up to `gap` apart counting as one run."""
    dts = sorted(dts)
    best, run_start, prev, n = None, None, None, 0
    for t in dts:
        if prev is not None and t - prev <= gap + 1e-6:
            n += 1
        else:
            run_start, n = t, 1
        if n >= min_run:
            best = t
        prev = t
    return best


TOWER_REACH = 4.5  # engine tiles from a crown tower's centre: a unit this close is fighting it (occluded)


def at_enemy_tower(points: list[tuple[float, float]], team: str) -> bool:
    """Whether any of these points (our frame) is within TOWER_REACH of a crown tower of the team `team` attacks."""
    from bot77.sim.geometry import ours_to_engine
    import clash_royale_env as cre
    enemy = 0 if team == "opp" else 1
    towers = [(3.0, cre.arena_princess_y(enemy)), (14.0, cre.arena_princess_y(enemy)), (8.5, cre.arena_king_y(enemy))]
    for x, y in points:
        ex, ey = ours_to_engine(x, y)
        if any((ex - tx) ** 2 + (ey - ty) ** 2 <= TOWER_REACH ** 2 for tx, ty in towers):
            return True
    return False


def metrics(g: Game, w: dict, r: dict) -> list[dict]:
    """One row per (metric, card, dt bin) for this window."""
    rows = []
    base = {"match_id": g.game_id, "split": g.split, "source": g.source, "t0": w["t0"], "t1": w["t1"],
            "n_plays": len(w["plays"]), "start_units": w["start_units"]}
    errs = collections.defaultdict(list)  # (card, bin) -> distances
    dx_dy = collections.defaultdict(list)
    count_err = collections.defaultdict(list)
    lane_same = collections.defaultdict(list)
    present = collections.defaultdict(lambda: {"sim": [], "det": [], "det_pos": []})  # (card, team) -> dts present
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
                present[k]["det_pos"] += [(dt, u["x"], u["y"]) for u in du if (u["card"], u["team"]) == k]
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
    # Bodies one side had and the other never did (2026-10-29): a replay unit (in a run of MIN_RUN frames) of a
    # card / team the sim never had is a unit the window didn't give the sim (a start unit the detector missed, a
    # play outside the window); a sim unit is a fight the replay didn't have. Either changes every life and winner
    # in the window, so the life / winner numbers are also read on windows where both are 0.
    sim_keys = {k for k, p in present.items() if p["sim"]}
    det_keys = {k for k, p in present.items() if _last_run(p["det"], MIN_RUN, RUN_GAP_S) is not None}
    rows.append({**base, "metric": "foreign_det", "card": "", "dt_bin": -1, "value": float(len(det_keys - sim_keys)),
                 "n": 1})
    rows.append({**base, "metric": "foreign_sim", "card": "", "dt_bin": -1, "value": float(len(sim_keys - det_keys)),
                 "n": 1})
    # ...and a replay card seen in at least FOREIGN_FRAMES frames (1.5 s): not a one-off mislabel
    solid = {k for k, p in present.items() if len(p["det"]) >= FOREIGN_FRAMES}
    rows.append({**base, "metric": "foreign_det_solid", "card": "", "dt_bin": -1,
                 "value": float(len(solid - sim_keys)), "n": 1})
    played = {(p["card"], SIDE[p["side"]]) for p in w["plays"]}
    for (card, team), p in present.items():
        if (card, team) not in played or card in HIDES:  # start units' HP is unknown; hiders vanish alive
            continue
        ls, ld = _last_run(p["sim"], 1), _last_run(p["det"], MIN_RUN, RUN_GAP_S)
        if ls is None or ld is None:
            continue
        if min(ls, ld) < end - 1.0:  # at least one side lost it before the window ended
            rows.append({**base, "metric": "life_diff", "card": card, "dt_bin": -1, "value": ls - ld, "n": 1})
        # Both sides lost it before the window ended (2026-10-29): the replay's detector loses units early, never
        # late, while the sim's death is exact, so a window where only the replay "died" reads the detector's
        # dropouts as sim units living too long. This one is censored on neither side.
        if max(ls, ld) < end - 1.0:
            rows.append({**base, "metric": "life_diff_both", "card": card, "dt_bin": -1, "value": ls - ld, "n": 1})
            # ...and the replay last saw it away from the towers it attacks (2026-11-15): a unit hitting a tower is
            # drawn behind it and the detector loses it alive (bl_2026-10-23_r13: Goblins last seen 19 s, our tower
            # kept losing HP to 28 s), so life_diff_both reads tower fights as the sim's units living too long.
            if not at_enemy_tower([(x, y) for t, x, y in p["det_pos"] if t >= ld - 0.5], team):
                rows.append({**base, "metric": "life_diff_field", "card": card, "dt_bin": -1, "value": ls - ld,
                             "n": 1})
    if r["frames"]:
        last = [f for f in r["frames"] if f[0] >= end - 1.0]  # the last second, averaged (detector flicker)
        vs = {k: np.mean([value(f[1])[k] for f in last]) for k in ("own", "opp")}
        vd = {k: np.mean([value(f[2])[k] for f in last]) for k in ("own", "opp")}
        ws = np.sign(round(vs["own"] - vs["opp"]))
        wd = np.sign(round(vd["own"] - vd["opp"]))
        both = any(p["side"] == "bottom" for p in w["plays"]) and any(p["side"] == "top" for p in w["plays"])
        rows.append({**base, "metric": "winner_agree" if both else "winner_agree_onesided", "card": "", "dt_bin": -1,
                     "value": float(ws == wd), "n": 1})
        # the margins themselves (2026-11-16): a near-tie flips winner_agree on one missed unit
        rows.append({**base, "metric": "margin_sim", "card": "", "dt_bin": -1, "value": vs["own"] - vs["opp"], "n": 1})
        rows.append({**base, "metric": "margin_det", "card": "", "dt_bin": -1, "value": vd["own"] - vd["opp"], "n": 1})
        rows.append({**base, "metric": "value_diff_own", "card": "", "dt_bin": -1, "value": vs["own"] - vd["own"], "n": 1})
        rows.append({**base, "metric": "value_diff_opp", "card": "", "dt_bin": -1, "value": vs["opp"] - vd["opp"], "n": 1})
    if getattr(g, "tower_hp", None) and r["frames"]:
        t_end = w["t0"] + r["frames"][-1][0]
        sim_end = r["sim"].towers()
        for side in ("own", "opp"):  # damage TO this side's towers, in princess-tower units
            real = sim_ = 0.0
            known = True
            for key in (f"{side}_king", f"{side}_l", f"{side}_r"):
                f0, f1 = tower_frac(g, key, w["t0"] - PRE_S), tower_frac(g, key, t_end)
                if f0 is None or f1 is None:
                    known = False
                    break
                k = KING_PER_PRINCESS if key.endswith("king") else 1.0
                real += (f0 - f1) * k
                # the sim's own King / Princess ratio (the tower preset's), not the replay's. A destroyed Princess reads
                # no max, so any standing one of either side's (one preset); with both of a side's down it read 1 and
                # the King's damage counted ~4000x (2026-10-30: the pocket's newly clean windows start with towers down)
                pmax = max(sim_end[f"{s}_{lr}"]["max"] for s in ("own", "opp") for lr in ("l", "r"))
                ks = (sim_end[key]["max"] / pmax if pmax > 0 else KING_PER_PRINCESS) if key.endswith("king") else 1.0
                sim_ += (r["towers0"][key] - sim_end[key]["frac"]) * ks
            if known:
                rows.append({**base, "metric": f"tower_dmg_{side}_real", "card": "", "dt_bin": -1, "value": real,
                             "n": 1})
                rows.append({**base, "metric": f"tower_dmg_{side}_sim", "card": "", "dt_bin": -1, "value": sim_,
                             "n": 1})
    pr = r["plays"]
    rows.append({**base, "metric": "plays_refused", "card": "", "dt_bin": -1, "value": float(sum(not p.played for p in pr)),
                 "n": len(pr)})
    rows.append({**base, "metric": "spot_moved", "card": "", "dt_bin": -1,
                 "value": float(np.mean([p.moved for p in pr])) if pr else 0.0, "n": len(pr)})
    return rows


def noise_floor(g: Game) -> list[float]:
    """Detector jitter: distances (our units) of a standing building's detections from its own median position, for
    every building seen in >= 8 frames at one spot. The replay's own noise, to read the position errors against."""
    pts: dict[tuple, list[tuple[float, float]]] = collections.defaultdict(list)
    for t in g.frame_ts:
        for u in g.frames[t]:
            e = sc.engine_id(u["card"])
            if e is None or not sc.engine_cards()[e]["is_building"]:
                continue
            # one building per card, team and 2-tile cell (two of a card at once sit apart)
            pts[(u["card"], u["team"], round(u["x"] / 2), round(u["y"] / 2))].append((u["x"], u["y"]))
    out = []
    for v in pts.values():
        if len(v) >= 8:
            a = np.array(v)
            out += list(np.hypot(*(a - np.median(a, axis=0)).T))
    return out
