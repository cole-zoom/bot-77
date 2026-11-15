"""A simulated match set up from one of our moments (docs/simulator.md S1, 2026-10-23).

`scenario_from_moment(moment_id)` -> Scenario(sim, notes). The deciding player is 'own' (engine team 0) and the
moment's frame is used as is (moments are stored in the deciding player's frame). Set from the moment: the clock,
both elixir bars, our hand, both decks (matches.deck; Evolutions / Heroes the player was seen playing), the towers
standing (and their HP when `tower_hp` is given), the tower troops seen, and every unit on the board.

What can't be carried over (`notes` says which applied): unit HP (injected at full), what a unit was doing (mid-attack,
its target, a charge), units still deploying (injected as deployed), spells on the board (skipped: a detected spell
would be cast again), the order of the opponent's queue and of ours past `next`, and the king tower's wake-up from
damage (it wakes when a princess tower is down). The opponent's elixir is estimated from their last play: elixir after
it, plus the real regeneration since, capped at 10.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import lancedb

from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.sim.geometry import box_to_feet

DB = "data/lancedb"
REAL_ELIXIR_S = 2.9  # seconds per elixir at 1x: our replays (idle moments 1 s apart, 23k pairs, IQR 2.86-2.94); engine 2.857
SQUAD_RADIUS = 2.5  # bodies of one card closer than this are one deploy
SPELL_CLASSES = ("tornado", "poison", "freeze", "rage", "earthquake", "graveyard", "zap", "fireball", "arrows",
                 "rocket", "the-log", "barbarian-barrel", "giant-snowball", "lightning", "void", "goblin-curse",
                 "royal-delivery", "clone", "vines")


@dataclass
class Scenario:
    sim: Sim
    moment: dict
    notes: dict = field(default_factory=dict)


def _db():
    return lancedb.connect(DB)


def _one(table: str, where: str, columns: list[str] | None = None) -> dict | None:
    q = _db().open_table(table).search().where(where)
    if columns:
        q = q.select(columns)
    rows = q.limit(1).to_arrow().to_pylist()
    return rows[0] if rows else None


def _rows(table: str, where: str, columns: list[str] | None = None, limit: int = 100_000) -> list[dict]:
    q = _db().open_table(table).search().where(where)
    if columns:
        q = q.select(columns)
    return q.limit(limit).to_arrow().to_pylist()


@lru_cache(maxsize=4096)
def _squad_size(card: int) -> int:
    """Bodies one engine card spawns (Skeletons 3, Skeleton Army 15, ...), measured by injection."""
    s = Sim.new([card], [card])
    s.reset()
    s.inject(card, 9.0, 20.0, "own")
    s.step(1)
    return max(1, len(s.units()))


def side_decks(game_id: str) -> dict[str, list[int]]:
    """{'bottom': [Supercell ids], 'top': [...]} from the two `matches` rows of a game."""
    out = {}
    for r in _rows("matches", f"match_id = '{game_id}' OR pair_match_id = '{game_id}'",
                   ["match_id", "pair_match_id", "pov_side", "deck"]):
        if r["match_id"] == game_id or r["pair_match_id"] == game_id:
            out[r["pov_side"]] = list(r["deck"] or [])
    return out


def _engine_deck(deck: list[int], plays: list[dict]) -> tuple[list[int], dict[int, int], list[str]]:
    """Our deck -> engine ids (an Evolution / Hero if the player played that form), {our id: engine id}, and the
    cards the engine lacks."""
    forms = {}
    for p in plays:
        if p["card_id"] and p["form"] in ("evo", "hero"):
            forms[p["card_id"]] = p["form"]
    for p in plays:  # cards played that the deck row missed (decks are sometimes 7 long)
        if p["card_id"] and p["card_id"] not in deck and p["kind"] == "play":
            deck = deck + [p["card_id"]]
    ids, missing = {}, []
    for c in deck:
        e = sc.engine_id(c, forms.get(c, "normal")) or sc.engine_id(c)
        if e is None:
            missing.append(sc.our_cards().get(c, str(c)))
        else:
            ids[c] = e
    return list(dict.fromkeys(ids.values())), ids, missing


def _opp_elixir(plays: list[dict], t_match: float, multiplier_at) -> tuple[float, str]:
    before = [p for p in plays if p["t_match"] <= t_match and p["elixir_after"] is not None]
    if not before:
        return 5.0 + t_match / REAL_ELIXIR_S, "start + regen"  # nothing played yet (capped below)
    last = max(before, key=lambda p: p["t_match"])
    e, t = last["elixir_after"], last["t_match"]
    while t < t_match:  # 0.1 s steps so a multiplier change mid-way counts
        e += 0.1 * multiplier_at(t) / REAL_ELIXIR_S
        t += 0.1
    return e, f"last play at {last['t_match']:.1f} s + regen"


def real_multiplier(t_match: float) -> int:
    """The elixir multiplier our replays show (moments.multiplier, 2026-10-23): 2x from 2:00, 3x from 4:00."""
    return 1 if t_match < 120 else 2 if t_match < 240 else 3


@lru_cache(maxsize=4096)
def _independent_bodies(card: int) -> bool:
    """A card of several bodies that are all the same troop and carry nothing (Goblins, Skeleton Army, Barbarians):
    one of them can stand on the board alone. Not a mount and rider (Ram Rider) or a mixed squad."""
    s = Sim.new([card], [card])
    s.reset()
    s.inject(card, 9.0, 20.0, "own")
    s.step(1)
    us = s.units()
    return len(us) > 1 and len({u["engine_name"] for u in us}) == 1 and all(u["attached_to"] < 0 for u in us)


ONE_BODY_R = 1.0  # our units: two detections of a one-body card this close are one unit


@lru_cache(maxsize=4096)
def _body_max_hp(card: int, level: int | None) -> float:
    """One body's full HP of an engine card at a level (None: the engine's 11)."""
    s = Sim.new([card], [card])
    if level:
        s.set_levels("own", {card: level})
    s.reset()
    s.inject(card, 9.0, 20.0, "own", bodies=1)
    s.step(1)
    us = s.units()
    return float(us[0]["hp"]) if us else 0.0


def place_units(sim: Sim, units: list[dict], per_body: bool = False) -> tuple[list[tuple], list[tuple]]:
    """Inject detected units (card, cls, team own/opp, x, y = FEET) as deployed, full HP. Bodies of one card within
    SQUAD_RADIUS are one deploy: one card per cluster (so a half-dead squad comes back whole). Spells are skipped.
    per_body: a one-body card or a card of independent bodies (_independent_bodies) is injected one body per
    detection, where it was seen, at its `hp_frac` of full HP when the unit carries one (its HP bar; none = full),
    instead (the fidelity check, 2026-11-15: a whole card per cluster put 4 Goblins for one flicker, and a Mini
    P.E.K.K.A that had fought came back fresh). Off by default: the live mirror and the imitation datasets were built
    with whole cards.
    -> (injected: (card, team, x, y, bodies seen, bodies injected), skipped: (card, why))."""
    injected, skipped, groups = [], [], []
    for u in units:
        cls = u.get("cls") or ""
        if cls in SPELL_CLASSES or any(cls.startswith(s) for s in SPELL_CLASSES):
            skipped.append((u["card"], "spell"))
            continue
        e = sc.engine_id(u["card"], "evo" if cls.endswith("-evo") else "hero" if cls.endswith("-hero")
                         else "normal") or sc.engine_id(u["card"])
        if e is None:
            skipped.append((u["card"], "not in engine"))
            continue
        x, y = u["x"], u["y"]
        for g in groups:
            if g["card"] == e and g["team"] == u["team"] and ((g["x"] - x) ** 2 + (g["y"] - y) ** 2) ** 0.5 < SQUAD_RADIUS:
                g["pts"].append((x, y))
                g["hp"].append(u.get("hp_frac"))
                break
        else:
            groups.append({"card": e, "team": u["team"], "x": x, "y": y, "pts": [(x, y)], "hp": [u.get("hp_frac")]})
    for g in groups:
        n = len(g["pts"])
        if per_body and (_squad_size(g["card"]) == 1 or _independent_bodies(g["card"])):
            lv = getattr(sim, "levels", {}).get(g["team"], {}).get(g["card"])
            pts = list(zip(g["pts"], g["hp"]))
            if _squad_size(g["card"]) == 1:  # one body: detections within ONE_BODY_R of each other are the same unit
                kept = []
                for (x, y), f in pts:  # (bl_2026-10-29_r19 71 s: one Battle Ram detected twice, injected twice)
                    if all((x - a) ** 2 + (y - b) ** 2 >= ONE_BODY_R ** 2 for (a, b), _ in kept):
                        kept.append(((x, y), f))
                pts = kept
            for (x, y), f in pts:
                hp = None if f is None else max(1.0, f * _body_max_hp(g["card"], lv))
                sim.inject(g["card"], x, y, g["team"], hp=hp, bodies=1)
            injected.append((sc.our_name(g["card"]), g["team"], round(g["x"], 2), round(g["y"], 2), n, len(pts)))
            continue
        k = -(-n // _squad_size(g["card"]))  # injections for this cluster
        cx, cy = sum(p[0] for p in g["pts"]) / n, sum(p[1] for p in g["pts"]) / n
        for _ in range(k):
            sim.inject(g["card"], cx, cy, g["team"])
        injected.append((sc.our_name(g["card"]), g["team"], round(cx, 2), round(cy, 2), n, k * _squad_size(g["card"])))
    return injected, skipped


def scenario_from_moment(moment_id: str, tower_hp: dict[str, float] | None = None) -> Scenario:
    """`tower_hp`: optional fractions {'own_l': 0.4, 'opp_king': 1.0, ...} (S2's tower_hp table)."""
    m = _one("moments", f"moment_id = '{moment_id}'")
    if m is None:
        raise KeyError(moment_id)
    own_side = m["side"]
    opp_side = "top" if own_side == "bottom" else "bottom"
    plays = _rows("plays", f"game_id = '{m['game_id']}' AND source = 'hand'",  # both sides (top: its own match_id)
                  ["side", "t_match", "card_id", "form", "kind", "elixir_after"])
    decks = side_decks(m["game_id"])
    own_deck, own_ids, own_missing = _engine_deck(decks.get(own_side, []), [p for p in plays if p["side"] == own_side])
    opp_deck, opp_ids, opp_missing = _engine_deck(decks.get(opp_side, []), [p for p in plays if p["side"] == opp_side])
    units = [u for u in m["units"] or [] if "tower" not in (u["cls"] or "") and "bar" not in (u["cls"] or "")]
    towers_cls = {u["team"]: u["cls"] for u in m["units"] or [] if (u["cls"] or "").endswith("-tower")
                  and u["cls"] in sc.TOWER_TROOPS and "king" not in u["cls"]}
    notes = {"missing_cards": own_missing + opp_missing, "unit_hp": "full", "deploying": "injected as deployed"}

    sim = Sim.new(own_deck, opp_deck, towers_cls.get("own", "queen-tower"), towers_cls.get("opp", "queen-tower"))
    sim.reset()
    sim.set_clock(m["t_match"])
    notes["engine_multiplier"] = sim.env.get_elixir_multiplier()
    notes["moment_multiplier"] = m["multiplier"]

    hand = [own_ids.get(c) for c in (m["hand"] or [])]
    notes["hand_set"] = sim.set_hand("own", [c for c in hand if c is not None])
    notes["hand_read"] = len(m["hand"] or [])
    sim.set_elixir("own", m["elixir"])
    opp_e, how = _opp_elixir([p for p in plays if p["side"] == opp_side], m["t_match"], real_multiplier)
    sim.set_elixir("opp", min(10.0, opp_e))
    notes["opp_elixir"] = (round(min(10.0, opp_e), 2), how)

    tw = m["towers"] or {}
    for key, team, slot in (("own_l", "own", "l"), ("own_r", "own", "r"), ("opp_l", "opp", "l"), ("opp_r", "opp", "r")):
        if tw.get(key) is False:
            sim.set_tower(team, slot, 0.0)
    for key, frac in (tower_hp or {}).items():
        team, slot = key.split("_")
        if frac > 0:
            sim.set_tower(team, slot, frac)
    notes["tower_hp"] = "from tower_hp" if tower_hp else "alive flags only (standing towers at full HP)"

    injected, skipped = place_units(sim, [{**u, "y": box_to_feet(u["x"], u["y"])[1]} for u in units])
    sim.step(1)  # commit the injected units (one tick)
    notes["injected"] = injected  # (card, team, x, y, bodies detected, bodies injected)
    notes["skipped"] = skipped
    return Scenario(sim, m, notes)
