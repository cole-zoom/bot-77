"""Turn per-frame unit detections into deploy events for both teams.

A deploy puts one or more new units on the arena at once (a Knight, three Skeletons, a
Skeleton Army). So: track units over time per team; a unit that appears, persists, and
isn't a broken-off track resuming is a birth; births of one team close together in time
and space are one deploy. Detections come from KataCR's detectors [Wu et al. 2025, §5.2];
positions are in tiles of the 18-wide arena grid [Wu et al. 2025, §2.3].
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from bot77.detect.names import DEATH_OR_SUMMON, EVO_SYMBOLS, class_to_card

MIN_CONF = 0.5
TRACK_RADIUS_TILES = 1.6  # a detection continues a track within this distance ...
TRACK_SPEED_TILES_S = 2.5  # ... plus this much per second since the track was last seen
TRACK_MAX_GAP_S = 1.5  # tracks unseen for longer end
RESUME_S, RESUME_TILES = 6.0, 5.0  # a birth this close to a same-card track that just ended resumes it
# (units jump - Firecracker's recoil, Bandit's dash - and detections drop out; replaying the
# same card within 6 s is rare even in fast cycle decks)
BUILDING_RESUME_S = 90.0  # buildings (e.g. Tesla hiding underground) resume from much longer ago
CONFIRM_S, CONFIRM_DETS = 1.0, 2  # a birth needs this many detections within this long
GROUP_S, GROUP_TILES = 1.0, 3.0  # births within this window and distance are one deploy
EVO_SYMBOL_S, EVO_SYMBOL_TILES = 1.5, 3.0
RIVER_Y_TILES = 14.9  # river row in KataCR arena-crop tiles (Alina's crop; the same crop geometry for every layout)
OWN_HALF_MARGIN = 0.8
# Units that keep spawning others. A birth of a spawned card near a live spawner of the same
# team is a spawn, not a card being played.
SPAWNERS = {
    "Goblin Hut": {"Spear Goblins"}, "Tombstone": {"Skeletons"}, "Furnace": {"Fire Spirit"},
    "Barbarian Hut": {"Barbarians"}, "Witch": {"Skeletons"}, "Night Witch": {"Bats"},
    "Mother Witch": {"Minions"}, "Goblin Cage": {"Goblin Cage"}, "Graveyard": {"Skeletons"},
    "Skeleton King": {"Skeletons"}, "Goblin Giant": {"Spear Goblins"}, "Elixir Golem": {"Elixir Golem"},
    "Golem": {"Golem"}, "Lava Hound": {"Lava Hound"}, "Phoenix": {"Phoenix"}, "Goblin Drill": {"Goblins"},
    "Goblin Barrel": {"Goblins"}, "Skeleton Barrel": {"Skeletons"}, "Royal Delivery": {"Royal Recruits"},
    "Barbarian Barrel": {"Barbarians"},
}
SPAWN_RADIUS_TILES = 4.0  # troops and buildings can only be placed on the owner's half (plus a little slack)


@dataclass
class Track:
    team: str
    card: str
    form: str
    cls: str
    first_t: float
    last_t: float
    x: float
    y: float
    dets: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    resumed: bool = False
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # detector boxes, same order as dets


@dataclass
class Deploy:
    team: str
    t: float  # video time of the first unit seen
    x: float  # tiles
    y: float
    card: str | None  # best guess
    card_votes: dict[str, int]
    units: int
    evo: bool
    subunit_only: bool  # only death spawns / summons (probably not a card being played)
    classes: list[str]


def track_units(dets: list[dict], card_names: list[str], tile_px: float, is_building) -> list[Track]:
    """`dets` rows: t_video, cls, team, conf, x0..y1 (frame pixels). Returns all tracks."""
    frames: dict[float, list[dict]] = {}
    for d in dets:
        if d["conf"] >= MIN_CONF:
            frames.setdefault(round(d["t_video"], 3), []).append(d)
    active: list[Track] = []
    ended: list[Track] = []
    for t in sorted(frames):
        still = []
        for tr in active:
            (still if t - tr.last_t <= TRACK_MAX_GAP_S else ended).append(tr)
        active = still
        claimed: set[int] = set()
        for d in sorted(frames[t], key=lambda d: -d["conf"]):
            card, form = class_to_card(d["cls"], card_names)
            if card is None:
                continue
            x = (d["x0"] + d["x1"]) / 2 / tile_px
            y = (d["y0"] + d["y1"]) / 2 / tile_px
            best, best_cost = None, None
            for k, tr in enumerate(active):
                if k in claimed or tr.team != d["team"]:
                    continue
                dist = ((tr.x - x) ** 2 + (tr.y - y) ** 2) ** 0.5
                if dist > TRACK_RADIUS_TILES + TRACK_SPEED_TILES_S * (t - tr.last_t):
                    continue
                cost = dist + (0 if tr.card == card else 1.0)  # prefer the same card
                if best is None or cost < best_cost:
                    best, best_cost = k, cost
            if best is not None:
                tr = active[best]
                claimed.add(best)
                tr.last_t, tr.x, tr.y = t, x, y
                tr.dets.append((t, x, y))
                tr.boxes.append((d["x0"], d["y0"], d["x1"], d["y1"]))
                continue
            tr = Track(d["team"], card, form, d["cls"], t, t, x, y, [(t, x, y)], boxes=[(d["x0"], d["y0"], d["x1"], d["y1"])])
            tr.resumed = any(
                o.team == tr.team and o.card == card and
                ((t - o.last_t <= RESUME_S and ((o.x - x) ** 2 + (o.y - y) ** 2) ** 0.5 <= RESUME_TILES) or
                 (is_building(card) and t - o.last_t <= BUILDING_RESUME_S and ((o.x - x) ** 2 + (o.y - y) ** 2) ** 0.5 <= 1.5))
                for o in ended + active)
            active.append(tr)
            claimed.add(len(active) - 1)
    return ended + active


def _on_own_half(tr: Track) -> bool:
    y = tr.dets[0][2]
    return y >= RIVER_Y_TILES - OWN_HALF_MARGIN if tr.team == "friendly" else y <= RIVER_Y_TILES + OWN_HALF_MARGIN


def _spawned(tr: Track, tracks: list[Track]) -> bool:
    """Born next to a live spawner of the same team that produces this card."""
    for o in tracks:
        if o is tr or o.team != tr.team or tr.card not in SPAWNERS.get(o.card, ()):
            continue
        if o.first_t < tr.first_t <= o.last_t + 0.5 and ((o.x - tr.x) ** 2 + (o.y - tr.y) ** 2) ** 0.5 <= SPAWN_RADIUS_TILES:
            return True
    return False


def deploys(dets: list[dict], card_names: list[str], tile_px: float, is_building, is_spell=lambda c: False,
            origin: tuple[float, float] = (0.0, 0.0)) -> list[Deploy]:
    """`origin`: the detector crop's top-left in frame pixels, so positions are arena tiles."""
    ox, oy = origin
    dets = [{**d, "x0": d["x0"] - ox, "x1": d["x1"] - ox, "y0": d["y0"] - oy, "y1": d["y1"] - oy} for d in dets]
    tracks = track_units(dets, card_names, tile_px, is_building)
    # Troops and buildings are placed on the owner's half; a unit first seen past the river
    # walked, dashed or resurfaced there (e.g. Mighty Miner's ability). Spells go anywhere.
    births = [tr for tr in tracks if not tr.resumed and
              sum(1 for t, _, _ in tr.dets if t - tr.first_t <= CONFIRM_S) >= CONFIRM_DETS and
              (is_spell(tr.card) or _on_own_half(tr)) and not _spawned(tr, tracks)]
    births.sort(key=lambda tr: tr.first_t)
    symbols = [((d["x0"] + d["x1"]) / 2 / tile_px, (d["y0"] + d["y1"]) / 2 / tile_px, d["t_video"], d["team"])
               for d in dets if d["cls"] in EVO_SYMBOLS and d["conf"] >= MIN_CONF]
    groups: list[list[Track]] = []
    for tr in births:
        for g in groups:
            g0 = g[0]
            if (g0.team == tr.team and tr.first_t - g0.first_t <= GROUP_S and
                    ((g0.x - tr.x) ** 2 + (g0.y - tr.y) ** 2) ** 0.5 <= GROUP_TILES):
                g.append(tr)
                break
        else:
            groups.append([tr])
    out = []
    for g in groups:
        votes = Counter()
        for tr in g:
            votes[tr.card] += len(tr.dets)
        x = sum(tr.dets[0][1] for tr in g) / len(g)
        y = sum(tr.dets[0][2] for tr in g) / len(g)
        t = g[0].first_t
        evo = any(tr.form == "evo" for tr in g) or any(
            team == g[0].team and abs(st - t) <= EVO_SYMBOL_S and ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5 <= EVO_SYMBOL_TILES
            for sx, sy, st, team in symbols)
        out.append(Deploy(g[0].team, t, x, y, votes.most_common(1)[0][0], dict(votes), len(g), evo,
                          all(tr.cls in DEATH_OR_SUMMON for tr in g), sorted({tr.cls for tr in g})))
    return out


def apply_deck(events: list[Deploy], deck: set[str]) -> list[Deploy]:
    """Re-pick each deploy's card among the deck (detector classes outside it are ignored);
    deploys with no deck card among their votes keep card=None."""
    for e in events:
        in_deck = {c: v for c, v in e.card_votes.items() if c in deck}
        e.card = max(in_deck, key=in_deck.get) if in_deck else None
    return events


def infer_deck(events: list[Deploy], size: int = 8) -> list[str]:
    """The opponent's deck: cards that won the most deploys (each deploy counts once)."""
    wins = Counter(e.card for e in events if e.card and not e.subunit_only)
    return [c for c, _ in wins.most_common(size)]


# Cards made of several units the detector knows individually, told apart by count.
COMPOSITES = [
    ("Skeletons", 6, "Skeleton Army"),
    ("Minions", 5, "Minion Horde"),
    ("Musketeer", 3, "Three Musketeers"),
]


def resolve_composites(events: list[Deploy]) -> list[Deploy]:
    for e in events:
        cards = set(e.card_votes)
        if {"Goblins", "Spear Goblins"} <= cards:
            e.card_votes = {"Goblin Gang": sum(e.card_votes.values())}
        else:
            for single, min_units, composite in COMPOSITES:
                if e.card_votes and max(e.card_votes, key=e.card_votes.get) == single and e.units >= min_units:
                    e.card_votes = {composite: sum(e.card_votes.values())}
        e.card = max(e.card_votes, key=e.card_votes.get) if e.card_votes else None
    return events


POV_WINDOW = (-1.5, 4.0)  # an enemy deploy this soon after a POV play of the same card is the POV unit


def drop_pov_units(events: list[Deploy], pov_plays: list[tuple[float, str]]) -> tuple[list[Deploy], list[Deploy]]:
    """Split enemy deploys into (kept, dropped): dropped ones match a play from our own play log
    (same card, just after it). The detector's faction call is unreliable on freshly placed
    units; the POV log is exact (100% on 490 labelled plays)."""
    kept, dropped = [], []
    for e in events:
        lo, hi = POV_WINDOW
        if any(lo <= e.t - t <= hi and card == e.card for t, card in pov_plays):  # main card only, not stray votes
            dropped.append(e)
        else:
            kept.append(e)
    return kept, dropped
