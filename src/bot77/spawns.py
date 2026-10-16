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

import os
MIN_CONF = float(os.environ.get("BOT77_MIN_CONF", "0.5"))  # dev-set sweep override
TRACK_RADIUS_TILES = 1.6  # a detection continues a track within this distance ...
TRACK_SPEED_TILES_S = 2.5  # ... plus this much per second since the track was last seen
TRACK_MAX_GAP_S = 1.5  # tracks unseen for longer end
RESUME_S, RESUME_TILES = 6.0, 5.0  # a birth this close to a same-card track that just ended resumes it
# (units jump - Firecracker's recoil, Bandit's dash - and detections drop out; replaying the
# same card within 6 s is rare even in fast cycle decks)
BUILDING_RESUME_S = 90.0  # Tesla hides underground, so it resumes from much longer ago ...
LONG_RESUME_CARDS = {"Tesla"}  # ... other buildings don't: a new Tombstone often goes on the same spot (dev set, 2026-10-10)
VOTE_TRACKS = os.environ.get("BOT77_VOTE_TRACKS", "1") == "1"  # dev A/B switch
# a detection may continue a track of another card (class flicker on one unit). Off: back-to-back plays of two cards
# (Bandit + Wall Breakers) stop vanishing into one track; flicker becomes a short extra track in the same group.
CROSS_CARD = os.environ.get("BOT77_CROSS_CARD", "0") == "1"  # off by default since 2026-10-15
VOTE_WINDOW_S = float(os.environ.get("BOT77_VOTE_WINDOW", "1.0"))  # only a track's first seconds vote
CONFIRM_S, CONFIRM_DETS = 2.0, 2  # a birth needs this many detections within this long (1.0 s lost slow-to-detect
# new cards on the dev set, 2026-10-10)
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
    votes: Counter = field(default_factory=Counter)  # (card, form, cls) -> summed confidence over the track

    def vote(self, card: str, form: str, cls: str, conf: float) -> None:
        """The track's card is its confidence-weighted majority, not its first detection (dev set, 2026-10-11:
        a Void seen 10 times was called Clone because its first frame read "clone")."""
        if self.dets and self.dets[-1][0] - self.first_t > VOTE_WINDOW_S:
            return  # later on the track may have jumped to another unit
        self.votes[(card, form, cls)] += conf
        if VOTE_TRACKS:
            by_card = Counter()
            for (c, _, _), v in self.votes.items():
                by_card[c] += v
            self.card = by_card.most_common(1)[0][0]
            self.form, self.cls = max(((f, k) for (c, f, k) in self.votes if c == self.card),
                                      key=lambda fk: self.votes[(self.card, *fk)])


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
    twin: bool = False  # another unit of the same card was alive elsewhere at this birth: a second copy, not the first


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
                if k in claimed or tr.team != d["team"] or (not CROSS_CARD and tr.card != card):
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
                tr.vote(card, form, d["cls"], d["conf"])
                continue
            tr = Track(d["team"], card, form, d["cls"], t, t, x, y, [(t, x, y)], boxes=[(d["x0"], d["y0"], d["x1"], d["y1"])])
            tr.vote(card, form, d["cls"], d["conf"])
            tr.resumed = any(
                o.team == tr.team and o.card == card and
                ((t - o.last_t <= RESUME_S and ((o.x - x) ** 2 + (o.y - y) ** 2) ** 0.5 <= RESUME_TILES) or
                 (card in LONG_RESUME_CARDS and t - o.last_t <= BUILDING_RESUME_S and ((o.x - x) ** 2 + (o.y - y) ** 2) ** 0.5 <= 1.5))
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
            origin: tuple[float, float] = (0.0, 0.0), spell_team: str | None = None, split_min_dets: int = 0) -> list[Deploy]:
    """`origin`: the detector crop's top-left in frame pixels, so positions are arena tiles.
    `spell_team`: give every spell detection this team. Spells carry no badge, so their team is guessed from the
    screen half, but an opponent's Void/Curse/Vines lands on OUR half (dev set, 2026-10-11: 16 of 20 detected Voids
    were called friendly). The opponent pipeline sets "enemy" and removes our own spells with the POV play log."""
    ox, oy = origin
    dets = [{**d, "x0": d["x0"] - ox, "x1": d["x1"] - ox, "y0": d["y0"] - oy, "y1": d["y1"] - oy} for d in dets]
    if spell_team:
        dets = [{**d, "team": spell_team} if is_spell(class_to_card(d["cls"], card_names)[0]) else d for d in dets]
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
    if split_min_dets:
        groups = [h for g in groups for h in _split(g, split_min_dets)]
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
        card = votes.most_common(1)[0][0]
        out.append(Deploy(g[0].team, t, x, y, card, dict(votes), len(g), evo,
                          all(tr.cls in DEATH_OR_SUMMON for tr in g), sorted({tr.cls for tr in g}),
                          twin=_twin(g, card, tracks)))
    return out


SPLIT_APART_TILES = 0.8


def _split(g: list[Track], min_dets: int) -> list[list[Track]]:
    """Two cards played together (Bandit + Wall Breakers 0.3 s apart) land in one group. Split it per card only on
    evidence (2026-10-15): each card has a unit seen at least `min_dets` times, and the second card's units start
    apart from the first card's (two boxes on one unit sit on top of each other)."""
    by = {}
    for tr in g:
        by.setdefault(tr.card, []).append(tr)
    strong = [c for c, trs in by.items() if max(len(tr.dets) for tr in trs) >= min_dets]
    if len(strong) < 2:
        return [g]
    main = max(strong, key=lambda c: sum(len(tr.dets) for tr in by[c]))
    out = [list(by[main])]
    rest = [tr for c, trs in by.items() if c not in strong for tr in trs]
    for c in strong:
        if c == main:
            continue
        apart = all(((a.dets[0][1] - b.dets[0][1]) ** 2 + (a.dets[0][2] - b.dets[0][2]) ** 2) ** 0.5 > SPLIT_APART_TILES
                    for a in by[c] for b in by[main])
        (out.append(list(by[c])) if apart else out[0].extend(by[c]))
    out[0].extend(rest)
    return [sorted(h, key=lambda tr: tr.first_t) for h in out]


TWIN_TILES, TWIN_ALIVE_S, TWIN_MIN_DETS = 3.0, 0.5, 4


def _twin(g: list[Track], card: str, tracks: list[Track]) -> bool:
    """Two units of this card on the field at once, apart: an older track of the same card and team, seen both before
    and after this birth, more than TWIN_TILES away at the birth, and the new unit seen at least TWIN_MIN_DETS times.
    Then the birth is a new copy (a Mirror, or the card played again), not the old unit walking back (2026-10-15)."""
    t, x, y = g[0].first_t, g[0].dets[0][1], g[0].dets[0][2]
    if max(len(tr.dets) for tr in g) < TWIN_MIN_DETS:
        return False
    for o in tracks:
        if o in g or o.team != g[0].team or o.card != card:
            continue
        if not (o.first_t < t - TWIN_ALIVE_S and o.last_t > t + TWIN_ALIVE_S):
            continue
        ot, ox, oy = min(o.dets, key=lambda d: abs(d[0] - t))
        if abs(ot - t) <= 1.0 and ((ox - x) ** 2 + (oy - y) ** 2) ** 0.5 > TWIN_TILES:
            return True
    return False


def apply_deck(events: list[Deploy], deck: set[str]) -> list[Deploy]:
    """Re-pick each deploy's card among the deck (detector classes outside it are ignored);
    deploys with no deck card among their votes keep card=None."""
    for e in events:
        in_deck = {c: v for c, v in e.card_votes.items() if c in deck}
        e.card = max(in_deck, key=in_deck.get) if in_deck else None
    return events


def infer_deck(events: list[Deploy], size: int = 8, w_unclocked: float = 1.0, is_spell=lambda c: False) -> list[str]:
    """The opponent's deck: cards that won the most deploys. A deploy counts once, or `w_unclocked` when it has no
    deploy clock and is not a spell (clocks need bot77.clocks.fuse_clocks; 1.0 = every deploy counts once)."""
    wins = Counter()
    for e in events:
        if e.card and not e.subunit_only:
            wins[e.card] += 1.0 if (getattr(e, "clock_t", None) is not None or is_spell(e.card)) else w_unclocked
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
