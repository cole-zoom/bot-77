"""Spells in flight, online, from the unit detector (2026-11-16, Cole: "spells in flight (Fireball / Rocket / Log /
arrows volleys travelling: type, position, landing spot, time to impact)").

The live detector has a class for each spell (fireball, rocket, the-log, arrows, zap, goblin-barrel, ...;
bot77.detect.names maps them to the card). What it actually sees, on 100 bot games (scripts/eval_spell_slots.py):
a Fireball in flight once or twice (its sprite between their King and the target, then the blast), Arrows almost
never (their volley's splash reads as Rage / Poison / Vines: bot77.live.opponent.SPLASH), Rocket / Log rarely.

Per spell sighting:
- a track: sightings of one card joined while they could be one projectile (FLIGHT speed + slack);
- whose: its motion when it has moved (down the screen = theirs, thrown from their King toward us; up = ours), else
  one of our plays of that card within OWN_WIN of its first sighting = ours, else theirs;
- where it lands: thrown spells fly from the caster's King in a straight line at a constant speed (game files). The
  landing spot isn't on screen until it lands, so it is guessed: of the target side's units and towers ahead of it
  on its line (within the spell's radius of the line), the spot that catches the most; none, a few tiles on;
- when: (distance left to the landing spot) / speed, minus the time since it was last seen. Rolling spells (Log,
  Barbarian Barrel) and a splash read (Arrows landing) are "landing now" (time to impact 0).

Positions are in our arena frame (x 0-18, y down the screen, river 14.9; bot77.live.state); distances and speeds in
engine tiles (bot77.sim.geometry: our y is foreshortened by Y_SCALE). The sprite in flight sits above the ground (its
arc): its y reads up to ~1 tile early. Speeds (engine tiles a second) are the game files' (speed / 60) where we know
them; Fireball's is measured on the bot games' replays (8 pairs of sightings, median 13.5 with the arc's height in
it; the files say 600 = 10): 12. Treat the rest as approximate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from bot77.live.opponent import SHOOTERS, SPLASH, builds_of

RIVER_Y = 14.9
# units whose shots read as a spell (bot77.live.opponent.SHOOTERS, plus the Mega Minion: 2026-11-16, single
# "Fireball" sightings next to one, ours or theirs, with no Fireball played)
SHOT_LIKE = {k: set(v) | ({"Mega Minion"} if k == "Fireball" else set()) for k, v in SHOOTERS.items()}
Y_SCALE = 0.7305  # bot77.sim.geometry.Y_SCALE (not imported: that module loads the simulator)

# engine tiles a second; rollers and instant spells: None (they're "landing now" when seen)
FLIGHT = {"Fireball": 12.0, "Rocket": 6.0, "Arrows": 18.0, "Goblin Barrel": 7.0, "Giant Snowball": 13.0,
          "Royal Delivery": 8.0, "Skeleton Barrel": 5.0}
ROLLERS = {"The Log", "Barbarian Barrel"}
# engine tiles (bot77 fork CardRegistry spell radii; Goblin Barrel: its Goblins' spread)
RADIUS = {"Fireball": 2.5, "Rocket": 2.0, "Arrows": 3.5, "Zap": 2.5, "The Log": 2.0, "Goblin Barrel": 1.5,
          "Barbarian Barrel": 1.3, "Giant Snowball": 2.5, "Lightning": 3.5, "Poison": 3.5, "Earthquake": 3.5,
          "Freeze": 3.0, "Tornado": 5.5}
# the slot's type one-hot, in this order (anything else: "other")
SPELL_TYPES = ["Fireball", "Rocket", "Arrows", "The Log", "Zap", "Goblin Barrel", "Barbarian Barrel",
               "Giant Snowball", "Lightning", "Poison", "other"]
# the Kings' feet in our frame (engine (8.5, 30.5) / (8.5, 2.5) through bot77.sim.geometry.engine_to_ours)
THEIR_KING, OUR_KING = (9.0, 4.67), (9.0, 25.13)
# princess towers' feet in our frame (box centres 5.94 / 21.28 + 1.29; bot77.sim.geometry)
THEIR_TOWERS = [(3.45, 7.23), (14.5, 7.23), THEIR_KING]
OUR_TOWERS = [(3.45, 22.57), (14.5, 22.57), OUR_KING]
FEET_DY = 1.27  # bot77.features.frames.FEET_DY
# where a thrown spell leaves from: the King's sprite (its box centre), not its feet: a Fireball is first seen level
# with the King's feet, so the line from the feet ran sideways (2026-11-16, eval_spell_slots landing check)
THEIR_LAUNCH, OUR_LAUNCH = (THEIR_KING[0], THEIR_KING[1] - FEET_DY), (OUR_KING[0], OUR_KING[1] - FEET_DY)


def edist(a, b) -> float:
    """Engine-tile distance between two points of our frame."""
    return math.hypot(a[0] - b[0], (a[1] - b[1]) / Y_SCALE)


@dataclass
class SpellTrack:
    card: str
    t0: float
    t: float
    x: float
    y: float
    conf: float
    pts: list = field(default_factory=list)  # (t, x, y)
    splash: bool = False
    owner: str = "opp"  # 'opp' | 'own'
    why: str = ""
    land: tuple[float, float] | None = None
    tti: float = 0.0  # seconds to impact, as of the last update
    counted: bool = False  # their play of it already counted by the PlayTracker


@dataclass
class SpellParams:
    conf: float = 0.35  # a sighting this sure starts a track
    join_s: float = 1.0  # a sighting this long after a track's last joins it if in reach
    slack: float = 3.0  # engine tiles of reach beyond its speed (arc, box jitter)
    own_win: tuple = (-2.0, 3.5)  # our play of the card this long around its first sighting: ours (our Fireball
    # is first seen up to ~3 s after the tap, landing on their side)
    move_min: float = 1.0  # engine tiles moved before its direction counts
    linger_s: float = 0.5  # a landed spell (time to impact 0) stays this long after its last sighting
    ahead_min: float = -1.0  # engine tiles: a target this far behind the spell (along its line) can still be hit
    line_slack: float = 1.0  # engine tiles beyond the radius (the line from the King is rough: the sprite arcs)
    cone: float = 0.25  # extra engine tiles of reach per tile along the line, when its direction is from the King
    default_ahead: float = 5.0  # no target on its line: lands this many engine tiles on
    shot_r: float = 2.5  # our tiles: a spell read this close to a unit that shoots like it is its shot
    build_dust_r: float = 1.5  # a splash read this close to a building is its landing dust (PlayTracker.Params)


class SpellTracker:
    def __init__(self, card_type: dict[str, str], our_deck: list[str] | set[str] = (), params: SpellParams | None = None):
        self.card_type = card_type
        self.our_deck = set(our_deck)
        self.p = params or SpellParams()
        self.tracks: list[SpellTrack] = []
        self.done: list[SpellTrack] = []  # every track that ended (for scoring)
        self.t = 0.0

    def _spell(self, card: str | None) -> bool:
        return card is not None and (self.card_type.get(card) == "spell" or card in SPLASH)

    def update(self, t: float, units, own_plays: list[dict] = (), their_cards: set[str] | None = None) -> None:
        """One read of the board. `units`: bot77.live.state.Unit or (card, team, x, y, conf); `own_plays`: our plays
        so far, {t, card} (the PlayTracker's .own); `their_cards`: their 8 once all are known (a spell in neither
        deck is then a misread)."""
        self.t = t
        dets, board, builds, shooters_at = [], [], None, []
        for u in units:
            card, team, x, y, conf = (u.card, u.team, u.x, u.y, u.conf) if hasattr(u, "card") else u
            if not card:
                continue
            if self._spell(card):
                if conf >= self.p.conf:
                    splash = card in SPLASH
                    if their_cards and SPLASH.get(card, card) not in their_cards | self.our_deck:
                        continue
                    if splash:  # a building landing raises the same dust (PlayTracker's rule)
                        builds = builds_of(units, self.card_type) if builds is None else builds
                        if any(math.hypot(x - bx, y - by) <= self.p.build_dust_r for bx, by in builds):
                            continue
                    dets.append((SPLASH.get(card, card), x, y, conf, splash))
            elif team in ("friendly", "enemy") and conf >= 0.3:
                board.append((team, x, y + FEET_DY))
                shooters_at.append((card, x, y))
        used = set()
        for k, (card, x, y, conf, splash) in enumerate(dets):
            best = None
            for tr in self.tracks:
                dt = t - tr.t
                if tr.card != card or tr.t >= t or dt > self.p.join_s:
                    continue
                reach = (FLIGHT.get(card) or 0.0) * dt + self.p.slack
                d = edist((tr.x, tr.y), (x, y))
                if d <= reach and (best is None or d < best[0]):
                    best = (d, tr)
            if best is not None and id(best[1]) not in used:
                tr = best[1]
                used.add(id(tr))
                tr.x, tr.y, tr.t, tr.conf = x, y, t, max(tr.conf, conf)
                tr.splash = tr.splash or splash
                tr.pts.append((t, x, y))
            else:
                # a shot read as a spell (their Skeleton Dragons' / Wizard's, our Bomber's: SHOOTERS) right by its
                # shooter starts nothing
                shooters = SHOT_LIKE.get(card, ())
                if any(c in shooters and math.hypot(x - ux, y - uy) <= self.p.shot_r for c, ux, uy in shooters_at):
                    continue
                tr = SpellTrack(card, t, t, x, y, conf, [(t, x, y)], splash)
                self.tracks.append(tr)
                used.add(id(tr))
        keep = []
        for tr in self.tracks:
            self._judge(tr, own_plays)
            self._aim(tr, board, t)
            if tr.tti <= 0 and t - tr.t > self.p.linger_s or t - tr.t > 3.0:
                self.done.append(tr)
                continue
            keep.append(tr)
        self.tracks = keep

    def _judge(self, tr: SpellTrack, own_plays) -> None:
        (t0, x0, y0), (_, x1, y1) = tr.pts[0], tr.pts[-1]
        if not tr.splash and tr.card not in ROLLERS and edist((x0, y0), (x1, y1)) >= self.p.move_min:
            tr.owner, tr.why = ("opp", "flies down") if y1 > y0 else ("own", "flies up")
            return
        lo, hi = self.p.own_win
        for o in own_plays:
            if lo <= tr.t0 - o["t"] <= hi:  # first seen up to 2 s before our play is logged, 2.5 s after
                if o["card"] == tr.card or (tr.splash and self.card_type.get(o["card"]) in ("spell", "building")):
                    tr.owner, tr.why = "own", f"our {o['card']}"
                    return
        tr.owner, tr.why = "opp", "no play of ours"

    def _aim(self, tr: SpellTrack, board, t: float) -> None:
        """Landing spot and time to impact."""
        speed = FLIGHT.get(tr.card)
        if speed is None or tr.splash:
            tr.land, tr.tti = (tr.x, tr.y), 0.0
            return
        origin = THEIR_LAUNCH if tr.owner == "opp" else OUR_LAUNCH
        targets = [(x, y) for team, x, y in board if team == ("friendly" if tr.owner == "opp" else "enemy")]
        towers = OUR_TOWERS if tr.owner == "opp" else THEIR_TOWERS
        (_, xa, ya) = tr.pts[0]
        p = (tr.x, tr.y)
        moved = edist((xa, ya), p) >= self.p.move_min
        a = (xa, ya) if moved else origin
        cone = 0.0 if moved else self.p.cone  # a line from the King only: rougher the further out
        # the line in engine-scaled coordinates
        ux, uy = p[0] - a[0], (p[1] - a[1]) / Y_SCALE
        n = math.hypot(ux, uy)
        if n < 1e-6:
            ux, uy = (0.0, 1.0) if tr.owner == "opp" else (0.0, -1.0)
        else:
            ux, uy = ux / n, uy / n
        r = RADIUS.get(tr.card, 2.5)
        cands = []
        for c in targets + towers:
            vx, vy = c[0] - p[0], (c[1] - p[1]) / Y_SCALE
            s = vx * ux + vy * uy
            perp = abs(vx * uy - vy * ux)
            if s >= self.p.ahead_min and perp <= r + self.p.line_slack + cone * max(s, 0.0):
                caught = sum(edist(c, o) <= r for o in targets) + 1.5 * sum(edist(c, o) <= r for o in towers)
                cands.append((-caught, perp, s, c))
        if cands:
            land = min(cands)[3]
        else:  # nothing on its line: on toward the target side's princess towers
            row = towers[0][1]
            k = (row - p[1]) / (uy * Y_SCALE) if abs(uy) > 0.2 and (row - p[1]) * uy > 0 else self.p.default_ahead
            land = (p[0] + ux * k, p[1] + uy * k * Y_SCALE)
        land = (min(18.0, max(0.0, land[0])), min(OUR_KING[1] + 1.0, max(THEIR_KING[1] - 1.0, land[1])))
        tr.land = land
        tr.tti = max(0.0, edist(p, land) / speed - (t - tr.t))

    # --- the brain's view ---------------------------------------------------------------------------------------------
    def slots(self, k: int = 3) -> list[SpellTrack]:
        """The spells in flight now, soonest impact first (theirs before ours on a tie), at most k."""
        return sorted(self.tracks, key=lambda tr: (tr.tti, tr.owner != "opp", -tr.conf))[:k]
