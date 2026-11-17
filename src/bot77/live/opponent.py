"""The opponent's plays, online, from the unit detector: a tracker that follows units frame to frame (2026-11-16).

`OpponentTracker` (bot77.live.state) counts an enemy unit of a card unseen for 6 s as a new play. Measured on 100 bot
games against their Battle Log replays (scripts/eval_live_tracker.py) it reported 43 plays a game for 20 real ones:
our own units read as theirs once past the river, our Arrows' splash read as Rage / Poison, our Goblin Hut's Spear
Goblins, squads and units lost from sight counted twice. `PlayTracker` replaces "card unseen" with tracks:

- every detection of a card joins the nearest track of that card that could have moved there (any team: the detector
  flips our Mega Minion to 'enemy' past the river), else starts a new track;
- a new track is not a play when something on the field explains it: a recently lost track of that card in reach
  (out of sight a while), a play of that card just counted near it (the rest of a squad), a building or unit that
  spawns it (Goblin Hut -> Spear Goblins, Battle Ram -> Barbarians), or one of OUR plays (a unit of the card we just
  played landing on our half, and the dust of our spells and buildings landing; once it walks over, its track
  carries it);
- what's left counts as a play only where they can deploy it (their half; anywhere for spells and Miner-likes), when
  their cycle allows it (a card played goes to the back of a 4-card queue: it can't come back before 4 other plays),
  and, once their cards are known, only if it is one of them (an opponent has 8 cards).

Also kept, for the brain's inputs: their likely hand from the cycle (`hand()`) and an elixir estimate (`elixir()`).
`replay()` runs it over cached detections (no video): scripts/eval_live_tracker.py --tracker new.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RIVER_Y = 14.9  # bot77.live.state.RIVER_Y (arena tiles, y grows toward our king)


def real_multiplier(elapsed_s: float) -> int:
    """bot77.sim.scenario.real_multiplier (not imported: that module loads the simulator): 2x from 2:00, 3x from 4:00."""
    return 1 if elapsed_s < 120 else 2 if elapsed_s < 240 else 3


# What a card puts on the board besides itself, in the detector's names (bot77.sim.validate.produces, measured in the
# sim 2026-11-16, plus death spawns it misses).
PRODUCES = {
    "Witch": {"Skeletons"}, "Battle Ram": {"Barbarians"}, "Goblin Gang": {"Spear Goblins", "Goblins"},
    "Night Witch": {"Bats"}, "Skeleton Barrel": {"Skeletons"}, "Goblin Giant": {"Spear Goblins"},
    "Goblin Hut": {"Spear Goblins"}, "Barbarian Hut": {"Barbarians"}, "Tombstone": {"Skeletons"},
    "Furnace": {"Fire Spirit"}, "Goblin Drill": {"Goblins"}, "Goblin Barrel": {"Goblins"}, "Graveyard": {"Skeletons"},
    "Barbarian Barrel": {"Barbarians"}, "Royal Delivery": {"Royal Recruits"}, "Skeleton King": {"Skeletons"},
    "Mother Witch": {"Cursed Hog"}, "Elixir Golem": {"Elixir Blob", "Elixir Golemite"}, "Golem": {"Golemite"},
    "Lava Hound": {"Lava Pups"}, "Goblin Cage": {"Goblin Brawler"}, "Suspicious Bush": {"Bush Goblins"},
    "Goblin Demolisher": {"Kamikaze Goblin Demolisher"}, "Phoenix": {"Phoenix Egg"},
}
# the Barbarians a Battle Ram / Barbarian Hut / Barbarian Barrel lets out are often read as Elite Barbarians (2026-11-16:
# our Battle Ram dying at their tower, then "Elite Barbarians" at its spot)
for _c in ("Battle Ram", "Barbarian Hut", "Barbarian Barrel"):
    PRODUCES[_c] = PRODUCES[_c] | {"Elite Barbarians"}
# The detector reads an Arrows volley's splash as Rage / Poison (and the volley just before it as Vines): ours and
# theirs alike (2026-11-16, tune games: every one of our Arrows showed Poison / Rage at -0.5..+0.9 s, at the landing
# spot, team either; the opponents' replays show no Rage or Poison played at all)
SPLASH = {"Rage": "Arrows", "Poison": "Arrows", "Vines": "Arrows"}
# Units whose shots the detector reads as a spell (2026-11-16: "Fireball" on our half next to their Skeleton Dragons,
# Wizard; our Bomber's bombs)
SHOOTERS = {"Fireball": {"Skeleton Dragons", "Wizard", "Baby Dragon", "Bomber", "Electro Dragon", "Firecracker"}}
# cards they can put down on our half too
ANYWHERE = {"Miner", "Goblin Barrel", "Mighty Miner", "Goblin Drill", "Graveyard", "Royal Delivery", "Skeleton Barrel"}


@dataclass
class Track:
    card: str
    x: float
    y: float
    t0: float
    t: float  # last seen
    hits: int = 1
    enemy: int = 0  # reads with team 'enemy'
    owner: str | None = None  # 'opp' | 'own' | None (undecided)
    play: int | None = None  # index into PlayTracker.plays of the play it belongs to
    why: str = ""  # what explained it, when not a play
    conf: float = 0.0
    path: list = field(default_factory=list)
    splash: bool = False  # read as Rage / Poison / Vines (SPLASH)


@dataclass
class Params:
    birth_conf: float = 0.35  # a detection this sure can start a track
    keep_conf: float = 0.3  # ... this sure can continue one
    confirm_hits: int = 2  # reads before a new track is judged
    confirm_s: float = 1.2  # ... within this long, else it's dropped as a flicker
    speed: float = 2.5  # tiles a second a unit can move (gate growth)
    gate: float = 1.5  # tiles, at dt 0
    gate_max: float = 7.0
    coast_s: float = 2.0  # a track unseen this long stops taking detections as "alive" (still explains)
    lost_s: float = 6.0  # a lost track of a card explains a new one in reach for this long
    squad_s: float = 3.0  # a play explains more units of its card this long after it
    squad_r: float = 3.5  # ... this close
    spawn_r: float = 4.0  # a spawner explains its spawns this close
    shot_r: float = 6.0  # a shooter explains a "spell" this close (SHOOTERS)
    own_s: float = 4.0  # our play of a card explains a new unit of it this long after (where it lands, unknown here)
    deploy_margin: float = 2.5  # tiles past the river still "their half"
    cycle_plays: int = 2  # counted plays of other cards before a card can be counted again (true cycle: 4)
    cycle_s: float = 8.0  # ... and seconds (true minimum ~9 s)
    new_card_hits: int = 3  # reads a card not yet among their known cards needs
    new_card_conf: float = 0.6  # a spell not yet among their cards: one read this sure
    new_unit_conf: float = 0.0  # a troop / building not yet among their cards: its surest read at least this
    dup_r: float = 1.0  # tiles: a new track this close to one of its card seen just now is a second box on it
    own_win0: float = -2.0  # our play of the card logged up to this long after a unit of it shows still explains it
    own_r: float = 3.0  # ... within this many tiles of where we put it (when the tap's spot is known; 0: anywhere)
    own_hold_s: float = 1.5  # a new unit of one of our cards near the river waits this long for our play to be logged
    full_deck_block: bool = True  # with 8 cards known, no new card
    ninth_hits: int = 6  # ... unless seen this many reads
    ninth_conf: float = 0.6  # ... once at least this sure
    ninth_spell_conf: float = 0.8  # a spell: one read this sure
    own_half: float = 1.5  # our troops land on our half: a fresh unit this far past the river (their side) isn't ours
    spell_conf: float = 0.4  # a spell is seen once or twice in flight: one read this sure is enough
    spell_hold_s: float = 1.6  # ... judged this long after it shows (our own play may be logged just after it lands)
    own_spell_win: tuple = (-2.0, 2.5)  # our spell explains its own card seen this long around our play
    splash_map: bool = True
    build_dust_r: float = 1.5  # a splash this close to a building in the same read is its landing dust
    el_start: float = 6.0  # their elixir at clock 3:00 (fit on the tune games: the replays read ~7.6 at clock 0:04)
    el_s: float = 2.8  # clock seconds an elixir at 1x (fit on the tune games: 2.9 drifts -0.4 elixir every 30 s)


class PlayTracker:
    def __init__(self, card_type: dict[str, str], cost: dict[str, float], our_deck: list[str],
                 produces: dict[str, set[str]] | None = None, params: Params | None = None):
        self.card_type, self.cost, self.our_deck = card_type, cost, set(our_deck)
        self.produces = produces if produces is not None else PRODUCES
        self.p = params or Params()
        self.tracks: list[Track] = []
        self.dead: list[Track] = []
        self.plays: list[dict] = []
        self.log: list[tuple] = []  # (t0, card, x, y, owner, why) of every judged track, for tuning
        self.own: list[dict] = []  # our plays {t, card, x, y}
        self.t = 0.0
        self.elapsed: float | None = None
        self._el = 5.0  # their elixir estimate
        self._el_t: float | None = None  # elapsed at which _el was last brought up to date

    @classmethod
    def from_cards(cls, cards: list[dict], our_deck: list[str], **kw) -> PlayTracker:
        return cls({c["name"]: c["type"] for c in cards},
                   {c["name"]: float(c["elixir"]) for c in cards if c.get("elixir") is not None}, our_deck, **kw)

    # --- inputs ---------------------------------------------------------------------------------------------------
    def own_play(self, card: str, t: float, x: float | None = None, y: float | None = None) -> None:
        """One of our plays (the bot's tap, or the cycle tracker's play event)."""
        self.own.append({"t": t, "card": card, "x": x, "y": y})

    def update(self, t: float, units, elapsed_s: float | None = None) -> list[dict]:
        """One read of the board. `units`: bot77.live.state.Unit or (card, team, x, y, conf). Returns new plays."""
        self.t = t
        if elapsed_s is not None:
            self._regen(elapsed_s)
        dets, builds = [], None
        for u in units:
            card, team, x, y, conf = (u.card, u.team, u.x, u.y, u.conf) if hasattr(u, "card") else u
            if card and conf >= self.p.keep_conf and team in ("enemy", "friendly"):  # (team "splash": see SPLASH)
                if self.p.splash_map and card in SPLASH:
                    # a building landing raises the same dust (our Goblin Hut: a "Rage" 0.6 tiles below it)
                    builds = builds_of(units, self.card_type) if builds is None else builds
                    if any(math.hypot(x - bx, y - by) <= self.p.build_dust_r for bx, by in builds):
                        continue
                    card = SPLASH[card]
                    team = "splash"
                dets.append((card, team, x, y, conf))
        self._associate(t, dets)
        new = []
        for tr in self.tracks:
            if tr.owner is None and self._ready(tr, t):
                p = self._judge(tr)
                if p is not None:
                    new.append(p)
        # drop flickers that never confirmed, retire tracks out of sight
        keep = []
        for tr in self.tracks:
            if tr.owner is None and t - tr.t0 > max(self.p.confirm_s, self.p.spell_hold_s + 0.1, self.p.own_hold_s + 0.6):
                self.log.append((tr.t0, tr.card, round(tr.x, 1), round(tr.y, 1), None,
                                 "flicker" if tr.hits < self.p.confirm_hits else "new card, few reads"))
                continue
            if t - tr.t > self.p.coast_s:
                if tr.owner is not None:
                    self.dead.append(tr)
                continue
            keep.append(tr)
        self.tracks = keep
        self.dead = [d for d in self.dead if t - d.t <= max(self.p.lost_s, 3.0)]
        return new

    def _spell(self, card: str) -> bool:
        return self.card_type.get(card) == "spell"

    def _ready(self, tr: Track, t: float) -> bool:
        if self._spell(tr.card):
            return tr.conf >= self.p.spell_conf and t - tr.t0 >= self.p.spell_hold_s
        if tr.card in self.our_deck and tr.y >= RIVER_Y - 3.0 and t - tr.t0 < self.p.own_hold_s:
            return False
        return tr.hits >= self.p.confirm_hits

    def update_state(self, st) -> list[dict]:
        """bot77.live.state.LiveState in."""
        return self.update(st.t, st.units, st.elapsed_s)

    # --- tracking -------------------------------------------------------------------------------------------------
    def _gate(self, dt: float) -> float:
        return min(self.p.gate_max, self.p.gate + self.p.speed * max(dt, 0.0))

    def _associate(self, t: float, dets: list) -> None:
        """Each detection to the nearest track of its card that could have moved there (any team), else a new track.
        (Tried 2026-11-16 and dropped: also joining a read of another card right on a track, as the same unit
        misread, and letting a new track take its majority card: F1 +-0.003 on the tune games.)"""
        pairs = []
        for i, (card, _, x, y, _c) in enumerate(dets):
            for j, tr in enumerate(self.tracks):
                if tr.card != card or tr.t >= t:
                    continue
                d = math.hypot(tr.x - x, tr.y - y)
                if d <= self._gate(t - tr.t):
                    pairs.append((d, i, j))
        used_d, used_t = set(), set()
        for d, i, j in sorted(pairs):
            if i in used_d or j in used_t:
                continue
            used_d.add(i); used_t.add(j)
            card, team, x, y, conf = dets[i]
            tr = self.tracks[j]
            tr.x, tr.y, tr.t = x, y, t
            tr.hits += 1
            tr.enemy += team == "enemy"
            tr.conf = max(tr.conf, conf)
            tr.path.append((t, x, y))
        for i, (card, team, x, y, conf) in enumerate(dets):
            if i in used_d or conf < (self.p.spell_conf if self._spell(card) else self.p.birth_conf):
                continue
            self.tracks.append(Track(card, x, y, t, t, 1, int(team == "enemy"), conf=conf, path=[(t, x, y)],
                                     splash=team == "splash"))

    # --- judging a new track --------------------------------------------------------------------------------------
    def _explain(self, tr: Track) -> tuple[str, str] | None:
        """(owner, why) when something already on the field accounts for this track, else None."""
        p, t = self.p, tr.t0
        # out of sight a while: a track of this card lost in reach
        for d in reversed(self.dead):
            if d.card == tr.card and not self._spell(d.card) and 0 <= t - d.t <= p.lost_s and \
                    math.hypot(d.x - tr.x, d.y - tr.y) <= self._gate(t - d.t) + 1.0:
                return d.owner, "lost track"
        for o in self.tracks:
            if o is tr or o.owner is None:
                continue
            dist = math.hypot(o.x - tr.x, o.y - tr.y)
            # the rest of a squad
            if o.card == tr.card and t - o.t0 <= p.squad_s and (dist <= p.squad_r or self._spell(o.card)):
                return o.owner, "squad"
            # a second box on a unit already followed
            if o.card == tr.card and dist <= p.dup_r and t - o.t <= 0.6:
                return o.owner, "second box"
            # a shot, read as a spell
            if o.card in SHOOTERS.get(tr.card, ()) and dist <= p.shot_r and t - o.t <= 1.0:
                return o.owner, f"shot of {o.card}"
            # a spawner on the field
            if tr.card in self.produces.get(o.card, ()) and dist <= p.spawn_r:
                return o.owner, f"spawn of {o.card}"
        # a spawner that just died (Battle Ram -> Barbarians)
        for d in reversed(self.dead):
            if tr.card in self.produces.get(d.card, ()) and 0 <= t - d.t <= 3.0 and \
                    math.hypot(d.x - tr.x, d.y - tr.y) <= p.spawn_r:
                return d.owner, f"spawn of {d.card}"
        # one of our plays
        for o in reversed(self.own):
            dt = t - o["t"]
            if dt < p.own_spell_win[0]:
                continue
            if dt > 32.0:
                break
            if self._spell(tr.card) and p.own_spell_win[0] <= dt <= p.own_spell_win[1] and (
                    o["card"] == tr.card or (tr.splash and self.card_type.get(o["card"]) in ("spell", "building"))):
                # our spell, or the dust of our spell / building landing (our play is logged up to ~1.5 s after)
                return "own", f"our {o['card']}"
            if o["card"] == tr.card and self._spell(tr.card):
                continue
            if o["card"] == tr.card and p.own_win0 <= dt <= p.own_s and tr.y >= RIVER_Y - p.own_half and (
                    o["x"] is None or not p.own_r or
                    math.hypot(o["x"] - tr.x, o["y"] - tr.y) <= p.own_r + p.speed * max(dt, 0.0)):
                return "own", "our play"
            if tr.card in self.produces.get(o["card"], ()) and dt <= 32.0 and tr.card not in self._known():
                return "own", f"spawn of our {o['card']}"
        return None

    def _known(self) -> set[str]:
        return {q["card"] for q in self.plays}

    def _judge(self, tr: Track) -> dict | None:
        out = self._judge1(tr)
        if tr.owner is not None:
            self.log.append((tr.t0, tr.card, round(tr.x, 1), round(tr.y, 1), tr.owner, tr.why or "PLAY"))
        return out

    def _judge1(self, tr: Track) -> dict | None:
        p = self.p
        ex = self._explain(tr)
        if ex is not None:
            tr.owner, tr.why = ex[0] or "opp", ex[1]
            return None
        if tr.enemy * 2 < tr.hits and not self._spell(tr.card):  # read as ours more often than not, and nothing
            # of theirs explains it (a spell's team reads either way)
            tr.owner, tr.why = "own", "friendly"
            return None
        kind = self.card_type.get(tr.card)
        if kind not in ("spell",) and tr.card not in ANYWHERE and tr.y > RIVER_Y + p.deploy_margin:
            tr.owner, tr.why = "opp", "not deployable there"
            return None
        known = self._known()
        if tr.card not in known:
            sure = tr.conf >= p.ninth_spell_conf if self._spell(tr.card) else (
                tr.hits >= p.ninth_hits and tr.conf >= p.ninth_conf)
            if p.full_deck_block and len(known) >= 8 and not sure:
                # their 8 are known: a 9th card needs a sure, steady sighting (one of the 8 was a misread)
                if tr.hits >= p.ninth_hits or self._spell(tr.card):
                    tr.owner, tr.why = "opp", "not in their 8"
                return None
            if self._spell(tr.card):
                if tr.conf < p.new_card_conf:
                    return None
            elif tr.hits < p.new_card_hits or tr.conf < p.new_unit_conf:
                return None  # wait for more / surer reads (dropped after confirm_s)
        last = next((k for k in range(len(self.plays) - 1, -1, -1) if self.plays[k]["card"] == tr.card), None)
        if last is not None and (len(self.plays) - 1 - last < p.cycle_plays or tr.t0 - self.plays[last]["t"] < p.cycle_s):
            tr.owner, tr.why = "opp", "cycle"
            return None
        tr.owner = "opp"
        # t: when it was first seen (the play's time, as near as the board shows it); t_known: when it was counted
        play = {"t": tr.t0, "t_known": self.t, "card": tr.card, "x": tr.x, "y": tr.y, "at": [(round(tr.x, 2), round(tr.y, 2))],
                "elapsed_s": self.elapsed}
        self._spend(tr.card)
        tr.play = len(self.plays)
        self.plays.append(play)
        return play

    # --- their cycle and elixir -----------------------------------------------------------------------------------
    def deck(self) -> list[str]:
        """Their cards seen so far (at most 8: the most played when more were counted)."""
        n: dict[str, int] = {}
        for q in self.plays:
            n[q["card"]] = n.get(q["card"], 0) + 1
        first = {q["card"]: k for k, q in reversed(list(enumerate(self.plays)))}
        return sorted(n, key=lambda c: (-n[c], first[c]))[:8]

    def hand(self) -> tuple[list[str | None], bool]:
        """(their likely 4 cards, None where unknown; True when all 8 of their cards are known). A card played goes to
        the back of their queue and needs 4 more plays to come back, so their hand is their cards outside the last 4
        played; longest-unplayed first when there are more candidates than slots."""
        deck = self.deck()
        recent = []
        for q in reversed(self.plays):
            if q["card"] in deck and q["card"] not in recent:
                recent.append(q["card"])
            if len(recent) == 4:
                break
        cand = [c for c in deck if c not in recent]
        last = {q["card"]: q["t"] for q in self.plays}
        cand.sort(key=lambda c: last.get(c, -1e9))
        out = cand[:4]
        return out + [None] * (4 - len(out)), len(deck) == 8

    def _regen(self, elapsed: float) -> None:
        if self._el_t is None:
            self._el_t = elapsed
            self._el = min(10.0, self.p.el_start + elapsed / self.p.el_s)
        e, t = self._el, self._el_t
        while t < elapsed:
            step = min(0.5, elapsed - t)
            e = min(10.0, e + step * real_multiplier(t) / self.p.el_s)
            t += step
        self._el, self._el_t = e, max(t, self._el_t)
        self.elapsed = elapsed

    def _spend(self, card: str) -> None:
        self._el = max(0.0, self._el - self.cost.get(card, 3.5))

    def elixir(self) -> float:
        """Their elixir now: 5 at the start, the real regeneration (x2 from 2:00, x3 from 4:00), capped at 10, minus
        each counted play's cost (never below 0)."""
        return self._el


def builds_of(units, card_type: dict) -> list[tuple[float, float]]:
    out = []
    for u in units:
        card, x, y = (u.card, u.x, u.y) if hasattr(u, "card") else (u[0], u[2], u[3])
        if card and card_type.get(card) == "building":
            out.append((x, y))
    return out


def replay(trk: PlayTracker, units: list, clock: list, own_plays: list[dict], hand_every: float = 0.5):
    """Run a tracker over cached detections (scripts/eval_live_tracker.py's per-game cache): `units` [t, [[card, team,
    x, y, conf], ...]], `clock` [(t, elapsed_s)] about once a second, `own_plays` [{t, card}]. Returns (plays,
    [(t, hand, all 8 known, elixir)] every hand_every s)."""
    own = sorted(own_plays, key=lambda o: o["t"])
    k, ci, hand, last_h = 0, 0, [], -1e9
    for t, us in units:
        while k < len(own) and own[k]["t"] <= t:
            trk.own_play(own[k]["card"], own[k]["t"], own[k].get("x"), own[k].get("y"))
            k += 1
        while ci + 1 < len(clock) and clock[ci + 1][0] <= t:
            ci += 1
        el = None
        if clock and clock[ci][0] <= t and clock[ci][1] is not None:
            el = clock[ci][1] + (t - clock[ci][0])
        trk.update(t, us, el)
        if t - last_h >= hand_every:
            h, sure = trk.hand()
            hand.append((t, h, sure, round(trk.elixir(), 2)))
            last_h = t
    return trk.plays, hand
