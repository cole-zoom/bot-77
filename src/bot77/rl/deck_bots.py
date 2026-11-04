"""Scripted opponents that play a standard deck (docs/decks/*.md; Cole 2026-11-04: "the general rules of how to play
each deck to counter the bots bad plays").

Each deck is a DeckBot subclass: the engine's team 1, with its own 8 cards, a legal hand and cycle, paid elixir. Every
tick it is asked for a move (tick()). It reads only what a player could: the board, the towers, its own hand and
elixir, the bot's elixir (counted, as good players do) and the bot's plays as they land (note_play, called by whoever
steps the game). It never sees the bot's hand; "out of the bot's hand" means among the bot's last 4 plays.

A rule that fires queues a plan: one or more plays, the first after a seeded 0.5-1.0 s reaction delay. A play waits
for its card, elixir and the engine's 2 s cooldown on a just-drawn card (up to WAIT_S, then it's dropped); a defence preempts a queued attack. Every fired rule is an
event (rule id + what it does) for the play page and the logs, so Cole can see which rule made each play.

Coordinates are the engine board's (docs/decks/README.md "The board"): the bot is team 0 and defends low y, this
opponent high y. Team 1's actions take its own frame (y mirrored, 33 - y; x unchanged); plan steps are in the board
frame and converted here. A target that isn't a legal tile moves to the nearest legal one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bot77.sim import cards as sc

TICK_S = 0.1
LANE_X = (3.0, 14.0)  # princess towers
BRIDGE_X = (2.5, 14.5)
MID_X = 8.5
RIVER_LO, RIVER_HI = 15.5, 17.5
MY_BRIDGE_Y = 18.5  # this opponent's bridge spot
MY_BACK_Y = 32.5
BOT_TOWER_Y = 6.0
DOUBLE_TICK, TRIPLE_TICK = 1200, 2400
REACT_S = (0.5, 1.0)
WAIT_S = 4.0  # a queued play that can't be made in this long is dropped
# the cards of the six decks that can hit a flying unit (the fallback never sends a ground-only card at one)
HITS_AIR = {"Musketeer", "Ice Spirit", "Fireball", "Arrows", "Zap", "Electro Wizard", "Magic Archer", "Baby Dragon",
            "Minions", "Firecracker", "Dart Goblin", "Inferno Tower", "Princess", "Electro Spirit", "Royal Delivery",
            "Rocket", "Poison", "Goblin Gang", "Vines", "Mega Minion", "Tesla", "Wizard", "Archers", "Bats"}
SPELL_NAMES = {c["name"] for c in sc.engine_cards().values() if c["is_spell"]}
OUR_DECK = ["Giant", "Battle Ram", "Mini PEKKA", "Mega Minion", "Bomber", "Goblin Hut", "Fireball", "Arrows"]


def card_id(name: str) -> int:
    for i, c in sc.engine_cards().items():
        if c["name"] == name and not c["evolution"] and not c["is_hero"]:
            return i
    i = sc.engine_id(name)
    if i is None:
        raise KeyError(name)
    return i


def lane_of(x: float) -> int:
    return 0 if x < MID_X else 1


@dataclass
class Seen:
    """A play of the bot's, as the opponent saw it land."""
    t: float
    name: str
    cost: float
    x: float
    y: float

    @property
    def lane(self) -> int:
        return lane_of(self.x)


@dataclass
class Step:
    card: str
    at: object  # (x, y) on the board, or a function of the bot -> (x, y) | None, worked out when it's played
    after_s: float = 0.0  # after the previous step of the plan
    due: float = 0.0
    expires: float = 0.0


@dataclass
class Plan:
    rule: str
    text: str
    steps: list = field(default_factory=list)
    urgent: bool = False


class DeckBot:
    NAME = ""
    DECK: list[str] = []
    OPEN_WAIT_S = 0.0  # heavy decks wait this long for the bot to play first (Cole, 2026-11-04)

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.ids = {n: card_id(n) for n in self.DECK}
        self.names = {i: n for n, i in self.ids.items()}
        self.info = {i: sc.engine_cards()[i] for i in self.ids.values()}
        self.our = {card_id(n): n for n in OUR_DECK}
        self.cost = {n: float(sc.engine_cards()[card_id(n)]["cost"]) for n in self.DECK + OUR_DECK}
        self.g = None
        self.seen: list[Seen] = []
        self.plans: list[Plan] = []
        self.events: list[dict] = []  # drained by the caller (the page shows them)
        self.fired: dict[str, int] = {}
        self.answered_at: dict[int, float] = {}  # bot unit id -> when a defence was sent for it
        self.mine: list[tuple] = []  # (t, card, x, y): this opponent's own plays
        self.st: dict = {}
        self._sent = None  # (plan, step, elixir spent before, spot): the play sent last tick

    # ------------------------------------------------------------------ the game
    def bind(self, game) -> "DeckBot":
        self.g = game
        return self

    @property
    def deck_ids(self) -> list[int]:
        return [self.ids[n] for n in self.DECK]

    @property
    def t(self) -> float:
        return self.g.get_current_tick() * TICK_S

    @property
    def phase(self) -> str:
        k = self.g.get_current_tick()
        return "single" if k < DOUBLE_TICK else "double" if k < TRIPLE_TICK else "triple"

    def hand(self) -> list[str]:
        return [self.names.get(c, str(c)) for c in self.g.get_hand_for_team(1)]

    @property
    def elixir(self) -> float:
        return float(self.g.get_elixir_for_team(1))

    @property
    def bot_elixir(self) -> float:
        return float(self.g.get_elixir_for_team(0))

    def has(self, *cards: str) -> bool:
        """All of these in hand (affordability is the plan's business: a step waits for elixir)."""
        h = self.hand()
        return all(c in h for c in cards)

    def can(self, card: str) -> bool:
        return card in self.hand() and self.elixir >= self.cost[card]

    # ------------------------------------------------------------------ what it sees of the bot
    def note_play(self, card: int, x: float, y: float):
        """The bot's play landed (the caller knows it: the page's click, the env's action)."""
        name = self.our.get(card, sc.engine_cards().get(card, {}).get("name", str(card)))
        s = Seen(self.t, name, float(sc.engine_cards()[card]["cost"]) if card in sc.engine_cards() else 0.0, x, y)
        self.seen.append(s)
        self.on_bot_play(s)

    def on_bot_play(self, s: Seen):
        """A hook: a rule that fires on a play (e.g. a 4+ card in the back)."""

    def out_of_bot_hand(self, name: str) -> bool:
        """The bot played it in its last 4 plays: it can't be in its hand."""
        return any(s.name == name for s in self.seen[-4:])

    def maybe_in_bot_hand(self, name: str) -> bool:
        return not self.out_of_bot_hand(name)

    def bot_units(self, flying: bool | None = None) -> list:
        out = [u for u in self.g.get_units() if u.team == 0 and not u.tower and u.hp > 0]
        return out if flying is None else [u for u in out if bool(u.flying) == flying]

    def bot_troops(self) -> list:
        """Bot units that walk or fly: not buildings, and not a spell's effect on the board (an Arrows, a Fireball)."""
        ec = sc.engine_cards()
        return [u for u in self.bot_units() if not u.building and u.name not in SPELL_NAMES
                and not (u.card_id in ec and ec[u.card_id]["is_spell"] and u.name == ec[u.card_id]["name"])]

    def bot_buildings(self, name: str | None = None) -> list:
        return [u for u in self.bot_units() if u.building and (name is None or u.name == name)]

    def crossed(self) -> list:
        """Bot troops on this opponent's side of the river (or in it)."""
        return [u for u in self.bot_troops() if u.y >= RIVER_LO]

    def committed(self, lane: int, window_s: float = 6.0) -> float:
        """Elixir the bot put into one lane in the last window_s (spells count where they landed)."""
        return sum(s.cost for s in self.seen if s.t >= self.t - window_s and s.lane == lane)

    def last_bot_play(self) -> Seen | None:
        return self.seen[-1] if self.seen else None

    def my_units(self, name: str | None = None) -> list:
        return [u for u in self.g.get_units() if u.team == 1 and not u.tower and u.hp > 0
                and (name is None or u.name == name)]

    def tower_hp(self, team: int, lane: int) -> float:
        return float(max(0, self.g.get_tower_hp(team, 1 + lane)))

    def weaker_bot_lane(self) -> int:
        hp = [self.tower_hp(0, k) for k in (0, 1)]
        if hp[0] <= 0 < hp[1] or hp[1] <= 0 < hp[0]:  # a fallen tower: the king's side of it
            return 0 if hp[0] <= 0 else 1
        return int(np.argmin(hp)) if hp[0] != hp[1] else int(self.rng.integers(2))

    def crowns(self) -> tuple[int, int]:
        """(the bot's, this opponent's): princess towers each has taken."""
        return (sum(self.tower_hp(1, k) <= 0 for k in (0, 1)), sum(self.tower_hp(0, k) <= 0 for k in (0, 1)))

    def leading(self) -> bool:
        b, m = self.crowns()
        if m != b:
            return m > b
        return sum(self.tower_hp(0, k) for k in (0, 1)) < sum(self.tower_hp(1, k) for k in (0, 1)) - 100

    # ------------------------------------------------------------------ acting
    def queue(self, rule: str, text: str, *steps: Step, urgent: bool = False) -> bool:
        if not steps:
            return False
        delay = float(self.rng.uniform(*REACT_S))
        due = self.t + delay
        for s in steps:
            due += s.after_s
            s.due, s.expires = due, due + WAIT_S
        p = Plan(rule, text, list(steps), urgent)
        if urgent:  # a defence goes first; a queued attack waits behind it
            self.plans.insert(0, p)
        else:
            self.plans.append(p)
        self.fired[rule] = self.fired.get(rule, 0) + 1
        self.events.append({"t": round(self.t, 1), "rule": rule, "text": text, "react_s": round(delay, 2)})
        return True

    def busy(self, urgent_only: bool = False) -> bool:
        return any(p.urgent or not urgent_only for p in self.plans)

    def legal_spot(self, card: str, x: float, y: float) -> tuple[float, float] | None:
        cid = self.ids[card]
        for r in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            pts = [(x, y)] if r == 0 else [(x + dx * r, y + dy * r) for dx, dy in
                                            ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))]
            for px, py in pts:
                px, py = float(np.clip(px, 0, 17)), float(np.clip(py, 0, 33))
                if self.g.is_valid_placement(cid, px, py, 1):
                    return px, py
        return None

    def tick(self):
        """This tick's move for team 1: (hand index, x, y) in its own frame, or None to wait."""
        if not self.busy(urgent_only=True):
            self.defend()
        if not self.plans:
            if self.t < self.OPEN_WAIT_S and not self.seen:
                pass  # heavy decks: the bot plays first (Cole), or it opens at OPEN_WAIT_S
            else:
                self.decide()
        return self._act()

    def shadow_tick(self):
        """While a person plays this deck (deck_play.py's "you play the deck" mode): what the rules would play this
        tick, as (rule, card, (x, y) on the board), or None. Nothing is sent; the step counts as played so the plan
        moves on. The person's own plays go in through note_own_play."""
        self.tick()
        if self._sent is None:
            return None
        p, s, _, spot = self._sent
        self._sent = None
        if p.steps and p.steps[0] is s:
            p.steps.pop(0)
        if not p.steps and p in self.plans:
            self.plans.remove(p)
        return p.rule, s.card, spot

    def note_own_play(self, card: int, x: float, y: float):
        """A play of this deck's that landed, made by someone else (the person piloting it), on the board."""
        self.mine.append((round(self.t, 1), self.names.get(card, str(card)), round(x, 1), round(y, 1)))

    def _act(self):
        """The first plan's next step, once it's due and affordable. A step stays until the engine takes it (a card
        just drawn can't be played for 2 s: the engine's hand cooldown, PlayerState.h), then it's recorded in self.mine."""
        if self._sent is not None:  # last tick's play: did it land?
            p, s, spent, spot = self._sent
            self._sent = None
            if float(self.g.get_elixir_spent(1)) > spent + 0.5:
                if p.steps and p.steps[0] is s:
                    p.steps.pop(0)
                if not p.steps and p in self.plans:
                    self.plans.remove(p)
                self.mine.append((round(self.t, 1), s.card, round(spot[0], 1), round(spot[1], 1)))
        while self.plans:
            p = self.plans[0]
            if not p.steps:
                self.plans.pop(0)
                continue
            s = p.steps[0]
            if self.t < s.due - 1e-9:
                return None
            if self.t > s.expires:
                self.events.append({"t": round(self.t, 1), "rule": p.rule, "text": f"dropped {s.card}: couldn't play it"})
                self.plans.pop(0)
                continue
            if s.card not in self.hand():  # cycled away or never there: skip it rather than block the plan
                self.events.append({"t": round(self.t, 1), "rule": p.rule, "text": f"skipped {s.card}: not in hand"})
                p.steps.pop(0)
                continue
            if not self.can(s.card):
                return None
            at = s.at(self) if callable(s.at) else s.at
            spot = None if at is None else self.legal_spot(s.card, *at)
            if spot is None:
                self.events.append({"t": round(self.t, 1), "rule": p.rule, "text": f"dropped {s.card}: no spot"})
                p.steps.pop(0)
                continue
            self._sent = (p, s, float(self.g.get_elixir_spent(1)), spot)
            return self.hand().index(s.card), spot[0], 33.0 - spot[1]
        return None

    # ------------------------------------------------------------------ defence (urgent)
    # The deck's answers to each of our cards' units, best first: {unit name: [option, ...]}, an option a list of
    # (card, where, after_s). The first option whose first card is in hand is played (its later cards only if in hand
    # too). `where`: "pull" the middle spot that pulls a building-targeter off its lane (8.5 +- 1, 24.5); "front" in
    # front of the princess tower in the unit's lane, a little inside (lane x -+ 2, 23); "on" on the unit, a tile
    # ahead of it; "behind" behind the princess tower (lane x, 29.5); "bridge" this side's bridge in the unit's lane.
    ANSWERS: dict[str, list] = {}
    CHEAP: list[str] = []  # the fallback when no option is in hand: the first of these in hand, in "front"
    REANSWER_S = 5.0  # a threat still alive and near the tower this long after its answer gets another one
    SWARMS = ("Spear Goblins", "Barbarians", "Barbarian", "Spear Goblin")

    def where(self, kind: str, u) -> tuple[float, float] | None:
        live = next((v for v in self.g.get_units() if v.id == u.id and v.hp > 0), None)
        if live is None:
            return None
        lane = lane_of(live.x)
        inward = 1.0 if lane == 0 else -1.0
        if kind == "pull":
            return MID_X - inward, 24.5
        if kind == "front":
            return LANE_X[lane] + 2.0 * inward, 23.0
        if kind == "on":
            return live.x, min(live.y + 1.5, 30.0)
        if kind == "behind":
            return LANE_X[lane], 29.5
        if kind == "bridge":
            return BRIDGE_X[lane], MY_BRIDGE_Y
        raise ValueError(kind)

    def answer_options(self, u) -> list:
        """The deck's options for this unit (override for conditions, e.g. no Skeletons into a Bomber)."""
        return self.ANSWERS.get(u.name, [])

    def threats(self) -> list:
        """Bot troops worth an answer: across the river (or in it), swarms only two or more together."""
        out = []
        for u in self.crossed():
            if u.name in self.SWARMS or not u.name:
                if not u.name or sum(1 for v in self.crossed() if v.name == u.name and abs(v.x - u.x) < 3) < 2:
                    continue
            out.append(u)
        return sorted(out, key=lambda u: -u.y)  # deepest first

    def defend(self):
        for u in self.threats():
            when = self.answered_at.get(u.id)
            if when is not None:
                near = abs(u.y - 27.0) < 5 and abs(u.x - LANE_X[lane_of(u.x)]) < 5
                if not (near and self.t - when >= self.REANSWER_S and not self.defenders_near(u)):
                    continue
            if self.answer(u):
                return

    def defenders_near(self, u, r: float = 4.0) -> bool:
        return any(abs(m.x - u.x) + abs(m.y - u.y) < r for m in self.my_units())

    def answer(self, u) -> bool:
        again = u.id in self.answered_at
        self.answered_at[u.id] = self.t
        if u.name in self.SWARMS:  # one answer for the whole group
            for v in self.crossed():
                if v.name == u.name and abs(v.x - u.x) < 3:
                    self.answered_at[v.id] = self.t
        opts = [o for o in self.answer_options(u) if o and self.has(o[0][0]) and (not u.flying or o[0][0] in HITS_AIR)]
        opts.sort(key=lambda o: self.cost[o[0][0]] > self.elixir)  # the ones it can pay for now first, else in order
        for opt in opts:
            steps = [Step(c, (lambda b, k=k, u=u: b.where(k, u)), a) for c, k, a in opt
                     if c == opt[0][0] or self.has(c)]
            return self.queue("DEF", f"{u.name}{' again' if again else ''}: " + " + ".join(s.card for s in steps),
                              *steps, urgent=True)
        for c in self.CHEAP:
            if self.has(c) and (not u.flying or c in HITS_AIR):
                return self.queue("DEF", f"{u.name}: {c} (fallback)", Step(c, lambda b, u=u: b.where("front", u)),
                                  urgent=True)
        self.events.append({"t": round(self.t, 1), "rule": "DEF", "text": f"{u.name}: nothing in hand to answer it"})
        return False

    # ------------------------------------------------------------------ the deck's rules
    def decide(self):
        """Everything else, top to bottom (the deck's punish rules and its default)."""


def step_at(card: str, x: float, y: float, after_s: float = 0.0) -> Step:
    return Step(card, (x, y), after_s)


DECKS: dict[str, type[DeckBot]] = {}


def register(cls):
    DECKS[cls.KEY] = cls
    return cls
