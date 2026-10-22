"""B v0's decision: when to play (rules), which card (step A v1), where (rules by the card's role).

Arena positions are KataCR arena-crop tiles: x 0-18 left to right, y down the screen; the river is at y 14.9, so our
half is y > 15. Every action is logged with its reason, so games can be reviewed and the rules replaced by learned
"when" / "where" models as the bot's own games come in.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot77.live.state import RIVER_Y, LiveState, Unit
from bot77.moves import MovePredictor, candidates

# Our half, in arena tiles (the crop is 18 wide, the river at 14.9)
BRIDGE = {"left": (3.5, 16.2), "right": (14.5, 16.2)}
BEHIND_TOWER = {"left": (3.5, 23.8), "right": (14.5, 23.8)}
CENTRE = (9.0, 20.0)
ENEMY_TOWER = {"left": (3.55, 5.7), "right": (14.5, 5.7)}  # princess tower centres (bot77.live.arena)
WIN_CONDITIONS = {"Hog Rider", "Royal Hogs", "Ram Rider", "Battle Ram", "Giant", "Golem", "Electro Giant", "Goblin Giant",
                  "Lava Hound", "Balloon", "Elixir Golem", "Wall Breakers", "Royal Giant", "Mega Knight", "P.E.K.K.A",
                  "Goblin Drill", "Skeleton Barrel", "Goblin Barrel"}
ON_TOWER = {"Miner", "Mighty Miner", "Goblin Barrel", "Goblin Drill", "Graveyard", "Rocket"}
CHEAP_DEFENDERS = {"Skeletons", "Electro Spirit", "Ice Spirit", "Fire Spirit", "Heal Spirit", "Goblins", "Bats",
                   "Spear Goblins", "Knight", "Ice Golem", "Valkyrie", "Mini P.E.K.K.A", "Guards", "Skeleton Army",
                   "Minions", "Mega Minion", "Firecracker", "Musketeer", "Archers", "Dart Goblin", "Electro Wizard"}


@dataclass
class Action:
    slot: int  # 0-3
    card: str
    x: float  # arena tiles
    y: float
    reason: str
    scores: list[float]


class Policy:
    def __init__(self, cards: list[dict], model_path: str = "data/models/moves_v1/model.pkl",
                 full_at: float = 9.5, idle_at: float = 7.0, idle_s: float = 6.0, min_gap_s: float = 1.2):
        self.a = MovePredictor(model_path)
        self.cards = {c["id"]: c for c in cards}
        self.names = {c["name"]: c["id"] for c in cards}
        self.full_at, self.idle_at, self.idle_s, self.min_gap_s = full_at, idle_at, idle_s, min_gap_s
        self.own: list[dict] = []  # our plays {"t", "card"}
        self.last_played: dict[int, float] = {}

    def record_own(self, t: float, card: str) -> None:
        self.own.append({"t": t, "card": card})
        if card in self.names:
            self.last_played[self.names[card]] = t

    def decide(self, st: LiveState, opp_plays: list[dict]) -> Action | None:
        if not st.in_battle or None in st.hand or st.elixir is None:
            return None
        since = st.t - self.own[-1]["t"] if self.own else 99.0
        if since < self.min_gap_s:
            return None
        threats = st.threats()
        why = ("defend" if threats else "full elixir" if st.elixir >= self.full_at else
               "idle" if st.elixir >= self.idle_at and since >= self.idle_s else None)
        if why is None:
            return None
        h = {"hand": st.hand, "next": st.next, "elixir": st.elixir, "t_match": st.elapsed_s, "multiplier": st.multiplier}
        rows = candidates(h, st.t, self.own, opp_plays, self.last_played, self.cards, self.names)
        scores = self.a.scores(rows)
        for k in sorted(range(4), key=lambda k: -scores[k]):
            info = self.cards.get(st.hand[k], {})
            cost = info.get("elixir") or 0
            if cost > st.elixir + 0.05:
                continue
            spot = self.place(info.get("name"), info.get("type"), threats, st)
            if spot is None:
                continue
            return Action(k, info.get("name"), spot[0], spot[1], why, scores)
        return None

    def place(self, name: str, typ: str, threats: list[Unit], st: LiveState) -> tuple[float, float] | None:
        """Where a card goes; None if it has no sensible target right now (e.g. a spell with nothing to hit)."""
        lane = (lambda x: "left" if x < 9 else "right")
        enemies = [u for u in st.units if u.team == "enemy" and u.card]
        if typ == "spell":
            # defending: only units on our half are targets (shadow check 2026-10-21: Barbarian Barrel was aimed at a
            # group still on the enemy side while a threat was at our tower)
            group = _biggest_group(threats if threats else enemies, min_units=1 if threats else 2)
            if group:
                return group
            if threats:
                return None  # defending: a spell with nothing on our side waits (dry run: Earthquake went to a tower)
            if name in ("Earthquake", "Rocket", "Fireball", "Poison", "Lightning"):
                return ENEMY_TOWER[_attack_lane(threats, enemies)]
            return None  # a small spell with no target waits
        if typ == "building":
            return (threats[0].x, max(threats[0].y + 2.5, 18.5)) if threats and name in ("Tesla", "Cannon", "Bomb Tower", "Inferno Tower") \
                else CENTRE
        if name in ON_TOWER and not threats:
            return ENEMY_TOWER[_attack_lane(threats, enemies)]
        if threats:
            t = max(threats, key=lambda u: u.y)  # the one closest to our towers
            if name in ON_TOWER and name != "Mighty Miner":
                return None  # Miner / Goblin Barrel don't defend; Mighty Miner tanks in front of the threat
            if name in WIN_CONDITIONS:  # counter-push in the other lane, only with elixir to spare
                return BRIDGE["right" if lane(t.x) == "left" else "left"] if (st.elixir or 0) >= 7 else None
            return (min(max(t.x, 1.0), 17.0), min(max(t.y + 2.0, 16.0), 24.0))
        if name in WIN_CONDITIONS:
            return BRIDGE[_attack_lane(threats, enemies)]
        return BEHIND_TOWER[_attack_lane(threats, enemies)]


def _attack_lane(threats: list[Unit], enemies: list[Unit]) -> str:
    """The lane with fewer enemy units (their defence) — left on a tie."""
    left = sum(1 for u in enemies if u.x < 9)
    right = len(enemies) - left
    return "right" if right < left else "left"


def _biggest_group(units: list[Unit], radius: float = 2.5, min_units: int = 2) -> tuple[float, float] | None:
    """Centre of the largest cluster of enemy units (>= min_units within radius tiles)."""
    best = None
    for u in units:
        near = [v for v in units if (v.x - u.x) ** 2 + (v.y - u.y) ** 2 <= radius ** 2]
        if len(near) >= min_units and (best is None or len(near) > len(best)):
            best = near
    if not best:
        return None
    return sum(v.x for v in best) / len(best), sum(v.y for v in best) / len(best)
