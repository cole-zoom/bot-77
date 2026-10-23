"""B v0's decision. Since 2026-10-22 the playbook rules (bot77.live.rules, config/playbook_rules.yaml, from Cole's
reviews) decide when and where and pick the answer to an attack; A v1 ranks the cards the rules leave open. rules=False
is the earlier B v0: when to play (rules), which card (step A v1), where (playbook spots / rules by the card's role).

Arena positions are KataCR arena-crop tiles: x 0-18 left to right, y down the screen; the river is at y 14.9, so our
half is y > 15. Every action is logged with its reason, so games can be reviewed and the rules replaced by learned
"when" / "where" models as the bot's own games come in.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot77.live.state import RIVER_Y, LiveState, Unit
from bot77.features import add_board_features, candidates, live_board
from bot77.moves import MovePredictor

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
    model_spot: tuple[float, float] | None = None  # where the placement model would put it (shadow, 2026-10-22)


class Policy:
    def __init__(self, cards: list[dict], model_path: str | None = None, playbook: bool = True, rules: bool = True,
                 place_path: str | None = None,
                 full_at: float = 9.5, idle_at: float = 7.0, idle_s: float = 6.0, min_gap_s: float = 1.2):
        from bot77.models import adopted_path  # the registry's adopted models (phase 5), today's files as fallback
        model_path = model_path or adopted_path("card", "data/models/moves_v1/model.pkl")
        place_path = place_path or adopted_path("place", "data/models/place_v2/model.pkl")
        self.a = MovePredictor(model_path)
        self.cards = {c["id"]: c for c in cards}
        self.names = {c["name"]: c["id"] for c in cards}
        self.full_at, self.idle_at, self.idle_s, self.min_gap_s = full_at, idle_at, idle_s, min_gap_s
        self.own: list[dict] = []  # our plays {"t", "card"}
        self.book = {}
        if playbook:  # LanceDB `playbook` (scripts/build_playbook.py): typical spot of each answer to each card
            import lancedb
            for b in lancedb.connect("data/lancedb").open_table("playbook").to_arrow().to_pylist():
                if b["ans_x"] is not None:
                    self.book[(b["opp_card"], b["opp_zone"], b["answer"])] = b
        self.last_played: dict[int, float] = {}
        self.answered: dict[str, float] = {}  # attacker card -> when we last answered it
        self.placer = None  # placement model (bot77.place), shadow only: logged next to each play, not tapped
        from pathlib import Path
        if Path(place_path).exists():
            from bot77.place import Placer
            self.placer = Placer(place_path)
        self.rules = None
        if rules:  # playbook rules from Cole's reviews (config/playbook_rules.yaml, bot77.live.rules)
            from bot77.live.rules import Rules
            self.rules = Rules({c["name"]: c.get("elixir") or 0 for c in cards})

    def record_own(self, t: float, card: str) -> None:
        self.own.append({"t": t, "card": card})
        if card in self.names:
            self.last_played[self.names[card]] = t

    def decide(self, st: LiveState, opp_plays: list[dict]) -> Action | None:
        act = self._decide(st, opp_plays)
        if act is not None and self.placer is not None:
            try:
                from bot77.features import summary_features
                units, towers = live_board(st.units)
                cid = self.names.get(act.card)
                info = self.cards.get(cid, {})
                el = {c["name"]: c.get("elixir") or 0 for c in self.cards.values()}
                act.model_spot = self.placer.place(act.card, info.get("type"), cid, units, towers,
                                                   summary_features(units, towers, el, self.names), el)
            except Exception:  # noqa: BLE001 -- shadow only: never cost a play
                pass
        return act

    def _decide(self, st: LiveState, opp_plays: list[dict]) -> Action | None:
        if not st.in_battle or None in st.hand or st.elixir is None:
            return None
        since = st.t - self.own[-1]["t"] if self.own else 99.0
        if since < self.min_gap_s:
            return None
        if self.rules is None:
            return self._decide_v0(st, opp_plays, since)
        h = {"hand": st.hand, "next": st.next, "elixir": st.elixir, "t_match": st.elapsed_s, "multiplier": st.multiplier}
        scores = self.a.scores(self._rows(h, st, opp_plays))
        by_a = sorted(range(4), key=lambda k: -scores[k])
        info = [self.cards.get(c, {}) for c in st.hand]
        name = [i.get("name") for i in info]
        ok = [(i.get("elixir") or 0) <= st.elixir + 0.05 for i in info]
        R = self.rules

        # defend: the playbook's answer to the most dangerous attacker that crossed, else A's pick among the cards that
        # can answer it; nothing that can -> wait and let the towers work
        for target in R.threats(st, self.answered):
            pref = [name.index(n) for n in R.answers(target.card) if n in name]
            for k in pref + [k for k in by_a if k not in pref]:
                if not ok[k] or not R.can_answer(name[k], info[k].get("type"), target):
                    continue
                spot = R.defend_spot(name[k], info[k].get("type"), target, st)
                if spot is not None:
                    self.answered[target.card] = st.t
                    return Action(k, name[k], spot[0], spot[1], "defend (rules)" if k in pref else "defend (A)", scores)

        # support our tank on its way to the bridge
        tank = R.support_target(st, self.own)
        if tank is not None:
            for n in R.sup["cards"]:
                if n in name and ok[name.index(n)]:
                    x, y = R.support_spot(tank)
                    return Action(name.index(n), n, x, y, "support (rules)", scores)

        # about to waste elixir: build a push from the back, else cycle a cheap card behind the king
        if st.elixir >= self.full_at:
            enemies = [u for u in st.units if u.team == "enemy" and u.card]
            att = _attack_lane([], enemies)
            order = [name.index(n) for n in R.r["leak_order"] if n in name]
            for k in order + [k for k in by_a if k not in order]:
                spot = R.leak_spot(name[k], info[k].get("type"), att) if ok[k] else None
                if spot is not None:
                    return Action(k, name[k], spot[0], spot[1], "full elixir (rules)", scores)
        return None

    def _rows(self, h: dict, st: LiveState, opp_plays: list[dict]) -> list[dict]:
        """A's 4 candidate rows; with the board features when the model was trained with them (A v2)."""
        rows = candidates(h, st.t, self.own, opp_plays, self.last_played, self.cards, self.names)
        if self.a.uses_board:
            add_board_features(rows, *live_board(st.units), self.cards)
        return rows

    def _decide_v0(self, st: LiveState, opp_plays: list[dict], since: float) -> Action | None:
        """B v0 before the playbook rules (2026-10-21): when rules + A's card + playbook / rule spots."""
        threats = st.threats()
        why = ("defend" if threats else "full elixir" if st.elixir >= self.full_at else
               "idle" if st.elixir >= self.idle_at and since >= self.idle_s else None)
        if why is None:
            return None
        h = {"hand": st.hand, "next": st.next, "elixir": st.elixir, "t_match": st.elapsed_s, "multiplier": st.multiplier}
        scores = self.a.scores(self._rows(h, st, opp_plays))
        for k in sorted(range(4), key=lambda k: -scores[k]):
            info = self.cards.get(st.hand[k], {})
            cost = info.get("elixir") or 0
            if cost > st.elixir + 0.05:
                continue
            spot, how = (self.answer_spot(info.get("name"), st, opp_plays) if info.get("type") != "spell" else None), "playbook"
            if spot is None:
                spot, how = self.place(info.get("name"), info.get("type"), threats, st), "rules"
            if spot is None:
                continue
            return Action(k, info.get("name"), spot[0], spot[1], f"{why} ({how})", scores)
        return None

    def answer_spot(self, name: str, st: LiveState, opp_plays: list[dict], recent_s: float = 6.0):
        """Where strong players put this card when answering the opponent's last play (playbook step 2): their
        typical spot, in the lane of the opponent's unit. On 48 held-back replays: median 5.0 tiles from the real
        spot and 75% same lane, vs 8.5 tiles and 31% for the rules (scripts/eval_playbook_place.py). None when the
        opponent hasn't played in the last few seconds or the playbook has no such answer."""
        if not self.book or not opp_plays or st.t - opp_plays[-1]["t"] > recent_s:
            return None
        opp = opp_plays[-1]["card"]
        units = [u for u in st.units if u.team == "enemy" and u.card == opp]
        if not units:
            return None
        u = max(units, key=lambda u: u.y)
        z = "our_half" if u.y > RIVER_Y + 1.5 else "bridge" if u.y > RIVER_Y - 3.0 else "their_half"
        b = self.book.get((opp, z, name)) or self.book.get((opp, "any", name))
        if b is None:
            return None
        x = b["ans_x"] if (u.x < 9) == (b["ans_x"] < 9) else 18 - b["ans_x"]
        return x, b["ans_y"]

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
