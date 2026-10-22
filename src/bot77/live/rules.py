"""Playbook rules (playbook step 3, 2026-10-22): the principles from Cole's play reviews as checks the policy applies
before the move predictor (A) gets a say. Read from config/playbook_rules.yaml (one source for the numbers and card
lists, so a review finding becomes a yaml edit). Principles and the reviews they come from:
.claude/skills/review-bot-plays/SKILL.md, data/ground_truth/reviews/bot_review_1_*.json.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from bot77.live.state import LiveState, Unit

RULES_PATH = Path("config/playbook_rules.yaml")


def lane(x: float) -> str:
    return "left" if x < 9 else "right"


class Rules:
    def __init__(self, elixir: dict[str, int], path: Path = RULES_PATH):
        r = yaml.safe_load(path.read_text())
        self.r, self.elixir = r, elixir
        self.spots = {k: {s: tuple(v) for s, v in d.items()} for k, d in r["spots"].items()}
        self.when = r["when"]
        self.not_threats, self.tanks = set(r["not_threats"]), set(r["tanks"])
        self.never_defend = set(r["never_defend_with"])
        self.ground_only, self.air = set(r["ground_only"]), set(r["air"])
        self.spell_kills = {c: set(g["kills"]) for g in r["spells"].values() if isinstance(g, dict) for c in g["cards"]}
        self.ranged, self.on_target = set(r["placement"]["ranged"]), set(r["placement"]["on_target"])
        self.sup = r["support"]

    # -- when -------------------------------------------------------------------------------------------------------
    def threats(self, st: LiveState, answered: dict[str, float]) -> list[Unit]:
        """Enemy units worth answering: across the bridge, not answered in the last few seconds, and not only cheap
        units the tower handles. Most dangerous first (card cost, then closest to our towers)."""
        crossed = [u for u in st.units if u.team == "enemy" and u.card and u.y >= self.when["crossed_y"]]
        crossed = [u for u in crossed if st.t - answered.get(u.card, -1e9) > self.when["answered_s"]]
        if not crossed:
            return []
        cards = {u.card for u in crossed}
        if cards <= self.not_threats and sum(self.elixir.get(c, 0) for c in cards) < self.r["not_worth_below"]:
            return []
        return sorted(crossed, key=lambda u: (-self.elixir.get(u.card, 0), -u.y))

    # -- what -------------------------------------------------------------------------------------------------------
    def answers(self, target: str) -> list[str]:
        a = self.r["answers"]
        return a.get(target) or (a["tank"] if target in self.tanks else [])

    def can_answer(self, name: str, typ: str, target: Unit) -> bool:
        if name in self.never_defend:
            return False  # "Giant isn't a defender"
        if target.card in self.air and name in self.ground_only:
            return False
        if typ == "spell":
            return target.card in self.spell_kills.get(name, set())  # no Arrows on a lone Knight / Giant / building
        return True

    # -- where ------------------------------------------------------------------------------------------------------
    def defend_spot(self, name: str, typ: str, target: Unit, st: LiveState) -> tuple[float, float] | None:
        if typ == "spell":
            return self.spell_spot(name, st)
        if typ == "building":
            return (target.x, min(max(target.y + 2.5, 18.5), 23.0))
        x = min(max(target.x, 1.0), 17.0)
        if name in self.ranged:  # away from what it shoots, covered by both towers
            return self.spots["front_of_king"][lane(target.x)]
        if name in self.on_target:
            return x, min(max(target.y, self.when["crossed_y"]), 24.0)
        return x, min(max(target.y + self.r["placement"]["melee_ahead"], self.when["crossed_y"]), 24.0)

    def spell_spot(self, name: str, st: LiveState) -> tuple[float, float] | None:
        """Centre of the best group on our half this spell kills, if what it kills is worth the spell."""
        kills, rad = self.spell_kills.get(name, set()), self.r["spells"]["radius"]
        units = [u for u in st.units if u.team == "enemy" and u.card in kills and u.y >= self.when["crossed_y"] - 1.5]
        best, best_v = None, 0.0
        for u in units:
            near = [v for v in units if (v.x - u.x) ** 2 + (v.y - u.y) ** 2 <= rad ** 2]
            val = sum(self.elixir.get(c, 0) for c in {v.card for v in near})
            if val > best_v:
                best, best_v = near, val
        if not best or best_v < self.r["spells"]["min_value_ratio"] * self.elixir.get(name, 99):
            return None
        return sum(v.x for v in best) / len(best), sum(v.y for v in best) / len(best)

    # -- support / leak ---------------------------------------------------------------------------------------------
    def support_target(self, st: LiveState, own: list[dict]) -> Unit | None:
        """Our tank walking to the bridge with nothing behind it yet ("support the Giant")."""
        lo, hi = self.sup["window"]
        tanks = [u for u in st.units if u.team == "friendly" and u.card in self.sup["after"] and lo <= u.y <= hi]
        if not tanks:
            return None
        last_tank = max((p["t"] for p in own if p["card"] in self.sup["after"]), default=None)
        if last_tank is None or any(p["t"] > last_tank and p["card"] in self.sup["cards"] for p in own):
            return None
        return max(tanks, key=lambda u: u.y)

    def support_spot(self, tank: Unit) -> tuple[float, float]:
        return tank.x, min(tank.y + self.sup["behind"], 25.0)

    def leak_spot(self, name: str, typ: str, attack_lane: str) -> tuple[float, float] | None:
        if typ == "spell":
            return None  # spells are never dumped
        if typ == "building":
            return 9.0, 20.0
        return self.spots["behind_king"][attack_lane]  # "Giant goes in the back"; cheap cycles behind the king
