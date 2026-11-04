"""Miner Control (Miner Rocket): docs/decks/miner-control.md as code. Rule ids match the doc's punish rules."""
from __future__ import annotations

from bot77.rl.situations import spell_tower_damage
from bot77.rl.deck_bots import (BRIDGE_X, LANE_X, MID_X, MY_BRIDGE_Y, DeckBot, Step, card_id, lane_of, register)

SAFE = ((1.0, 5.0), (16.0, 5.0))  # two tiles off each bot princess tower's outer back corner, out of king range
INFERNO_SPOT = (MID_X, 24.0)
ROCKET_R, TOWER_R = 2.0, 1.0  # blast radius; a princess tower is hit when its centre is within ROCKET_R + TOWER_R
VALUE = {"Giant": 5, "Battle Ram": 4, "Barbarians": 1, "Barbarian": 1, "Mini PEKKA": 4, "Mega Minion": 3,
         "Bomber": 2, "Goblin Hut": 5, "Spear Goblins": 0.7, "Spear Goblin": 0.7}
OVERTIME_S = 180.0
_MINER_CHIP: list[float] = []


def miner_tower_damage() -> float:
    """A lone Miner's damage to a princess tower before the tower kills it, measured in the engine (rule 1)."""
    if not _MINER_CHIP:
        import clash_royale_env as cre
        g = cre.ClashRoyaleEnv([0, 1, 2, 3, 5, 6, 7, 9], [0, 1, 2, 3, 5, 6, 7, 9], 3600,
                               cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
        g.seed(0)
        hp0 = g.get_tower_hp(1, 1)
        g.inject(card_id("Miner"), 1.0, 28.0, 0, -1.0, -1)
        g.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 200)
        _MINER_CHIP.append(float(hp0 - g.get_tower_hp(1, 1)))
    return _MINER_CHIP[0]


@register
class MinerControl(DeckBot):
    KEY = "miner"
    NAME = "Miner Control"
    DECK = ["Miner", "Rocket", "The Log", "Royal Delivery", "Inferno Tower", "Knight", "Electro Spirit", "Goblins"]

    # ------------------------------------------------------------------ helpers
    def out_of_cycle(self, name: str) -> bool:
        """The doc's cycle rule: among the bot's last 3 plays."""
        return any(s.name == name for s in self.seen[-3:])

    def target_lane(self) -> int:
        """One lane all game (Nate): the most damaged standing tower, the left one if neither is damaged."""
        hp = [self.tower_hp(0, k) or 1e9 for k in (0, 1)]  # a fallen tower: the other one
        return 1 if hp[1] < hp[0] else 0

    def pairing(self) -> str | None:
        """R6: Goblins with the Miner when Arrows and Fireball are both out of cycle, Knight if one is, else alone."""
        out = self.out_of_cycle("Arrows") + self.out_of_cycle("Fireball")
        return "Goblins" if out == 2 else "Knight" if out == 1 else None

    def miner_ready(self) -> bool:
        recent = any(c == "Miner" and self.t - t < 6.0 for t, c, _, _ in self.mine)
        return self.has("Miner") and not self.my_units("Miner") and not recent

    def inferno_reserve(self) -> bool:
        """Single elixir: keep elixir for the Inferno while the bot's Giant may be in its hand."""
        return self.phase == "single" and self.maybe_in_bot_hand("Giant") and self.has("Inferno Tower")

    def miner_plan(self, rule: str, text: str, lane: int, partner: str | None = None) -> bool:
        """Miner at the lane's safe spot; a ground partner can't cross, so it goes at this side's bridge."""
        steps = [Step("Miner", SAFE[lane])]
        if partner and self.has(partner):
            steps.append(Step(partner, (BRIDGE_X[lane], MY_BRIDGE_Y), 0.1))
        return self.queue(rule, text, *steps)

    def rocket_target(self, min_troops=6.0, min_with_tower=3.0):
        """R3: the blast centre that hits the most: troops worth min_troops, or min_with_tower plus a princess tower."""
        units = [u for u in self.bot_units() if u.name in VALUE]
        towers = [(LANE_X[k], 6.0) for k in (0, 1) if self.tower_hp(0, k) > 0]
        cands = [(u.x, u.y) for u in units]
        for u in units:  # pulled toward a tower, as far as the unit stays in the blast
            for tx, ty in towers:
                d = max(1e-6, ((tx - u.x) ** 2 + (ty - u.y) ** 2) ** 0.5)
                f = min(0.9 * ROCKET_R, d) / d
                cands.append((u.x + f * (tx - u.x), u.y + f * (ty - u.y)))
        best = None
        for cx, cy in cands:
            val = sum(VALUE[u.name] for u in units if (u.x - cx) ** 2 + (u.y - cy) ** 2 <= ROCKET_R ** 2)
            tw = any((tx - cx) ** 2 + (ty - cy) ** 2 <= (ROCKET_R + TOWER_R) ** 2 for tx, ty in towers)
            if (val >= min_troops or (tw and val >= min_with_tower)) and (best is None or val + 3 * tw > best[0]):
                best = (val + 3 * tw, (cx, cy), val, tw)
        return best

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[("Inferno Tower", "pull", 0)], [("Knight", "on", 0), ("Goblins", "on", 1.0)]],
        "Battle Ram": [[("Knight", "on", 0)], [("Goblins", "on", 0)]],  # the Barbarians it drops get their own answer
        "Mini PEKKA": [[("Goblins", "on", 0)], [("Knight", "on", 0), ("Electro Spirit", "on", 0.2)]],
        "Mega Minion": [[("Electro Spirit", "on", 0), ("Royal Delivery", "on", 0.5)], [("Royal Delivery", "on", 0)]],
        "Bomber": [[("Knight", "on", 0)], [("Royal Delivery", "on", 0)]],
        "Barbarians": [[("The Log", "on", 0)], [("Royal Delivery", "on", 0)], [("Knight", "front", 0)]],
        "Barbarian": [[("The Log", "on", 0)], [("Royal Delivery", "on", 0)], [("Knight", "front", 0)]],
    }
    CHEAP = ["Goblins", "Knight", "Electro Spirit", "Royal Delivery", "Inferno Tower"]  # the Inferno: nothing else left

    def answer_options(self, u):
        opts = super().answer_options(u)
        giant = [g for g in self.bot_troops() if g.name == "Giant" and abs(g.x - u.x) + abs(g.y - u.y) < 5]
        if u.name == "Battle Ram" and giant:  # Inferno only if the Ram leads a Giant push
            opts = [[("Inferno Tower", "pull", 0)]] + opts
        if u.name == "Bomber" and u.y >= 22:  # Miner on it near the tower
            opts = opts + [[("Miner", "on", 0)]]
        return opts

    def threats(self):
        # our troops only (the bot's Fireball/Arrows show up as units too); R8: Spear Goblins are left to the towers
        return [u for u in super().threats() if u.name in VALUE and u.name not in ("Spear Goblins", "Spear Goblin")]

    def answer(self, u):
        """Nothing that hits air in hand for a Mega Minion: leave it to the towers, no ground fallback."""
        if u.flying and not any(o and self.has(o[0][0]) for o in self.answer_options(u)):
            self.answered_at[u.id] = self.t
            return False
        return super().answer(u)

    def defend(self):
        # R2: a Giant up a lane -> Inferno at (8.5, 24) as it reaches the river; Goblins/Knight first on a Mini PEKKA or
        # Ram walking ahead of it; Log/Delivery on the support once it's near the Inferno
        for g in self.bot_troops():
            if g.name != "Giant" or g.y < 14.0 or g.id in self.answered_at or not self.has("Inferno Tower") \
                    or self.elixir < 5:
                continue
            self.answered_at[g.id] = self.t
            steps = []
            lead = [u for u in self.bot_troops() if u.name in ("Mini PEKKA", "Battle Ram") and u.y > g.y
                    and abs(u.x - g.x) < 4]
            for u in lead[:1]:
                self.answered_at[u.id] = self.t
                c = "Goblins" if self.has("Goblins") else "Knight" if self.has("Knight") else None
                if c:
                    steps.append(Step(c, lambda b, u=u: b.where("on", u)))
            wait = max(0.2 if steps else 0.0, 16.0 - g.y - 0.75)  # the Giant walks ~1 tile/s: down as it reaches ~17
            steps.append(Step("Inferno Tower", INFERNO_SPOT, wait))
            sup = [u for u in self.bot_troops() if u.name in ("Bomber", "Barbarian", "Barbarians", "Spear Goblins")
                   and abs(u.x - g.x) + abs(u.y - g.y) < 5]
            spell = "The Log" if self.has("The Log") else "Royal Delivery" if self.has("Royal Delivery") else None
            if sup and spell:
                steps.append(Step(spell, lambda b, u=sup[0]: b.where("on", u), 2.5))
            return self.queue("R2", f"Giant up the lane: {' + '.join(s.card for s in steps)}", *steps, urgent=True)
        # R8: Spear Goblins only get the Log when they stack with another troop
        for u in self.crossed():
            if u.name in ("Spear Goblins", "Spear Goblin") and u.id not in self.answered_at and self.has("The Log"):
                if any(v.name not in ("Spear Goblins", "Spear Goblin") and abs(v.x - u.x) + abs(v.y - u.y) < 2.5
                       for v in self.crossed()):
                    for v in self.crossed():
                        if abs(v.x - u.x) + abs(v.y - u.y) < 3:
                            self.answered_at[v.id] = self.t
                    return self.queue("R8", "Spear Goblins stacked with a troop: Log them",
                                      Step("The Log", lambda b, u=u: b.where("on", u)), urgent=True)
        super().defend()

    # ------------------------------------------------------------------ the punish rules and the default
    def on_bot_play(self, s):
        if self.committed(s.lane, 8.0) > 5:  # R4: remember the all-in; the counter waits for the bot to be low
            self.st["r4"] = (s.lane, self.t)
        miner = next(((t, x) for t, c, x, _ in reversed(self.mine) if c == "Miner"), None)
        if miner and self.t - miner[0] < 6.0 and s.lane == lane_of(miner[1]) and s.y <= 12.0:  # R5: paid on my Miner
            prev = self.st.get("r5")
            spent = prev[2] if prev and prev[0] == miner[0] else 0.0
            self.st["r5"] = (miner[0], lane_of(miner[1]), spent + s.cost)
        reach = (1.9 * ROCKET_R + TOWER_R) ** 2  # Hut and tower in one blast (rocket_target's pulled centre)
        if s.name == "Goblin Hut" and not any(self.tower_hp(0, k) > 0 and (s.x - LANE_X[k]) ** 2 + (s.y - 6.0) ** 2
                                              <= reach for k in (0, 1)):  # R8: out of R3's reach -> leave it
            self.fired["R8"] = self.fired.get("R8", 0) + 1
            self.events.append({"t": round(self.t, 1), "rule": "R8", "text": "Goblin Hut out of Rocket reach: leave"})

    def decide(self):
        el, bot_el, lane = self.elixir, self.bot_elixir, self.target_lane()
        # OPEN: wait to ~9, then Miner alone at the safe spot (the bot playing first is answered by DEF)
        if not self.seen and not self.mine:
            if el >= 9 and self.has("Miner"):
                self.miner_plan("OPEN", "opening: Miner alone at the safe spot", lane)
            return
        deep = any(u.y > 18.0 for u in self.bot_troops())
        # R1: a tower within a Rocket (or a lone Miner's chip) and nothing of the bot's past y 18 -> finish it
        for k in (0, 1):
            hp = self.tower_hp(0, k)
            if 0 < hp <= spell_tower_damage(self.ids["Rocket"]) and el >= 6 and not deep and self.has("Rocket"):
                return self.queue("R1", "Rocket the low tower", Step("Rocket", (LANE_X[k], 6.0)))
            if 0 < hp <= miner_tower_damage() and not deep and self.miner_ready():
                return self.miner_plan("R1", "Miner the low tower", k)
        # R3: 6+ elixir in one blast, or 3+ with a tower (a Hut by a tower) -> Rocket, unless a Giant needs the Inferno
        giant = any(u.name == "Giant" and u.y >= 10 for u in self.bot_troops()) and self.has("Inferno Tower")
        hit = self.rocket_target() if el >= 6 and self.has("Rocket") and not giant else None
        if hit and not any(u.y >= 12 and (u.x - hit[1][0]) ** 2 + (u.y - hit[1][1]) ** 2 > ROCKET_R ** 2
                           for u in self.bot_troops() if u.name in VALUE):  # and no other push coming
            return self.queue("R3", f"Rocket {hit[2]:.0f} elixir{' + tower' if hit[3] else ''}", Step("Rocket", hit[1]))
        # R4: after the all-in, the bot on 3 or less -> Miner with the surviving Knight/Goblins, same lane
        r4 = self.st.get("r4")
        if r4 and self.t - r4[1] < 15 and bot_el <= 3 and self.miner_ready():
            if any(lane_of(m.x) == r4[0] for n in ("Knight", "Goblins") for m in self.my_units(n)) and \
                    not [u for u in self.crossed() if lane_of(u.x) == r4[0]]:
                del self.st["r4"]
                return self.miner_plan("R4", "all-in answered: Miner with the survivors, same lane", r4[0])
        # R5: the bot paid 4+ for a lone Miner -> Miner + Electro Spirit on the same tower within 10 s
        r5 = self.st.get("r5")
        if r5 and r5[2] >= 4 and self.t - r5[0] < 10 and bot_el <= 4 and self.miner_ready():
            del self.st["r5"]
            return self.miner_plan("R5", f"bot paid {r5[2]:.0f} for the Miner: Miner again", r5[1], "Electro Spirit")
        # R6: Arrows and Fireball both out of cycle -> Miner + Goblins at the damaged tower (8+ while the Inferno waits
        # for a Giant)
        if self.pairing() == "Goblins" and el >= (8 if self.inferno_reserve() else 5) and self.miner_ready():
            return self.miner_plan("R6", "no small spell in the bot's hand: Miner + Goblins", lane, "Goblins")
        # R7: the bot is on 2 or less and nothing of its across -> Miner + Electro Spirit
        if bot_el <= 2 and not self.crossed() and el >= 4 and self.miner_ready():
            return self.miner_plan("R7", "bot is empty: Miner + Electro Spirit", lane, "Electro Spirit")
        return self._default()

    def _default(self):
        el, lane, rocket = self.elixir, self.target_lane(), spell_tower_damage(self.ids["Rocket"])
        calm = not self.crossed() and not any(u.y > 10 for u in self.bot_troops())
        if self.t >= OVERTIME_S:  # overtime: Rocket cycle on the lowest tower
            if el >= 6 and calm and self.has("Rocket"):
                return self.queue("DEFAULT", "overtime: Rocket cycle", Step("Rocket", (LANE_X[lane], 6.0)))
        elif self.phase != "single":  # double: Rocket a tower within two; else Miner + one cheap card each cycle
            if el >= 6 and calm and self.tower_hp(0, lane) <= 2 * rocket and self.has("Rocket"):
                return self.queue("DEFAULT", "double: Rocket a tower within two", Step("Rocket", (LANE_X[lane], 6.0)))
            if el >= 6 and self.miner_ready():
                partner = self.pairing() or "Electro Spirit"
                return self.miner_plan("DEFAULT", f"double: Miner + {partner}", lane, partner)
        elif el >= 9 and self.miner_ready():  # single: passive, Miner at the safe spot at 9+
            return self.miner_plan("DEFAULT", "single: Miner at the safe spot", lane)
        return self._cycle()

    def _cycle(self):
        """Nothing better at 9.5+: cycle a cheap card at the back so the bar never leaks."""
        if self.elixir < 9.5:
            return
        for card, at in (("Electro Spirit", (MID_X, 30.0)), ("Goblins", (MID_X, 31.0)), ("Knight", (MID_X, 30.0)),
                         ("Miner", SAFE[self.target_lane()])):
            if self.has(card):
                return self.queue("CYCLE", f"full bar: cycle {card}", Step(card, at))
