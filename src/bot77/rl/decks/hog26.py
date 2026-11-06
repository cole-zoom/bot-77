"""2.6 Hog Cycle: docs/decks/hog-2.6.md as code. Rule ids match the doc's punish rules (R1 = Cole's example)."""
from __future__ import annotations

from bot77.rl.situations import spell_tower_damage
from bot77.rl.deck_bots import (LANE_X, MID_X, MY_BRIDGE_Y, DeckBot, Step, lane_of, register)

CANNON_SPOT = (MID_X, 24.5)
HOG_X = (1.0, 17.0)  # Cole: the outer edge of the bridge (20 of his 27 Hogs, 2026-11-06)
MUSK_DEEP_Y = 28.0


@register
class Hog26(DeckBot):
    KEY = "hog26"
    NAME = "2.6 Hog Cycle"
    DECK = ["Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons", "Ice Spirit", "Fireball", "The Log"]

    # ------------------------------------------------------------------ helpers
    def hog_plan(self, rule: str, text: str, lane: int, *, spirit=False, golem=False, musk=False) -> bool:
        """Hog at this lane's bridge, with an Ice Golem in front / Ice Spirit or Musketeer behind if asked and in hand."""
        bx = HOG_X[lane]
        steps = []
        if golem and self.has("Ice Golem"):
            steps.append(Step("Ice Golem", (bx, MY_BRIDGE_Y)))
        steps.append(Step("Hog Rider", (bx, MY_BRIDGE_Y), 0.3 if steps else 0.0))
        if spirit and self.has("Ice Spirit"):
            steps.append(Step("Ice Spirit", (bx, MY_BRIDGE_Y + 1.0), 0.2))
        if musk and self.has("Musketeer"):
            steps.append(Step("Musketeer", (LANE_X[lane], MY_BRIDGE_Y + 1.5), 0.3))
        return self.queue(rule, text, *steps)

    def hog_guard(self) -> bool:
        """R10: hold the Hog if it leaves us < 3 while the bot has a heavy card in hand and 5+ elixir."""
        heavy = any(self.maybe_in_bot_hand(c) for c in ("Giant", "Battle Ram", "Mini PEKKA"))
        return self.elixir - self.cost["Hog Rider"] < 3 and heavy and self.bot_elixir >= 5

    def musk_y(self) -> float:
        """R5: Musketeer deep (out of a Fireball with the Cannon) while the bot may have Fireball."""
        return MUSK_DEEP_Y if self.maybe_in_bot_hand("Fireball") else 25.5

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[("Cannon", "pull", 0)], [("Ice Golem", "pull", 0), ("Skeletons", "on", 2.0)],
                  [("Musketeer", "behind", 0)]],
        "Battle Ram": [[("Cannon", "pull", 0), ("The Log", "on", 1.5)],  # R6
                       [("Ice Spirit", "on", 0), ("Musketeer", "behind", 0.3)], [("Ice Golem", "pull", 0)]],
        "Mini PEKKA": [[("Skeletons", "on", 0)], [("Ice Golem", "on", 0), ("Cannon", "front", 0.3)],
                       [("Cannon", "front", 0)], [("Musketeer", "behind", 0)]],
        "Mega Minion": [[("Musketeer", "behind", 0)], [("Ice Spirit", "on", 0)]],
        "Bomber": [[("Ice Spirit", "on", 0)], [("Musketeer", "behind", 0)], [("Ice Golem", "on", 0)]],
        "Barbarians": [[("The Log", "on", 0)], [("Cannon", "front", 0)]],
        "Barbarian": [[("The Log", "on", 0)], [("Cannon", "front", 0)]],
        "Spear Goblins": [[("The Log", "on", 0)], [("Ice Spirit", "on", 0)]],
    }
    CHEAP = ["Ice Spirit", "Skeletons", "Ice Golem", "Cannon", "Musketeer"]

    def answer_options(self, u):
        opts = super().answer_options(u)
        if u.name == "Mini PEKKA":  # R8: no Skeletons into a Bomber or a likely Arrows
            bomber = any(b.name == "Bomber" and abs(b.x - u.x) + abs(b.y - u.y) < 5 for b in self.bot_troops())
            if bomber or (self.maybe_in_bot_hand("Arrows") and self.bot_elixir >= 3):
                opts = [o for o in opts if o[0][0] != "Skeletons"]
        if self.maybe_in_bot_hand("Fireball"):  # R5: a Musketeer goes deep, off the Cannon
            opts = [[(c, "behind" if c == "Musketeer" else k, a) for c, k, a in o] for o in opts]
        return opts

    def defend(self):
        lanes = {lane_of(u.x) for u in self.threats()}
        if len(lanes) == 2 and not self.st.get("r7_at") or (len(lanes) == 2 and self.t - self.st["r7_at"] > 15):
            # R7: both lanes at once -> Cannon on the heavier side, Musketeer on the other
            heavy = max((0, 1), key=lambda k: self.committed(k, 8.0))
            self.st["r7_at"] = self.t
            for u in self.threats():
                self.answered_at[u.id] = self.t
            steps = [Step(c, at) for c, at in (("Cannon", (6.0 if heavy == 0 else 11.0, 24.5)),
                                               ("Musketeer", (LANE_X[1 - heavy], 27.0))) if self.has(c)]
            if steps:
                self.queue("R7", f"both lanes: {' + '.join(s.card for s in steps)}", *steps, urgent=True)
                return
        super().defend()

    # ------------------------------------------------------------------ the punish rules and the default
    def on_bot_play(self, s):
        # R2: a 4+ card in the back in single elixir -> Hog the other bridge within 1 s
        if self.phase == "single" and s.cost >= 4 and s.y <= 6.0 and s.name not in ("Fireball", "Arrows"):
            if self.has("Hog Rider") and not self.hog_guard():
                self.hog_plan("R2", f"{s.name} in the back: Hog the other bridge", 1 - s.lane,
                              spirit=self.elixir >= 5)

    def decide(self):
        t, el, bot_el = self.t, self.elixir, self.bot_elixir
        # Opening: wait; Hog + Ice Spirit at a bridge once at 7 elixir if the bot hasn't played
        if not self.seen and not self.mine:
            if el >= 7 and self.has("Hog Rider"):
                self.hog_plan("OPEN", "opening: Hog + Ice Spirit at a bridge", int(self.rng.integers(2)), spirit=True)
            return
        if not self.has("Hog Rider"):
            return self._default_no_hog()
        # R1 (Cole): the bot put > 5 into one lane, nothing in the other -> Hog + Ice Spirit the other bridge
        for lane in (0, 1):
            key = ("r1", int(t // 6))
            if self.committed(lane) > 5 and self.committed(1 - lane) == 0 and key not in self.st:
                self.st[key] = True
                return self.hog_plan("R1", f"all-in in the {'left' if lane == 0 else 'right'} lane: Hog + Ice "
                                           "Spirit the other bridge", 1 - lane, spirit=True)
        guard = self.hog_guard()
        # R11: a bot princess tower within a Fireball (in overtime, Fireball + Log) -> spell it
        for lane in (0, 1):
            hp = self.tower_hp(0, lane)
            if 0 < hp <= spell_tower_damage(self.ids["Fireball"]) and self.can("Fireball"):
                return self.queue("R11", "Fireball the low tower", Step("Fireball", (LANE_X[lane], 6.0)))
        # R3: the bot is on 2 or less, nothing of its across -> Hog the weaker tower's lane
        if bot_el <= 2 and not [u for u in self.bot_troops() if u.y >= 15.5] and el >= 4 and not guard:
            return self.hog_plan("R3", "bot is empty: Hog the weaker tower", self.weaker_bot_lane(), musk=el >= 8)
        # R4: Mini P.E.K.K.A out of the bot's hand and no Goblin Hut up -> Hog the other lane of its last play
        last = self.last_bot_play()
        other = 1 - last.lane if last else int(self.rng.integers(2))
        if self.out_of_bot_hand("Mini PEKKA") and not self.bot_buildings("Goblin Hut") and not guard and el >= 4:
            return self.hog_plan("R4", "Mini P.E.K.K.A is out of cycle: Hog", other, golem=el >= 6)
        # R9: ahead -> Hog every cycle, never sit at 10
        if self.leading() and el >= 4 and not guard:
            return self.hog_plan("R9", "ahead: Hog every cycle", other, spirit=el >= 6)
        # DEFAULT
        if self.phase == "single":
            if el >= 9.5 or (el >= 7 and not guard and self.maybe_in_bot_hand("Mini PEKKA") is False):
                return self.hog_plan("DEFAULT", "single: Hog the lane opposite the bot's last push", other,
                                     golem=self.maybe_in_bot_hand("Mini PEKKA") and el >= 6)
        else:
            if el >= 6 and not guard:
                if self.has("Ice Golem"):
                    return self.hog_plan("DEFAULT", f"{self.phase}: Ice Golem + Hog", other, golem=True)
                return self.hog_plan("DEFAULT", f"{self.phase}: Hog + Musketeer", other,
                                     musk=self.maybe_in_bot_hand("Fireball") is False)

    def _default_no_hog(self):
        """No Hog in hand: cycle a cheap card at 9.5+ so the bar never leaks, toward the next Hog."""
        if self.elixir < 9.5:
            return
        for card, at in (("Ice Spirit", (MID_X, 30.0)), ("Skeletons", (MID_X, 31.0)), ("Ice Golem", (MID_X, 30.0)),
                         ("Musketeer", (LANE_X[int(self.rng.integers(2))], 29.0))):
            if self.has(card):
                return self.queue("CYCLE", f"full bar, no Hog: cycle {card} at the back", Step(card, at))
