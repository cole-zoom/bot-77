"""Classic Log Bait: docs/decks/log-bait.md as code. Rule ids match the doc's punish rules (R1..R11)."""
from __future__ import annotations

from bot77.rl.situations import REGULATION_END_TICK, spell_tower_damage
from bot77.rl.deck_bots import (BOT_TOWER_Y, LANE_X, MID_X, MY_BRIDGE_Y, RIVER_LO, DeckBot, Step, lane_of, register)

INFERNO_SPOT = (MID_X, 24.5)
PRINCESS_BRIDGE_Y = 19.0
CORNER = ((2.5, 31.5), (14.5, 31.5))
LOGABLE = ("Bomber", "Spear Goblins", "Barbarians", "Barbarian")
SPELLS = ("Arrows", "Fireball")


@register
class LogBait(DeckBot):
    KEY = "log_bait"
    NAME = "Log Bait"
    DECK = ["Goblin Barrel", "Princess", "Goblin Gang", "Knight", "Inferno Tower", "Rocket", "The Log", "Ice Spirit"]

    # ------------------------------------------------------------------ helpers
    @property
    def clock_s(self) -> float:  # seconds left in regulation (negative in overtime)
        return (REGULATION_END_TICK - self.g.get_current_tick()) / 10.0

    def bomber_near(self, lane: int, r: float = 4.0) -> bool:
        return any(u.name == "Bomber" and abs(u.x - LANE_X[lane]) + abs(u.y - BOT_TOWER_Y) < r
                   for u in self.bot_troops())

    def barrel_lane(self, prefer: int | None = None) -> int | None:
        """R3/R8: a standing bot tower with no Bomber at it; `prefer` if it qualifies, else the lower-HP one."""
        ok = [k for k in (0, 1) if self.tower_hp(0, k) > 0 and not self.bomber_near(k)]
        if prefer in ok:
            return prefer
        return min(ok, key=lambda k: (self.tower_hp(0, k), self.rng.random())) if ok else None

    def barrel(self, rule: str, text: str, lane: int, *extra: Step) -> bool:
        return self.queue(rule, text, Step("Goblin Barrel", (LANE_X[lane], BOT_TOWER_Y)), *extra)

    def princess_at(self, lane: int, y: float = PRINCESS_BRIDGE_Y, after_s: float = 0.0) -> Step:
        return Step("Princess", (LANE_X[lane], y), after_s)

    def other_lane(self) -> int:
        return 1 - self.seen[-1].lane if self.seen else int(self.rng.integers(2))

    def alone(self, u, r: float = 5.0) -> bool:
        return not any(v.id != u.id and abs(v.x - u.x) + abs(v.y - u.y) < r for v in self.bot_troops())

    def where(self, kind, u):
        if kind == "ahead":  # R7: in front of a Ram / Mini P.E.K.K.A, unit y + 2.5 capped at 26
            live = next((v for v in self.g.get_units() if v.id == u.id and v.hp > 0), None)
            return None if live is None else (live.x, min(live.y + 2.5, 26.0))
        return super().where(kind, u)

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[("Inferno Tower", "pull", 0)], [("Knight", "on", 0), ("Goblin Gang", "on", 1.0)],
                  [("Goblin Gang", "on", 0)]],
        "Battle Ram": [[("Knight", "on", 0), ("Ice Spirit", "on", 0.2)], [("Goblin Gang", "on", 0)],
                       [("Ice Spirit", "on", 0)], [("Inferno Tower", "pull", 0)]],
        "Mini PEKKA": [[("Goblin Gang", "ahead", 0)], [("Knight", "on", 0), ("Ice Spirit", "on", 0.2)],
                       [("Inferno Tower", "front", 0)]],
        "Mega Minion": [[("Princess", "behind", 0)], [("Goblin Gang", "on", 0)], [("Inferno Tower", "front", 0)]],
        "Bomber": [[("Princess", "behind", 0)], [("Knight", "on", 0)]],
        "Barbarians": [[("The Log", "on", 0)], [("Knight", "front", 0)]],
        "Barbarian": [[("The Log", "on", 0)], [("Knight", "front", 0)]],
        "Spear Goblins": [[("The Log", "on", 0)], [("Princess", "behind", 0)], [("Ice Spirit", "on", 0)]],
    }
    CHEAP = ["Ice Spirit", "Knight", "Goblin Gang", "Princess"]

    def answer_options(self, u):
        opts = super().answer_options(u)
        if u.name == "Mini PEKKA" and any(v.name == "Giant" for v in self.crossed()):  # Inferno if it comes with Giant
            opts = [[("Inferno Tower", "pull", 0)]] + opts
        return opts

    def threats(self):
        # Spells landing on our side aren't troops to answer; R1: the Giant waits for the Inferno until y 18.5
        hold = self.has("Inferno Tower")
        return [u for u in super().threats() if u.name not in SPELLS
                and not (hold and u.name == "Giant" and u.y < MY_BRIDGE_Y)]

    def defend(self):
        for u in self.threats():
            if u.id in self.answered_at:
                continue
            # R1: Giant across, Inferno in hand, 5+ -> (Log on Log-able support), Inferno middle, Knight on support
            if u.name == "Giant" and self.has("Inferno Tower") and self.elixir >= 5:
                sup = [v for v in self.crossed() if v.id != u.id and abs(v.x - u.x) + abs(v.y - u.y) < 5]
                logable = [v for v in sup if v.name in LOGABLE]
                steps = []
                if logable and self.has("The Log"):
                    steps.append(Step("The Log", lambda b, v=logable[0]: b.where("on", v)))
                steps.append(Step("Inferno Tower", INFERNO_SPOT, 0.3 if steps else 0.0))
                if sup and self.has("Knight"):
                    steps.append(Step("Knight", lambda b, v=sup[0]: b.where("on", v), 1.0))
                for v in [u] + sup:
                    self.answered_at[v.id] = self.t
                self.st["r1_lane"] = lane_of(u.x)
                return self.queue("R1", "Giant: " + " + ".join(s.card for s in steps), *steps, urgent=True)
            # R7: a Ram / Mini P.E.K.K.A alone -> Gang ahead of it (or Knight + Ice Spirit), Princess to counter
            if u.name in ("Battle Ram", "Mini PEKKA") and self.alone(u):
                if self.has("Goblin Gang"):
                    steps = [Step("Goblin Gang", lambda b, u=u: b.where("ahead", u))]
                elif self.has("Knight"):
                    steps = [Step("Knight", lambda b, u=u: b.where("ahead", u))]
                    if self.has("Ice Spirit"):
                        steps.append(Step("Ice Spirit", lambda b, u=u: b.where("on", u), 0.2))
                else:
                    continue
                if self.has("Princess"):
                    steps.append(self.princess_at(lane_of(u.x), after_s=2.0))
                self.answered_at[u.id] = self.t
                return self.queue("R7", f"{u.name} alone: " + " + ".join(s.card for s in steps), *steps, urgent=True)
        super().defend()

    # ------------------------------------------------------------------ the punish rules and the default
    def decide(self):
        t, el, bot_el = self.t, self.elixir, self.bot_elixir
        arrows_out, fireball_out = self.out_of_bot_hand("Arrows"), self.out_of_bot_hand("Fireball")
        # OPEN: Barrel a tower at 7+; the bot played first -> Barrel the other lane if it spent 4+
        if not self.st.get("opened"):
            if not self.seen:
                if el >= 7 and self.has("Goblin Barrel"):
                    self.st["opened"] = True
                    return self.barrel("OPEN", "opening: Barrel a princess tower", int(self.rng.integers(2)))
                if el >= 9.5:
                    return self._cycle("OPEN", "opening, no Barrel: ")
                return
            self.st["opened"] = True
            first = self.seen[0]
            lane = self.barrel_lane(1 - first.lane)
            if self.committed(first.lane, 10.0) >= 4 and self.has("Goblin Barrel") and lane is not None:
                return self.barrel("OPEN", f"bot opened {first.name}: Barrel the other lane", lane)
        # OPEN (inferred): a Princess into play in the first 20 s
        if t < 20 and not any(c == "Princess" for _, c, _, _ in self.mine) and self.can("Princess"):
            return self.queue("OPEN", "Princess early, in the back", self.princess_at(self.other_lane(), 31.0))
        lane = self.barrel_lane(self.other_lane())
        barrel = self.can("Goblin Barrel") and lane is not None
        # R2: > 5 into one lane in 8 s and the bot on 3 or less -> Barrel + Princess the other lane
        for k in (0, 1):
            key = ("r2", int(t // 8))
            other = self.barrel_lane(1 - k)
            if self.committed(k, 8.0) > 5 and bot_el <= 3 and key not in self.st and self.can("Goblin Barrel") \
                    and other == 1 - k:
                self.st[key] = True
                extra = [self.princess_at(other, after_s=0.3)] if self.has("Princess") else []
                return self.barrel("R2", f"all-in in the {'left' if k == 0 else 'right'} lane: Barrel + Princess "
                                         "the other lane", other, *extra)
        # R3: both small spells out -> Barrel, plus Gang / Princess at the bridge at 6+
        if arrows_out and fireball_out and barrel:
            extra = [Step(c, (LANE_X[lane], MY_BRIDGE_Y if c == "Goblin Gang" else PRINCESS_BRIDGE_Y), 0.3)
                     for c in ("Goblin Gang", "Princess") if el >= 6 and self.has(c)][:1]
            return self.barrel("R3", "Arrows and Fireball out: Barrel" + (f" + {extra[0].card}" if extra else ""),
                               lane, *extra)
        # R4: Arrows out -> Barrel (a Fireball for it is a fine trade)
        if arrows_out and barrel:
            return self.barrel("R4", "Arrows out: Barrel", lane)
        # R5: the bot holds Arrows -> bait with Princess / Gang at the bridge, hold the Barrel
        hold = barrel and not arrows_out and bot_el > 2
        bait_out = [m for m in self.my_units() if m.name in ("Princess", "Goblin Gang", "Spear Goblins") and m.y < 22]
        if hold and el >= 5 and not bait_out:
            for c, y in (("Princess", PRINCESS_BRIDGE_Y), ("Goblin Gang", MY_BRIDGE_Y)):
                if self.has(c):  # the Barrel waits for R3 / R4 / R6
                    return self.queue("R5", f"bait Arrows with {c}, hold the Barrel", Step(c, (LANE_X[lane], y)))
        # R6: the bot on 2 or less, us on 6+ -> Knight at the bridge, Barrel 1 s later
        if bot_el <= 2 and el >= 6 and barrel:
            knight = [Step("Knight", (LANE_X[lane], MY_BRIDGE_Y))] if self.has("Knight") else []
            return self.queue("R6", "bot is empty: Knight + Barrel", *knight,
                              Step("Goblin Barrel", (LANE_X[lane], BOT_TOWER_Y), 1.0 if knight else 0.0))
        # R8: a Bomber at a bot tower -> Princess at that bridge (it outranges the Bomber)
        for b in self.bot_troops():
            k = lane_of(b.x)
            if b.name == "Bomber" and self.bomber_near(k) and ("r8", b.id) not in self.st and self.can("Princess"):
                self.st[("r8", b.id)] = True
                return self.queue("R8", "Bomber at the tower: Princess at that bridge", self.princess_at(k, 19.5))
        rocket = spell_tower_damage(self.ids["Rocket"])
        # R9: Mega Minion / Bomber / Goblin Hut next to a bot princess tower -> Rocket the clump and the tower
        for u in self.bot_units():
            k = lane_of(u.x)
            if u.name in ("Mega Minion", "Bomber", "Goblin Hut") and self.tower_hp(0, k) > 0 and self.can("Rocket") \
                    and abs(u.x - LANE_X[k]) + abs(u.y - BOT_TOWER_Y) <= 3.5:
                at = ((u.x + LANE_X[k]) / 2, (u.y + BOT_TOWER_Y) / 2)
                return self.queue("R9", f"{u.name} by the tower: Rocket both", Step("Rocket", at))
        # R10: a tower in Rocket range -> Rocket it; within 2x and the last minute / overtime -> Rocket cycle it
        live = [k for k in (0, 1) if self.tower_hp(0, k) > 0]
        for k in sorted(live, key=lambda k: k != self.other_lane()):
            hp = self.tower_hp(0, k)
            if self.can("Rocket") and (hp <= rocket or (hp <= 2 * rocket and self.clock_s <= 60)):
                return self.queue("R10", f"tower at {hp:.0f}: Rocket it", Step("Rocket", (LANE_X[k], 6.0)))
        # R11: the last 1:30 and ahead on towers -> defend, Rocket cycle the same tower
        if self.clock_s <= 90 and self.crowns()[1] > self.crowns()[0] and live:
            k = self.st.setdefault("r11_lane", min(live, key=lambda k: self.tower_hp(0, k)))
            k = k if k in live else live[0]
            if self.can("Rocket"):
                return self.queue("R11", "ahead late: Rocket cycle the tower", Step("Rocket", (LANE_X[k], BOT_TOWER_Y)))
            return self._cycle("CYCLE", "ahead late, full bar: ") if el >= 9.5 else None
        # DEFAULT
        if self.clock_s <= 0:  # overtime: Rocket the weakest tower, else defend
            if self.can("Rocket") and live:
                k = min(live, key=lambda k: self.tower_hp(0, k))
                return self.queue("DEFAULT", "overtime: Rocket the weakest tower", Step("Rocket", (LANE_X[k], 6.0)))
        elif self.phase == "double":  # both lanes: Princess at each bridge, Barrel every cycle (not into Arrows)
            if (arrows_out or bot_el < 3) and el >= 5 and self.has("Princess"):
                for k in (0, 1):
                    if not [m for m in self.my_units("Princess") if lane_of(m.x) == k]:
                        return self.queue("DEFAULT", "double: Princess at the bridge", self.princess_at(k))
            if barrel and not hold and el >= 6:
                return self.barrel("DEFAULT", "double: Barrel", lane)
        elif el >= 8:  # single: a pre-placed Knight / Princess in the back corner
            for c in ("Knight", "Princess"):
                if self.has(c):
                    return self.queue("DEFAULT", f"single: {c} in the back corner", Step(c, CORNER[self.other_lane()]))
        if el >= 9.5:
            return self._cycle("CYCLE", "full bar: ")

    def _cycle(self, rule: str, why: str):
        """A full bar never sits: cycle the cheapest troop at the back (never the Inferno on nothing); with only
        Barrel / Log / Rocket / Inferno in hand, the Barrel goes in, else the Log at the bridge."""
        k = self.barrel_lane(self.other_lane())
        for card, at in (("Ice Spirit", (MID_X, 31.0)), ("Knight", CORNER[0]), ("Princess", CORNER[1]),
                         ("Goblin Gang", (MID_X, 31.0)), ("Goblin Barrel", k is not None and (LANE_X[k], 6.0)),
                         ("The Log", (MID_X, MY_BRIDGE_Y))):
            if self.has(card) and at:
                return self.queue(rule, f"{why}cycle {card}", Step(card, at))
