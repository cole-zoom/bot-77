"""Giant Double Prince: docs/decks/giant-double-prince.md as code. Rule ids match the doc's punish rules (R1-R11)."""
from __future__ import annotations

from bot77.rl.deck_bots import (BRIDGE_X, LANE_X, MID_X, MY_BACK_Y, MY_BRIDGE_Y, RIVER_HI, RIVER_LO, WAIT_S,
                                DeckBot, Step, lane_of, register)

BACK_Y = 32.0
PRINCES = ("Prince", "Dark Prince")
SPELLS = ("Fireball", "Arrows")
AIR_HITTERS = ("Minions", "Baby Dragon", "Vines", "Zap")


@register
class GiantDoublePrince(DeckBot):
    KEY = "gdp"
    NAME = "Giant Double Prince"
    DECK = ["Giant", "Prince", "Dark Prince", "Baby Dragon", "Minions", "Goblin Gang", "Vines", "Zap"]
    OPEN_WAIT_S = 30.0  # reactive in single elixir: never plays first before 0:30 (Cole)

    # ------------------------------------------------------------------ helpers
    def tick(self):
        # bookkeeping every tick, busy or not: how long the bot has been low (R7), own units' full HP (R6)
        if self.bot_elixir <= 2:
            self.st.setdefault("low_since", self.t)
        else:
            self.st.pop("low_since", None)
        hp0 = self.st.setdefault("hp0", {})
        for u in self.my_units():
            hp0.setdefault(u.id, u.hp)
        return super().tick()

    def soon(self, card: str) -> bool:
        """In hand and affordable before a queued play of it expires (WAIT_S of elixir income)."""
        rate = {"single": 1 / 2.8, "double": 2 / 2.8, "triple": 3 / 2.8}[self.phase]
        return self.has(card) and self.elixir + rate * (WAIT_S - 0.5) >= self.cost[card]

    def push_lane(self, lane: int) -> int:
        """Goblin Hut: a hut off-centre pulls the Giant and Princes in its lane, so push the other one."""
        huts = [h for h in self.bot_buildings("Goblin Hut") if abs(h.x - MID_X) > 2]
        return 1 - lane_of(huts[0].x) if huts else lane

    def mpekka_in_hand(self) -> bool:
        """R9's test: Mini P.E.K.K.A not among the bot's last 3 plays (the doc's window)."""
        return not any(s.name == "Mini PEKKA" for s in self.seen[-3:])

    def prince_plan(self, rule: str, text: str, lane: int, *, giant=False, support=False) -> bool:
        """Prince at this lane's bridge; a Giant in front of it, or Goblin Gang / Dark Prince behind it, if asked."""
        bx = BRIDGE_X[lane]
        steps = []
        if giant and self.has("Giant"):
            steps.append(Step("Giant", (bx, MY_BRIDGE_Y)))
        steps.append(Step("Prince", (bx, MY_BRIDGE_Y + (1.5 if steps else 0.0)), 1.0 if steps else 0.0))
        if support:
            behind = next((c for c in ("Goblin Gang", "Dark Prince") if self.has(c)), None)
            if behind:
                steps.append(Step(behind, (LANE_X[lane], 21.0), 0.8))
        return self.queue(rule, text, *steps)

    def support_plan(self) -> bool:
        """R11: one support card behind our walking Giant. The bot's answer near it decides (Cole, 2026-11-06): a Goblin
        Hut -> splash (Dark Prince, Baby Dragon); Mini P.E.K.K.A -> damage (Prince); Mega Minion or anything flying ->
        Baby Dragon / Minions. Nothing shown by the time the Giant reaches the bridge -> Baby Dragon, else Dark Prince.
        [inferred: the no-answer default]"""
        done = self.st.setdefault("supported", set())
        g = next((u for u in self.my_units("Giant") if u.id not in done and u.y >= RIVER_LO), None)
        if g is None:
            return False
        lane = lane_of(g.x)
        near = [u for u in self.bot_troops() if abs(u.x - g.x) + abs(u.y - g.y) < 10]
        if any(lane_of(h.x) == lane or abs(h.x - MID_X) <= 2 for h in self.bot_buildings("Goblin Hut")):
            why, cards = "Goblin Hut", ("Dark Prince", "Baby Dragon")
        elif any(u.name == "Mini PEKKA" for u in near):
            why, cards = "Mini P.E.K.K.A", ("Prince", "Dark Prince")
        elif any(u.flying for u in near):
            why, cards = "air", ("Baby Dragon", "Minions")
        elif g.y <= MY_BRIDGE_Y + 1.0:
            why, cards = "no answer yet", ("Baby Dragon", "Dark Prince", "Prince")
        else:
            return False
        card = next((c for c in cards if self.soon(c)), None)
        if card is None:
            return False
        done.add(g.id)
        behind = lambda b, gid=g.id: next(((u.x, max(u.y + 1.5, MY_BRIDGE_Y)) for u in b.my_units("Giant")
                                            if u.id == gid), None)
        return self.queue("R11", f"Giant's support vs {why}: {card} behind it", Step(card, behind))

    def survivors(self) -> list:
        """R6: own Princes at >= 50% HP back on our side of the river."""
        hp0 = self.st.get("hp0", {})
        return [u for u in self.my_units() if u.name in PRINCES and u.y >= RIVER_HI
                and u.hp >= 0.5 * hp0.get(u.id, u.hp)]

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[("Prince", "front", 0), ("Goblin Gang", "front", 0.5)],
                  [("Goblin Gang", "front", 0), ("Vines", "on", 1.0)],
                  [("Dark Prince", "front", 0)], [("Minions", "behind", 0)]],
        "Battle Ram": [[("Dark Prince", "on", 0)], [("Goblin Gang", "on", 0)]],
        "Mini PEKKA": [[("Goblin Gang", "on", 0)], [("Vines", "on", 0)], [("Minions", "behind", 0)],
                       [("Baby Dragon", "behind", 0)]],
        "Mega Minion": [[("Minions", "on", 0)], [("Baby Dragon", "behind", 0)]],
        "Bomber": [[("Dark Prince", "on", 0)], [("Prince", "on", 0)], [("Minions", "on", 0)]],
        "Barbarians": [[("Dark Prince", "on", 0)], [("Baby Dragon", "behind", 0)]],
        "Barbarian": [[("Dark Prince", "on", 0)], [("Baby Dragon", "behind", 0)]],
        "Spear Goblins": [[("Zap", "on", 0)], [("Dark Prince", "on", 0)], [("Baby Dragon", "behind", 0)]],
    }
    CHEAP = ["Goblin Gang", "Vines", "Minions", "Dark Prince", "Baby Dragon"]  # never a lone Prince as a fallback

    def answer_options(self, u):
        opts = super().answer_options(u)
        if u.flying:  # air: every air-hitter before the (ground) cheap fallback
            opts = opts + [[(c, "on", 0)] for c in AIR_HITTERS if all(o[0][0] != c for o in opts)]
        r1 = self.st.get("r1")
        if r1 and r1[0] == lane_of(u.x) and self.t - r1[1] < 15:  # R1: the all-in lane gets the cheapest stop
            opts = [[("Goblin Gang", "front", 0), ("Vines", "on", 1.5)], [("Vines", "on", 0)]] + opts
        if u.name == "Giant" and self.t - self.st.get("r2_at", -99) < 25 and not self.my_units("Prince"):
            opts = [o for o in opts if o[0][0] != "Prince"]  # R2: our Prince died on offence -> Goblin Gang + Vines
        near = lambda name: any(b.name == name and abs(b.x - u.x) + abs(b.y - u.y) < 5 for b in self.bot_troops())
        if near("Bomber"):  # never Goblin Gang near a Bomber
            opts = [o for o in opts if o[0][0] != "Goblin Gang"]
        if u.name == "Mini PEKKA":  # never a lone uncharged Prince into it
            opts = [o for o in opts if o != [("Prince", "on", 0)]]
        return [o for o in opts if self.soon(o[0][0])]  # an answer that can't be paid in time is no answer

    def threats(self):
        return [u for u in super().threats() if u.name not in SPELLS]  # a spell landing isn't a troop to answer

    def defend(self):
        for u in self.threats():
            if u.id in self.answered_at:
                continue
            # R5: Battle Ram -> Dark Prince in its path (Goblin Gang if no Dark Prince)
            if u.name == "Battle Ram":
                card = next((c for c in ("Dark Prince", "Goblin Gang") if self.has(c)), None)
                if card:
                    self.answered_at[u.id] = self.t
                    self.queue("R5", f"Battle Ram: {card} in its path", Step(card, lambda b, u=u: b.where("on", u)),
                               urgent=True)
                    return
            # R8: Giant + Mega Minion -> Minions on the Mega Minion now, Prince on the Giant once it locks the tower
            if u.name == "Mega Minion" and self.has("Minions"):
                giant = next((g for g in self.bot_troops() if g.name == "Giant"
                              and abs(g.x - u.x) + abs(g.y - u.y) < 8), None)
                if giant:
                    self.answered_at[u.id] = self.answered_at[giant.id] = self.t
                    steps = [Step("Minions", lambda b, u=u: b.where("on", u))]
                    if self.has("Prince"):
                        steps.append(Step("Prince", (LANE_X[lane_of(giant.x)], 24.0), 2.0))
                    self.queue("R8", "Giant + Mega Minion: " + " + ".join(s.card for s in steps), *steps, urgent=True)
                    return
        super().defend()
        if any(p.rule == "DEF" for p in self.plans):
            self.st["def_at"] = self.t

    # ------------------------------------------------------------------ the punish rules and the default
    def on_bot_play(self, s):
        heavy = s.name not in ("Fireball", "Arrows")
        # R1: >= 7 in one lane and the bot on 3 or less -> cheap stop there (answer_options), Prince the other bridge
        if self.committed(s.lane) >= 7 and self.bot_elixir <= 3 and heavy \
                and self.t - self.st.get("r1", (0, -99))[1] > 10:
            self.st["r1"] = (s.lane, self.t)  # the cheap stop applies even if we can't counter
            if self.soon("Prince"):
                return self.prince_plan("R1", "all-in in one lane: Prince the other bridge", 1 - s.lane,
                                        support=self.elixir >= 8)
            if self.soon("Dark Prince"):  # no Prince to hand: the Dark Prince goes instead
                return self.queue("R1", "all-in in one lane: Dark Prince the other bridge",
                                  Step("Dark Prince", (BRIDGE_X[1 - s.lane], MY_BRIDGE_Y)))
        # R2: Giant in the back in single elixir and we have 5 -> Prince the opposite bridge, Goblin Gang behind
        if s.name == "Giant" and s.y <= 4 and self.phase == "single" and self.elixir >= 5 and self.has("Prince"):
            self.st["r2_at"] = self.t
            return self.prince_plan("R2", "Giant in the back: Prince the opposite bridge", 1 - s.lane, support=True)

    def decide(self):
        t, el, bot_el = self.t, self.elixir, self.bot_elixir
        last = self.last_bot_play()
        # OPEN (0:30, the bot hasn't played): Baby Dragon in the back, else Goblin Gang / Dark Prince at a bridge
        if not self.seen and not self.mine:
            lane = int(self.rng.integers(2))
            for card, at in (("Baby Dragon", (7.5 if lane == 0 else 9.5, BACK_Y)),
                             ("Goblin Gang", (BRIDGE_X[lane], MY_BRIDGE_Y)),
                             ("Dark Prince", (BRIDGE_X[lane], MY_BRIDGE_Y))):
                if self.has(card):
                    return self.queue("OPEN", f"0:30, no info: {card}", Step(card, at))
            return
        # R11 (Cole, 2026-11-06): our Giant is walking in -> support behind it, picked by what the bot answers with
        sup = self.support_plan()
        if sup:
            return sup
        # R3: Mini P.E.K.K.A just played, our Giant or Prince alive -> Prince the other lane (Giant in front at 9+)
        mp = next((s for s in reversed(self.seen[-4:]) if s.name == "Mini PEKKA"), None)
        if mp and ("r3", mp.t) not in self.st and (self.my_units("Giant") or self.my_units("Prince")) \
                and self.has("Prince") and el >= 5:
            self.st[("r3", mp.t)] = True
            return self.prince_plan("R3", "Mini P.E.K.K.A is out: Prince the other lane", 1 - mp.lane, giant=el >= 9)
        # R4: Arrows just played and we have 8 -> Giant (bridge; back in double) + Minions behind it 2 s later
        ar = next((s for s in reversed(self.seen[-4:]) if s.name == "Arrows"), None)
        if ar and ("r4", ar.t) not in self.st and el >= 8 and self.has("Giant", "Minions"):
            self.st[("r4", ar.t)] = True
            lane = self.push_lane(self.weaker_bot_lane())
            gy = MY_BRIDGE_Y if self.phase == "single" else BACK_Y
            return self.queue("R4", "Arrows is out: Giant + Minions", Step("Giant", (BRIDGE_X[lane], gy)),
                              Step("Minions", (LANE_X[lane], gy + 1.5), 2.0))
        # R6: a defence just ended with a Prince alive -> Giant a tile in front of the survivors
        surv = self.survivors()
        if surv and not self.threats() and t - self.st.get("def_at", -99) < 20 and el >= 5 and self.has("Giant") \
                and (bot_el <= 4 or not self.mpekka_in_hand()) and t - self.st.get("r6_at", -99) > 15:
            self.st["r6_at"] = t
            u = max(surv, key=lambda v: -v.y)
            return self.queue("R6", f"counter-push: Giant in front of the {u.name}", Step("Giant", (u.x, u.y - 1.0)))
        # R7 (rare, Cole 2026-11-06): the bot has been on 1 or less for 2 s AND has no Prince answer up (Mini P.E.K.K.A
        # just played, no Goblin Hut in the lane) -> Prince at the weaker tower's bridge. Never a lone Dark Prince.
        lane = self.push_lane(self.weaker_bot_lane())
        if self.bot_elixir <= 1 and t - self.st.get("low_since", t) >= 2.0 and el >= 5 and self.has("Prince") \
                and not self.crossed() and self.out_of_bot_hand("Mini PEKKA") \
                and not any(lane_of(h.x) == lane for h in self.bot_buildings("Goblin Hut")):
            self.st.pop("low_since", None)
            return self.queue("R7", "bot is empty, no Prince answer up: Prince at the bridge",
                              Step("Prince", (BRIDGE_X[lane], MY_BRIDGE_Y)))
        # R10 (Cole, 2026-11-06): no push on, nothing to defend, 8 elixir (7 in double) -> Giant behind the king on
        # the weaker tower's side; R11 adds the support once the bot shows its answer
        if not self.my_units("Giant") and not self.threats() and self.has("Giant") \
                and el >= (8 if self.phase == "single" else 7):
            self.st["giant_at"] = t
            return self.queue("R10", "Giant in the back", Step("Giant", (7.5 if lane == 0 else 9.5, MY_BACK_Y)))
        other = 1 - last.lane if last else int(self.rng.integers(2))
        # DEFAULT
        if self.phase == "double" or (self.phase == "triple" and not self.leading()):
            if el >= 9.5 and self.has("Giant", "Prince"):
                if self.mpekka_in_hand() and bot_el >= 4:  # R9: no Giant + Prince while the tank killer is up
                    return self._cycle("R9", "Mini P.E.K.K.A may be in hand: cycle instead")
                lane = self.push_lane(self.weaker_bot_lane())
                steps = [Step("Giant", (LANE_X[lane], BACK_Y)), Step("Prince", (BRIDGE_X[lane], MY_BRIDGE_Y), 4.0)]
                if self.has("Baby Dragon"):
                    steps.append(Step("Baby Dragon", (BRIDGE_X[lane], MY_BRIDGE_Y + 1.5), 2.0))
                return self.queue("DEFAULT", f"{self.phase}: Giant in the back, Prince behind it", *steps)
            if el >= 9 and self.has("Prince", "Dark Prince"):
                lane = self.push_lane(other)
                return self.queue("DEFAULT", f"{self.phase}: split Prince / Dark Prince",
                                  Step("Prince", (BRIDGE_X[lane], MY_BRIDGE_Y)),
                                  Step("Dark Prince", (BRIDGE_X[1 - lane], MY_BRIDGE_Y), 0.5))
        if el >= 9.5:  # single elixir, overtime when ahead, or nothing above fits: cycle, never sit at 10
            return self._cycle("DEFAULT" if self.phase == "single" else "CYCLE", "full bar: cycle")

    def _cycle(self, rule: str, text: str):
        """Cheap cycle (Baby Dragon in the back, Goblin Gang, Zap on a tower); any card in the back if nothing fits."""
        lane = int(self.rng.integers(2))
        for card, at in (("Baby Dragon", (7.5 if lane == 0 else 9.5, BACK_Y)), ("Goblin Gang", (MID_X, 31.0)),
                         ("Zap", (LANE_X[self.weaker_bot_lane()], 6.0)), ("Minions", (MID_X, 31.0)),
                         ("Vines", (LANE_X[self.weaker_bot_lane()], 6.0)), ("Dark Prince", (LANE_X[lane], BACK_Y))):
            if self.has(card):
                return self.queue(rule, f"{text}: {card}", Step(card, at))
        card = next((c for c in self.hand() if c in self.DECK), None)
        if card:
            at = (LANE_X[lane], BACK_Y)
            return self.queue("CYCLE", f"full bar: {card} at the back", Step(card, at))
