"""P.E.K.K.A Bridge Spam: docs/decks/pekka-bridge-spam.md as code. Rule ids match the doc's punish rules (R1-R8)."""
from __future__ import annotations

from bot77.rl.deck_bots import (BRIDGE_X, LANE_X, MY_BACK_Y, MY_BRIDGE_Y, RIVER_LO, DeckBot, Step, lane_of, register)

PK = "P.E.K.K.A."  # card_id() picks the normal one (13), not the evo (148)
HIGH_Y, LOW_Y = 20.5, 24.5
FINISH_HP = 700.0  # overtime, behind or level: Fireball a tower under this


@register
class PekkaBS(DeckBot):
    KEY = "pekka_bs"
    NAME = "P.E.K.K.A Bridge Spam"
    DECK = [PK, "Battle Ram", "Electro Wizard", "Magic Archer", "Fireball", "Bandit", "Royal Ghost", "Zap"]
    OPEN_WAIT_S = 30.0  # reactive in single elixir: never plays first before 0:30 (Cole)

    # ------------------------------------------------------------------ helpers
    def bridge_plan(self, rule: str, text: str, lane: int, *, bandit=False, ghost=False) -> bool:
        """Battle Ram at this lane's bridge, Bandit (or Ghost) behind it at the high spot if asked and in hand."""
        bx = BRIDGE_X[lane]
        steps = [Step("Battle Ram", (bx, MY_BRIDGE_Y))] if self.has("Battle Ram") else []
        for card, ok in (("Bandit", bandit), ("Royal Ghost", ghost)):
            if ok and self.has(card):
                steps.append(Step(card, (bx, HIGH_Y) if steps else (bx, MY_BRIDGE_Y), 0.3 if steps else 0.0))
                break
        return self.queue(rule, text, *steps)

    def pekkas(self) -> list:
        return self.my_units(PK)

    def hut_free_lane(self, prefer: int) -> int:
        """The lane without the bot's Goblin Hut on the path (its x side), else `prefer`."""
        huts = {lane_of(h.x) for h in self.bot_buildings("Goblin Hut")}
        return prefer if prefer not in huts or len(huts) == 2 else 1 - prefer

    def spaced_ok(self, card: str, x: float, y: float) -> bool:
        """Fireball rule: E-Wiz and Magic Archer >= 2.5 tiles apart unless the bot's Fireball is out of cycle."""
        other = {"Electro Wizard": "Magic Archer", "Magic Archer": "Electro Wizard"}.get(card)
        if other is None or self.out_of_bot_hand("Fireball"):
            return True
        return all(abs(m.x - x) + abs(m.y - y) >= 2.5 for m in self.my_units(other))

    def where(self, kind, u):
        if kind == "low":  # in front of the princess tower in the unit's lane
            live = next((v for v in self.g.get_units() if v.id == u.id and v.hp > 0), None)
            return None if live is None else (LANE_X[lane_of(live.x)], LOW_Y)
        return super().where(kind, u)

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[(PK, "low", 0)], [("Royal Ghost", "on", 0), ("Electro Wizard", "behind", 0.5)],
                  [("Bandit", "on", 0)]],
        "Battle Ram": [[("Royal Ghost", "low", 0)], [("Bandit", "low", 0)], [("Electro Wizard", "low", 0)]],
        "Mini PEKKA": [[("Electro Wizard", "on", 0), ("Royal Ghost", "on", 0.3)], [("Royal Ghost", "on", 0)],
                       [("Magic Archer", "behind", 0)]],
        "Mega Minion": [[("Electro Wizard", "behind", 0)], [("Magic Archer", "behind", 0)]],
        "Bomber": [[("Bandit", "on", 0)], [("Royal Ghost", "on", 0)]],
        "Barbarians": [[("Fireball", "on", 0)], [("Electro Wizard", "front", 0)], [(PK, "low", 0)]],
        "Barbarian": [[("Electro Wizard", "front", 0)], [("Royal Ghost", "on", 0)]],
        "Spear Goblins": [[("Zap", "on", 0)], [("Electro Wizard", "on", 0)]],
    }
    CHEAP = ["Royal Ghost", "Bandit", "Electro Wizard", "Magic Archer"]

    def answer_options(self, u):
        opts = super().answer_options(u)
        r2 = self.st.get("r2")
        if u.name == "Mini PEKKA" and r2 and r2[0] == lane_of(u.x) and self.t - r2[1] < 10:
            opts = [[(PK, "low", 0)]] + opts  # R2: P.E.K.K.A low on a Mini P.E.K.K.A after an all-in
        if u.name == "Battle Ram" or (u.name == "Giant" and self.elixir < self.cost[PK] - 2):
            opts = [o for o in opts if o[0][0] != PK]  # never P.E.K.K.A on a lone Ram; no long wait on a Giant
        for o in list(opts):  # Fireball rule: keep E-Wiz and Magic Archer apart
            spot = self.where(o[0][1], u)
            if spot and not self.spaced_ok(o[0][0], *spot):
                opts.remove(o)
        return opts

    def threats(self):  # the bot's spells show up as units on the board: nothing to answer
        return [u for u in super().threats() if u.name not in ("Fireball", "Arrows")]

    def defend(self):
        # R8: a bot Mini P.E.K.K.A or Mega Minion on the P.E.K.K.A -> Zap/E-Wiz reset, Ghost or Magic Archer to kill
        for pk in self.pekkas():
            for u in self.bot_troops():
                if u.name not in ("Mini PEKKA", "Mega Minion") or ("r8", u.id) in self.st:
                    continue
                if u.target_id == pk.id or abs(u.x - pk.x) + abs(u.y - pk.y) < 3.0:
                    self.st[("r8", u.id)] = self.answered_at[u.id] = self.t
                    reset = next((c for c in ("Zap", "Electro Wizard") if self.has(c)), None)
                    kill = next((c for c in (("Royal Ghost",) if u.name == "Mini PEKKA" else ())
                                 + ("Magic Archer",) if self.has(c) and c != reset), None)
                    steps = [Step(c, (lambda b, u=u, k=k: b.where(k, u)), a) for c, k, a in
                             ((reset, "on", 0.0), (kill, "behind" if kill == "Magic Archer" else "on", 0.3)) if c]
                    if steps:
                        self.queue("R8", f"{u.name} on the P.E.K.K.A: " + " + ".join(s.card for s in steps),
                                   *steps, urgent=True)
                        return
        # R7: a bot Giant or Battle Ram walking to the bridge -> E-Wiz high at that bridge to block and reset
        for u in self.bot_troops():
            if u.name in ("Giant", "Battle Ram") and 9.0 <= u.y < RIVER_LO and ("r7", u.id) not in self.st:
                if self.has("Electro Wizard") and self.elixir >= 3:
                    self.st[("r7", u.id)] = True
                    self.answered_at[u.id] = self.t
                    self.queue("R7", f"{u.name} to the bridge: E-Wiz high to block",
                               Step("Electro Wizard", (BRIDGE_X[lane_of(u.x)], HIGH_Y)), urgent=True)
                    return
        super().defend()

    # ------------------------------------------------------------------ the punish rules and the default
    def on_bot_play(self, s):
        # R1: a 5+ troop at the bot's back -> Battle Ram at the opposite bridge within 1 s, Bandit behind if rich
        if s.cost >= 5 and s.y <= 8.0 and s.name not in ("Fireball", "Arrows") and self.elixir >= 4:
            ram = self.has("Battle Ram")  # no Ram in hand: Bandit alone (the game plan's Ram/Bandit/Ghost punish)
            if (ram or self.has("Bandit")) and not (self.phase == "triple" and self.leading()):
                self.bridge_plan("R1", f"{s.name} in the back: {'Ram' if ram else 'Bandit'} the opposite bridge",
                                 1 - s.lane, bandit=not ram or (self.elixir >= 7 and self.bot_elixir <= 4))
        # R5: the bot spent 6+ in 6 s while our P.E.K.K.A pushes (at the bridge or past) -> Ram + Bandit the other bridge
        push = [p for p in self.pekkas() if p.y <= HIGH_Y]
        if push and sum(x.cost for x in self.seen if x.t >= self.t - 6.0) >= 6 and "r5" not in self.st:
            self.st["r5"] = (1 - lane_of(push[0].x), self.t)

    def decide(self):
        t, el, bot_el = self.t, self.elixir, self.bot_elixir
        overtime_ahead = self.phase == "triple" and self.leading()
        # OPEN (0:30, the bot hasn't played): a cheap card at the back, Bandit or Ghost
        if not self.seen and not self.mine:
            card = next((c for c in ("Bandit", "Royal Ghost", "Zap", "Electro Wizard") if self.has(c)), None)
            if card:
                self.queue("OPEN", f"0:30, no bot play: {card} at the back",
                           Step(card, (LANE_X[int(self.rng.integers(2))], MY_BACK_Y - 0.5)))
            return
        # R5 follow-through
        r5 = self.st.get("r5")
        if r5 and r5 != "done" and not overtime_ahead:
            if t - r5[1] > 12:
                self.st.pop("r5")
            elif el >= 7 and (self.has("Battle Ram") or self.has("Bandit")):
                self.st["r5"] = "done"  # once per P.E.K.K.A push (cleared when it dies)
                return self.bridge_plan("R5", "bot spent 6+ on the P.E.K.K.A: Ram + Bandit the other bridge",
                                        r5[0], bandit=True)
        if not self.pekkas():
            self.st.pop("r5", None)
        # R2: > 5 in one lane in 4 s and the bot on 3 or less -> defend cheap, then Ram that lane once it's clear
        for lane in (0, 1):
            if self.committed(lane, 4.0) > 5 and bot_el <= 3 and not self.st.get("r2"):
                self.st["r2"] = (lane, t)
                self.events.append({"t": round(t, 1), "rule": "R2", "text": "all-in: defend cheap, Ram after"})
        r2 = self.st.get("r2")
        if r2:
            alive = [u for u in self.bot_troops() if lane_of(u.x) == r2[0] and u.y >= 9.0]
            if t - r2[1] > 20:
                self.st.pop("r2")
            elif not alive and t - r2[1] > 2 and el >= 4 and not overtime_ahead and self.has("Battle Ram"):
                self.st.pop("r2")
                self.bridge_plan("R2", "bot's push is dead: Ram that bridge", r2[0])
                if el >= 7 and self.has("Bandit"):
                    self.queue("R2", "and Bandit the other bridge", Step("Bandit", (BRIDGE_X[1 - r2[0]], MY_BRIDGE_Y)))
                return
        # R3: the bot is on 2 or less, nothing of its across -> Ram the weaker tower's bridge
        if bot_el <= 2 and not self.crossed() and el >= 4 and self.has("Battle Ram"):
            return self.bridge_plan("R3", "bot is empty: Ram the weaker tower", self.weaker_bot_lane(), bandit=el >= 7)
        if overtime_ahead:  # defend and bridge-block only
            return self._cycle()
        # R4: Mini P.E.K.K.A out of cycle, Mega Minion out or across -> P.E.K.K.A at the bridge in the Hut-free lane
        mm_far = lambda ln: self.out_of_bot_hand("Mega Minion") and not any(  # out of cycle, and if alive, away
            u.name == "Mega Minion" and lane_of(u.x) == ln for u in self.bot_troops())
        lane = self.hut_free_lane(self.weaker_bot_lane())
        if self.out_of_bot_hand("Mini PEKKA") and mm_far(lane) and el >= 7 and self.has(PK) and not self.pekkas():
            self.st["r4"] = (lane, t)
            return self.queue("R4", "Mini P.E.K.K.A out of cycle: P.E.K.K.A at the bridge",
                              Step(PK, (BRIDGE_X[lane], MY_BRIDGE_Y)))
        r4 = self.st.get("r4")
        if r4 and self.pekkas() and el >= 4 and t - r4[1] < 15:
            self.st.pop("r4")
            card = next((c for c in ("Battle Ram", "Bandit") if self.has(c)), None)
            if card:
                pk = self.pekkas()[0]
                return self.queue("R4", f"{card} in front of the P.E.K.K.A", Step(card, (pk.x, max(pk.y - 1.0, 18.5))))
        # R6: Fireball out of cycle and a P.E.K.K.A push on the field -> E-Wiz + Magic Archer behind it together
        pk = next((p for p in self.pekkas() if p.y <= 22.0), None)
        if pk and self.out_of_bot_hand("Fireball") and "r6" not in self.st:
            near = lambda c: any(abs(m.x - pk.x) + abs(m.y - pk.y) < 5 for m in self.my_units(c))
            want = [c for c in ("Electro Wizard", "Magic Archer") if not near(c)]
            if want and all(self.has(c) for c in want) and el >= 4 * len(want) - 1.0:  # both, or the missing one
                self.st["r6"] = True
                return self.queue("R6", "Fireball out of cycle: " + " + ".join(want) + " behind the P.E.K.K.A",
                                  *[Step(c, (pk.x + 0.5 * k, pk.y + 2.0 + 0.5 * k), 0.2 * k) for k, c in enumerate(want)])
        # DEFAULT
        if self.phase == "triple" and el >= 4 and self.can("Fireball"):  # behind or level: finish a tower
            for ln in (0, 1):
                if 0 < self.tower_hp(0, ln) < FINISH_HP:
                    return self.queue("DEFAULT", "overtime: Fireball the low tower", Step("Fireball", (LANE_X[ln], 6.0)))
        if pk and pk.y <= 20.5 and "sup" not in self.st and el >= 4:  # support the P.E.K.K.A at the bridge
            for card in ("Electro Wizard", "Magic Archer"):
                spot = (pk.x, pk.y + 2.5)
                if self.has(card) and self.spaced_ok(card, *spot):
                    self.st["sup"] = True
                    return self.queue("DEFAULT", f"{card} behind the P.E.K.K.A", Step(card, spot))
        if not self.pekkas():
            self.st.pop("sup", None)
            self.st.pop("r6", None)
        if self.phase == "single":
            if el >= 9 and self.has(PK) and not self.pekkas():
                tank = next((s for s in reversed(self.seen) if s.name in ("Giant", "Battle Ram")), None)
                ln = 1 - tank.lane if tank else self.weaker_bot_lane()
                return self.queue("DEFAULT", "single: P.E.K.K.A at the back", Step(PK, (LANE_X[ln], MY_BACK_Y)))
        else:  # double/triple: punish every 5+ commitment in the other lane
            for ln in (0, 1):
                key = ("dbl", int(t // 6))
                if self.committed(ln, 6.0) >= 5 and self.committed(1 - ln, 6.0) == 0 and key not in self.st:
                    if el >= 6 and self.has("Battle Ram"):
                        self.st[key] = True
                        return self.bridge_plan("DEFAULT", f"{self.phase}: Ram + Bandit/Ghost the other lane", 1 - ln,
                                                bandit=True, ghost=True)
        return self._cycle()

    def _cycle(self):
        """Full bar with nothing better: cycle a cheap card at the back toward P.E.K.K.A / E-Wiz."""
        if self.elixir < 9.5:
            return
        for card in ("Royal Ghost", "Bandit", "Zap", "Electro Wizard", "Magic Archer", PK):
            if self.has(card):
                at = (LANE_X[self.weaker_bot_lane()], 6.0) if card == "Zap" else (LANE_X[int(self.rng.integers(2))], MY_BACK_Y - 0.5)
                return self.queue("CYCLE", f"full bar: cycle {card} at the back", Step(card, at))
