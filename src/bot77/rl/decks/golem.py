"""Golem Beatdown (Golem Hog hybrid): docs/decks/golem-beatdown.md as code. Rule ids match the doc's punish rules."""
from __future__ import annotations

from bot77.rl.deck_bots import (BRIDGE_X, LANE_X, MID_X, MY_BACK_Y, MY_BRIDGE_Y, DeckBot, Step, card_id,
                                lane_of, register)

KING_BACK = (MID_X, 33.0)
PATH_Y = 25.0  # R3/R4: in a ground win condition's path, in front of the princess tower
OVERTIME_TICK = 1800
_HOG_HIT: list[float] = []


def hog_hit() -> float:
    """One Hog Rider hit on a princess tower, measured by sending one at it (no copy of the engine's numbers)."""
    if not _HOG_HIT:
        import clash_royale_env as cre
        g = cre.ClashRoyaleEnv([0, 1, 2, 3, 5, 6, 7, 9], [0, 1, 2, 3, 5, 6, 7, 9], 3600,
                               cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
        g.seed(0)
        hp0 = g.get_tower_hp(0, 1)
        g.inject(card_id("Hog Rider"), 3.0, 9.0, 1, -1.0, -1)
        while g.get_tower_hp(0, 1) >= hp0 and g.get_current_tick() < 100:
            g.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
        _HOG_HIT.append(float(hp0 - g.get_tower_hp(0, 1)))
    return _HOG_HIT[0]


@register
class Golem(DeckBot):
    KEY = "golem"
    NAME = "Golem Beatdown"
    DECK = ["Golem", "Barbarians", "Hog Rider", "Valkyrie", "Firecracker", "Dart Goblin", "Skeletons", "Heal Spirit"]
    OPEN_WAIT_S = 30.0  # single elixir is reactive: it never plays first before 0:30 (Cole)

    # ------------------------------------------------------------------ helpers
    def hog_lane(self) -> int:
        """The lane away from a Goblin Hut, else the lane the bot didn't play in last."""
        huts = self.bot_buildings("Goblin Hut")
        if huts:
            return 1 - lane_of(huts[0].x)
        last = self.last_bot_play()
        return 1 - last.lane if last else int(self.rng.integers(2))

    def hog_plan(self, rule: str, text: str, lane: int, *, cheap=True) -> bool:
        """Hog at this lane's bridge, Heal Spirit or Skeletons right behind it if asked and in hand."""
        bx = BRIDGE_X[lane]
        steps = [Step("Hog Rider", (bx, MY_BRIDGE_Y))]
        one = next((c for c in ("Heal Spirit", "Skeletons") if self.has(c)), None) if cheap else None
        if one:
            steps.append(Step(one, (bx, MY_BRIDGE_Y + 1.0), 0.2))
        return self.queue(rule, text, *steps)

    def golem(self):
        """My Golem on the board (the first one), or None."""
        return next(iter(self.my_units("Golem")), None)

    def golem_recent(self, s: float = 6.0):
        """R4: when and where (lane) it spent 8 on a Golem less than s ago, or None."""
        g = next((m for m in reversed(self.mine) if m[1] == "Golem"), None)
        return lane_of(g[2]) if g and self.t - g[0] < s else None

    def where(self, kind, u):
        if kind == "path":  # R3: in the unit's path in front of the princess tower
            live = next((v for v in self.g.get_units() if v.id == u.id and v.hp > 0), None)
            return None if live is None else (LANE_X[lane_of(live.x)], PATH_Y)
        return super().where(kind, u)

    # ------------------------------------------------------------------ defence (urgent): the doc's "Against our deck"
    ANSWERS = {
        "Giant": [[("Barbarians", "path", 0)], [("Valkyrie", "path", 0), ("Skeletons", "on", 1.0)],
                  [("Skeletons", "on", 0)]],
        "Battle Ram": [[("Barbarians", "path", 0)], [("Skeletons", "on", 0)], [("Valkyrie", "path", 0)]],
        "Mini PEKKA": [[("Skeletons", "on", 0)], [("Barbarians", "on", 0)], [("Valkyrie", "on", 0)]],
        "Mega Minion": [[("Dart Goblin", "behind", 0)], [("Firecracker", "behind", 0)]],
        "Bomber": [[("Dart Goblin", "behind", 0)], [("Valkyrie", "on", 0)]],
        **{n: [[("Valkyrie", "on", 0)], [("Firecracker", "behind", 0)], [("Dart Goblin", "behind", 0)]]
           for n in ("Barbarians", "Barbarian", "Spear Goblins", "Spear Goblin")},  # swarms: splash
    }
    CHEAP = ["Skeletons", "Valkyrie", "Firecracker", "Dart Goblin", "Barbarians"]

    def answer_options(self, u):
        opts = super().answer_options(u)
        if u.name in ("Giant", "Battle Ram") and any(v.name in ("Bomber", "Mega Minion") and abs(v.x - u.x) < 4
                                                     for v in self.crossed()):
            # R3: the bot added Bomber / Mega Minion: Firecracker behind the tower too
            opts = [o + [("Firecracker", "behind", 0.3)] for o in opts]
        if self.maybe_in_bot_hand("Fireball"):  # spacing: no second ranged troop next to a live one
            ranged = [m for m in self.my_units() if m.name in ("Firecracker", "Dart Goblin")]
            near = lambda c: any(abs(m.x - LANE_X[lane_of(u.x)]) + abs(m.y - 29.5) < 4 for m in ranged)
            opts = [o for o in opts if o[0][0] not in ("Firecracker", "Dart Goblin") or not near(o[0][0])] or opts
        return opts

    def threats(self):
        """The framework's threats, less the bot's spells (they show up as units where they land)."""
        return [u for u in super().threats() if u.name not in ("Fireball", "Arrows")]

    def defend(self):
        for u in self.threats():
            if u.id in self.answered_at:
                continue
            lane = lane_of(u.x)
            # R1: >= 5 into one lane in 5 s and the bot is on <= 3: defend there, then Hog + 1 at the other bridge
            mark = self.st.get("r1")  # (lane, when): set by on_bot_play when the bot went all-in
            if mark and mark[0] == lane and self.t - mark[1] < 12:
                d = next((c for c in ("Barbarians", "Valkyrie") if self.has(c)), None)
                if d and self.has("Hog Rider"):
                    self.st["r1"] = None
                    self.answered_at[u.id] = self.t
                    steps = [Step(d, (LANE_X[lane], PATH_Y))]
                    if u.name == "Giant" and self.has("Skeletons"):
                        steps.append(Step("Skeletons", lambda b, u=u: b.where("on", u), 0.5))
                    react = float(self.rng.uniform(0.5, 1.0))  # 0.5-1 s after the defender lands
                    steps.append(Step("Hog Rider", (BRIDGE_X[1 - lane], MY_BRIDGE_Y), react))
                    if self.has("Heal Spirit"):
                        steps.append(Step("Heal Spirit", (BRIDGE_X[1 - lane], MY_BRIDGE_Y + 1.0), 0.2))
                    return self.queue("R1", f"all-in in one lane: {d} there, Hog the other bridge", *steps,
                                      urgent=True)
            # R4: Golem just went down in the back and the bot attacks the other lane: cheapest ground answer
            mark = self.st.get("r4")  # (lane, when): set by on_bot_play, the bot's play in the other lane
            if mark and mark[0] == lane and self.t - mark[1] < 15:
                self.st["r4"] = None
                self.answered_at[u.id] = self.t
                if self.tower_hp(1, lane) < 0.3 * self.g.get_tower_max_hp(1, 1 + lane) and self.elixir < 6:
                    self.events.append({"t": round(self.t, 1), "rule": "R4", "text": "tower low: let it go"})
                    return
                g = None if u.flying else next((c for c in ("Valkyrie", "Skeletons", "Barbarians") if self.has(c)),
                                               None)
                steps = [Step(g, (LANE_X[lane], PATH_Y))] if g else []
                if self.has("Firecracker"):
                    steps.append(Step("Firecracker", (LANE_X[lane], 29.0), 0.3))
                if steps:
                    return self.queue("R4", f"Golem in the back, {u.name} in the other lane: "
                                            + " + ".join(s.card for s in steps), *steps, urgent=True)
            # R3: a Battle Ram or Giant across: Barbarians in its path (answer_options adds the Firecracker)
            if u.name in ("Battle Ram", "Giant") and self.has("Barbarians"):
                self.answered_at[u.id] = self.t
                steps = [Step(c, (lambda b, k=k, u=u: b.where(k, u)), a) for c, k, a in self.answer_options(u)[0]
                         if self.has(c)]
                return self.queue("R3", f"{u.name} across: " + " + ".join(s.card for s in steps), *steps,
                                  urgent=True)
        super().defend()

    # ------------------------------------------------------------------ the punish rules and the default
    def on_bot_play(self, s):
        # R1: >= 5 into one lane in 5 s and the bot is on <= 3 -> mark the lane; defend() answers when it crosses
        if s.name not in ("Fireball", "Arrows") and self.committed(s.lane, 5.0) >= 5 and self.bot_elixir <= 3:
            self.st["r1"] = (s.lane, self.t)
        # R4: a play in the other lane less than 6 s after its Golem went down in the back -> mark that lane
        gl = self.golem_recent()
        if gl is not None and s.lane != gl and s.name not in ("Fireball", "Arrows"):
            self.st["r4"] = (s.lane, self.t)
        # R6 (second half): the bot commits a defender to the Golem's lane -> Hog at the other bridge
        g = self.golem()
        if g is not None and g.y <= 22 and s.lane == lane_of(g.x) and s.name not in ("Fireball", "Arrows"):
            if self.has("Hog Rider") and ("r6h", g.id) not in self.st:
                self.st[("r6h", g.id)] = True
                self.hog_plan("R6", f"{s.name} on the Golem: Hog the other bridge", 1 - lane_of(g.x), cheap=False)

    def decide(self):
        t, el, bot_el, phase = self.t, self.elixir, self.bot_elixir, self.phase
        for b in self.my_units("Barbarian") + self.my_units("Barbarians"):  # max HP seen, for the Heal Spirit
            self.st[("hp", b.id)] = max(self.st.get(("hp", b.id), 0), b.hp)
        # OPEN: 0:30 and nothing played -> Hog + Skeletons at a bridge, or Firecracker behind a tower; or, if the
        # bot played first, Hog in the other lane once its play is answered
        if not self.st.get("opened"):
            self.st["opened"] = True
            if not self.seen:
                lane = int(self.rng.integers(2))
                if self.rng.random() < 0.5 and self.has("Hog Rider"):
                    return self.queue("OPEN", "0:30, nothing played: Hog + Skeletons at a bridge",
                                      Step("Hog Rider", (BRIDGE_X[lane], MY_BRIDGE_Y)),
                                      *([Step("Skeletons", (BRIDGE_X[lane], MY_BRIDGE_Y + 1.0), 0.2)]
                                        if self.has("Skeletons") else []))
                c = next((c for c in ("Firecracker", "Dart Goblin", "Skeletons") if self.has(c)), "Heal Spirit")
                return self.queue("OPEN", f"0:30, nothing played: {c} behind a tower", Step(c, (LANE_X[lane], 30.0)))
            if self.has("Hog Rider") and el >= 4:
                return self.hog_plan("OPEN", "the bot opened: Hog the other lane", self.hog_lane())
        g = self.golem()
        # R6: my Golem at its bridge -> ranged 3 behind it, Barbarians at the bridge, Heal Spirit on them
        if g is not None and g.y <= MY_BRIDGE_Y and ("r6", g.id) not in self.st:
            self.st[("r6", g.id)] = True
            gx, lane = g.x, lane_of(g.x)
            ranged = [c for c in ("Firecracker", "Dart Goblin") if self.has(c)]
            steps = [Step(ranged[0], (gx, g.y + 3.0))] if ranged else []
            if self.has("Barbarians"):
                steps.append(Step("Barbarians", (BRIDGE_X[lane], MY_BRIDGE_Y), 0.5))
                if self.has("Heal Spirit"):
                    steps.append(Step("Heal Spirit", (BRIDGE_X[lane], MY_BRIDGE_Y), 1.5))
            if steps:
                return self.queue("R6", "Golem at the bridge: " + " + ".join(s.card for s in steps), *steps)
        # R7: a ranged troop already behind the pushing Golem and the other in hand -> stack it on the first if the
        # bot's Fireball and Arrows are both out of its hand, else 4+ tiles behind it
        if g is not None and g.y <= MY_BRIDGE_Y + 2 and ("r7", g.id) not in self.st:
            first = [m for m in self.my_units() if m.name in ("Firecracker", "Dart Goblin") and abs(m.x - g.x) < 5]
            other = [c for c in ("Firecracker", "Dart Goblin") if self.has(c)]
            if first and other and el >= 3:
                self.st[("r7", g.id)] = True
                m, out = first[0], self.out_of_bot_hand("Fireball") and self.out_of_bot_hand("Arrows")
                at = (m.x, max(m.y, MY_BRIDGE_Y)) if out else (m.x, max(m.y + 4.0, MY_BRIDGE_Y + 4.0))
                return self.queue("R7", f"{'stack' if out else 'spread'} {other[0]} with the {m.name}",
                                  Step(other[0], at))
        # R3 (end): Heal Spirit on Barbarians below half
        hurt = [b for b in self.my_units("Barbarian") if b.hp < 0.5 * self.st.get(("hp", b.id), b.hp)]
        if len(hurt) >= 2 and self.has("Heal Spirit") and ("r3h", int(t // 5)) not in self.st:
            self.st[("r3h", int(t // 5))] = True
            return self.queue("R3", "Barbarians below half: Heal Spirit", Step("Heal Spirit", (hurt[0].x, hurt[0].y)))
        # Overtime: Hog the lowest bot tower whenever one hit finishes it
        if self.g.get_current_tick() >= OVERTIME_TICK and self.has("Hog Rider") and el >= 4:
            for lane in (0, 1):
                if 0 < self.tower_hp(0, lane) <= hog_hit():
                    return self.hog_plan("DEFAULT", "overtime: one Hog hit finishes the tower", lane, cheap=False)
        # R5: double, 9+, nothing of the bot's across -> Golem in the back of the weaker lane, or behind the King
        if phase != "single" and el >= 9 and not self.crossed() and self.has("Golem") and g is None:
            lane = self.weaker_bot_lane()
            at = KING_BACK if self.rng.random() < 0.3 else (LANE_X[lane], MY_BACK_Y)
            return self.queue("R5", f"{phase}: Golem in the back at {at}", Step("Golem", at))
        # R8: single, ahead on towers, bot on <= 3 -> Golem in the back anyway
        if phase == "single" and self.leading() and bot_el <= 3 and el >= 7 and self.has("Golem") and g is None:
            return self.queue("R8", "single but ahead and the bot is low: Golem in the back",
                              Step("Golem", (LANE_X[self.weaker_bot_lane()], MY_BACK_Y)))
        if self.has("Hog Rider"):
            # R2: Mini P.E.K.K.A out of the bot's hand -> Hog now; in hand and < 8 -> cycle instead of a lone Hog
            mp = next((s.t for s in reversed(self.seen[-4:]) if s.name == "Mini PEKKA"), None)
            if mp is not None and el >= 4 and ("r2", mp) not in self.st:  # once per Mini P.E.K.K.A play
                self.st[("r2", mp)] = True
                return self.hog_plan("R2", "Mini P.E.K.K.A out of cycle: Hog", self.hog_lane(), cheap=el >= 6)
            if mp is None and 7 <= el < 8 and phase == "single" and ("r2c", int(t // 10)) not in self.st:
                self.st[("r2c", int(t // 10))] = True
                c = next((c for c in ("Skeletons", "Firecracker") if self.has(c)), None)
                if c:
                    return self.queue("R2", f"Mini P.E.K.K.A may be in hand: cycle {c}, no lone Hog",
                                      Step(c, (MID_X, 31.0) if c == "Skeletons" else (LANE_X[self.hog_lane()], 30.0)))
            # DEFAULT: single, Hog + a 1-elixir card at 8+ (or right after a defence held); double, Hog in the
            # other lane during a Golem push
            last_def = max((m[0] for m in self.mine if m[1] in ("Barbarians", "Valkyrie")), default=-99)
            held = t - last_def < 4 and not self.crossed() and el >= 5
            if phase == "single" and (el >= 8 or held):
                return self.hog_plan("DEFAULT", "single: Hog + a 1-elixir card", self.hog_lane())
            if phase != "single" and g is not None and el >= 5:
                return self.hog_plan("DEFAULT", f"{phase}: Hog the other lane of the Golem", 1 - lane_of(g.x))
            if phase != "single" and el >= 9.5 and not self.has("Golem"):
                return self.hog_plan("DEFAULT", f"{phase}: full bar, no Golem: Hog", self.hog_lane())
        self._cycle()

    def _cycle(self):
        """CYCLE: a full bar with nothing better -> a cheap card at the back, so the bar never leaks."""
        if self.elixir < 9.5:
            return
        for card, at in (("Skeletons", (MID_X, 31.0)), ("Heal Spirit", (MID_X, 31.0)),
                         ("Firecracker", (LANE_X[int(self.rng.integers(2))], 30.0)),
                         ("Dart Goblin", (LANE_X[int(self.rng.integers(2))], 30.0)), ("Valkyrie", (MID_X, 29.0))):
            if self.has(card):
                return self.queue("CYCLE", f"full bar: cycle {card} at the back", Step(card, at))
