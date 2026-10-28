"""Defence situations for the RL against pro clones (2026-10-27, designed with Cole; phase one: his scenarios 2, 3, 4, 7).

The fork's own scenarios (python_ai/envs/scenarios.py) start a share of episodes with a threat on the board but score
them with the ordinary game reward, and their success read-out ("reward > -0.5") hid that the agent answered a bridge
push by counter-rushing and lost ~60% of a princess tower (decisions.md, 2026-10-27). These replace them: each
template draws a randomised mid-fight state (the threat already developing, both elixir bars set, a hand that holds
at least one affordable answer), ends as soon as the board is neutral again (the threat dead and no enemy unit on our
side; at least MIN_SECONDS, at most the template's timeout: Cole, 2026-10-28, "the opponent isn't gonna just let the
mini pekka take tower" — a window that ran on let surviving defenders walk over and hit an undefended tower), and is
scored on the outcome only:

  score = -(our tower HP lost / one princess tower's max HP)       tower preservation first
          - ELIXIR_COST * elixir we spent                           then efficiency
          + SURVIVOR_BONUS * elixir value of our units still alive  (useful defenders left, capped at 10 elixir)
          - TIMEOUT_PENALTY if the threat is still alive at the timeout

No credit for damage to the enemy's towers (Cole's rule: no offensive reward in defensive tests). Cole's scoring
safeguards this follows: rushing earns nothing; stalling hits the timeout; free starting troops are never ours (we
start with none); randomised states, and the exam (scripts/situation_exam.py) uses held-out seeds.
"""
from __future__ import annotations

import numpy as np

# engine card ids (bot77.sim.cards.engine_id)
HOG, BATTLE_RAM, MINI_PEKKA, KNIGHT, PRINCE, GIANT, GOLEM = 15, 81, 5, 0, 14, 2, 19
MUSKETEER, WIZARD, EXECUTIONER, MINIONS = 6, 11, 36, 41
MEGA_MINION, BOMBER, GOBLIN_HUT, FIREBALL, ARROWS = 43, 9, 95, 7, 3
MINION_HORDE, SKELETON_ARMY, GOBLIN_GANG, BATS, ARCHERS, WITCH, FLYING_MACHINE = 42, 12, 112, 78, 1, 71, 67

COST = {HOG: 4, BATTLE_RAM: 4, MINI_PEKKA: 4, KNIGHT: 3, PRINCE: 5, GIANT: 5, GOLEM: 8, MUSKETEER: 4, WIZARD: 5,
        EXECUTIONER: 5, MINIONS: 3, MEGA_MINION: 3, BOMBER: 2, GOBLIN_HUT: 4, FIREBALL: 4, ARROWS: 3,
        MINION_HORDE: 5, SKELETON_ARMY: 3, GOBLIN_GANG: 3, BATS: 2, ARCHERS: 3, WITCH: 5, FLYING_MACHINE: 4}

LANE_X = (3.0, 14.0)  # the princess-tower lanes (engine x; bridges at 2.5 / 14.5)
DROP_Y = 18.0  # their side of the bridge: where a player drops a Hog / Ram / melee card (Cole, 2026-10-28)
RIVER_Y = 17.0  # inside the river band: a unit spawned here walks out of the bridge (the fork's scenarios do the same)

# Goblin Hut in the middle (Cole: "building in the middle asap"), 3.5 tiles from the river: a Giant notices buildings
# within 7.5 tiles in this engine, and at y 11 the Hut was ~7.8 from where it leaves the bridge, so it walked past
HUT_PULL = (8.5, 12.0)
ELIXIR_COST = 0.04
SURVIVOR_BONUS = 0.02
TIMEOUT_PENALTY = 0.3
TRADE_COEF = 0.1  # spell situations: 0.1 a net elixir of value
ELIM_BONUS = 0.25  # of a body's value when it dies, so finishing beats spreading
MIN_SECONDS = 3  # don't call the board neutral before the threat has had a chance to arrive


def _lane(rng) -> float:
    return float(rng.choice(LANE_X))


def hog_rider(rng) -> dict:
    """Cole's 2: a solo Hog Rider at the bridge; 4-6 elixir; a building pull or a troop answer in hand."""
    x = _lane(rng)
    return {"name": "hog_rider", "timeout": 20, "own_elixir": rng.uniform(4, 6),
            "opp_elixir": rng.uniform(0, 3), "units": [(HOG, x + rng.uniform(-0.4, 0.4), DROP_Y + rng.uniform(0, 0.8))], "drop": True,
            "answers": [MINI_PEKKA, GOBLIN_HUT, MEGA_MINION]}  # Hut first failed to pull (x 8.5, y 9): Cole to place it


def battle_ram(rng) -> dict:
    """Cole's 3: a Battle Ram from the bridge at different distances; 4-6 elixir; scored over the whole attack
    (the Barbarians it drops included)."""
    x = _lane(rng)
    return {"name": "battle_ram", "timeout": 26, "own_elixir": rng.uniform(4, 6),
            "opp_elixir": rng.uniform(0, 3), "units": [(BATTLE_RAM, x, DROP_Y + rng.uniform(0, 1.5))], "drop": True,
            "answers": [MINI_PEKKA, GOBLIN_HUT, MEGA_MINION], "answers2": [BOMBER, MEGA_MINION]}


def melee(rng) -> dict:
    """Cole's 4: a Mini P.E.K.K.A, Knight or Prince approaching from the bridge; 3-5 elixir; cheap distraction in
    hand (the variant is kept in the name: one defence doesn't work equally against all three)."""
    card = int(rng.choice([MINI_PEKKA, KNIGHT, PRINCE]))
    x = _lane(rng)
    name = {MINI_PEKKA: "melee_mini_pekka", KNIGHT: "melee_knight", PRINCE: "melee_prince"}[card]
    return {"name": name, "timeout": 20, "own_elixir": rng.uniform(3, 5),
            "opp_elixir": rng.uniform(0, 3), "units": [(card, x + rng.uniform(-0.4, 0.4), DROP_Y + rng.uniform(0, 1.0))], "drop": True,
            "answers": [MEGA_MINION, GOBLIN_HUT, BOMBER]}  # a lone Bomber lost to the Knight too


def supported_push(rng) -> dict:
    """Cole's 7: a Giant approaching the bridge with a support behind it (Musketeer, Wizard, Executioner or Minions),
    or a slower Golem push from the back; 6-8 elixir; a tank answer and a support answer in hand."""
    x = _lane(rng)
    if rng.random() < 0.2:
        tank, ty, sup = GOLEM, rng.uniform(25.0, 28.0), int(rng.choice([WIZARD, MUSKETEER]))
        name, timeout = "supported_golem", 40
    else:
        tank, ty, sup = GIANT, rng.uniform(18.0, 20.5), int(rng.choice([MUSKETEER, WIZARD, EXECUTIONER, MINIONS]))
        name, timeout = "supported_giant", 32
    sup_answers = [ARROWS, MEGA_MINION] if sup == MINIONS else [FIREBALL, MINI_PEKKA, MEGA_MINION]
    return {"name": name, "timeout": timeout, "own_elixir": rng.uniform(6, 8),
            "opp_elixir": rng.uniform(0, 2), "units": [(tank, x, ty), (sup, x + rng.uniform(-0.5, 0.5), ty + rng.uniform(2.0, 3.5))],
            "answers": [GOBLIN_HUT, MINI_PEKKA], "answers2": sup_answers}


def _cluster(rng, card, cx, cy, spread):
    return [(card, cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread))]


def arrows_value(rng) -> dict:
    """Cole's 5: swarms approaching or supporting a tank, Arrows in hand, 3-5 elixir; some spread so one cast can't
    hit everything. Scored on the trade (enemy elixir value removed against what we spent), towers first."""
    x = _lane(rng)
    combo = int(rng.integers(5))
    spread = 0.6 if rng.random() < 0.7 else 2.2  # a third spread out: one cast can't take everything
    y = rng.uniform(14.5, 17.5)
    units = {0: [(MINION_HORDE, x, y)],
             1: [(SKELETON_ARMY, x, y), (GOBLIN_GANG, x + rng.uniform(-1.5, 1.5), y + 1.0)],
             2: [(MINIONS, x, y), (BATS, x + rng.uniform(-1.5, 1.5), y + 0.8)],
             3: [(ARCHERS, x, y + 1.5), (GOBLIN_GANG, x + rng.uniform(-1.5, 1.5), y)],
             4: [(int(rng.choice([GIANT, KNIGHT])), x, y - 1.0), (int(rng.choice([MINIONS, GOBLIN_GANG, SKELETON_ARMY])), x, y + 1.2)]}[combo]
    units = [u for c, ux, uy in units for u in _cluster(rng, c, ux, uy, spread)]
    return {"name": "arrows_value", "timeout": 20, "own_elixir": rng.uniform(3, 5), "opp_elixir": rng.uniform(0, 2),
            "units": units, "answers": [ARROWS], "score": "trade"}


def fireball_value(rng) -> dict:
    """Cole's 6: two or three medium-HP supports approaching together (some behind a tank, some already damaged),
    Fireball in hand, 4-6 elixir. Damage credited by the share of each unit's max HP removed, plus an elimination
    bonus, so finishing a unit beats spreading chip damage."""
    x = _lane(rng)
    pair = [(MUSKETEER, WIZARD), (MUSKETEER, WITCH), (WIZARD, MEGA_MINION), (MUSKETEER, FLYING_MACHINE)][int(rng.integers(4))]
    y = rng.uniform(18.5, 21.0)
    hurt = rng.random() < 0.3
    units = [(c, x + rng.uniform(-0.9, 0.9), y + rng.uniform(-0.6, 0.6), rng.uniform(0.4, 0.7) if hurt else 1.0)
             for c in pair]
    if rng.random() < 0.3:
        units.insert(0, (GIANT, x, y - 1.6, 1.0))  # a tank in front: the spell still belongs on the support
    return {"name": "fireball_value", "timeout": 20, "own_elixir": rng.uniform(4, 6), "opp_elixir": rng.uniform(0, 2),
            "units": units, "answers": [FIREBALL], "score": "trade"}


def emergency(rng) -> dict:
    """Cole's 10: a big push already on the board while we're down (4-5 elixir, they have 0-2 after building it),
    far enough back to regenerate before contact. Limiting the damage is the win."""
    x = _lane(rng)
    push = [(GOLEM, WIZARD, MEGA_MINION), (GIANT, MUSKETEER, MINI_PEKKA), (GIANT, WITCH, MINIONS)][int(rng.integers(3))]
    ty = rng.uniform(24.0, 27.0) if push[0] == GOLEM else rng.uniform(21.0, 23.0)
    units = [(push[0], x, ty), (push[1], x + rng.uniform(-0.6, 0.6), ty + rng.uniform(1.5, 2.5)),
             (push[2], x + rng.uniform(-0.8, 0.8), ty + rng.uniform(0.5, 1.5))]
    sup_answers = [ARROWS, MEGA_MINION] if MINIONS in push else [FIREBALL, MINI_PEKKA, MEGA_MINION]
    return {"name": "emergency", "timeout": 40, "own_elixir": rng.uniform(4, 5), "opp_elixir": rng.uniform(0, 2),
            "units": units, "answers": [GOBLIN_HUT, MINI_PEKKA], "answers2": sup_answers}


TEMPLATES = [hog_rider, battle_ram, melee, supported_push, arrows_value, fireball_value, emergency]
BY_NAME = {"hog_rider": hog_rider, "battle_ram": battle_ram, "melee_mini_pekka": melee, "melee_knight": melee,
           "melee_prince": melee, "supported_giant": supported_push, "supported_golem": supported_push,
           "arrows_value": arrows_value, "fireball_value": fireball_value, "emergency": emergency}


def _supported_defence(env, spec, played) -> tuple[int, float, float]:
    """Cole's defence for a supported push (2026-10-28): the building in the middle as soon as possible to pull the
    tank; without it in hand, cycle cheap cards (Bomber, our ranged card, in the back); once the push packs up at the
    bridge, Fireball the support (Arrows on Minions); then Mini P.E.K.K.A / Mega Minion on the support to finish it."""
    hand, el = list(env.game.get_hand()), env.game.get_elixir()
    lane_x, tank_id, sup_id = spec["units"][0][1], spec["units"][0][0], spec["units"][-1][0]
    units = env.game.get_units()
    tank = next((u for u in units if u.team == 1 and u.card_id == tank_id), None)
    sups = [u for u in units if u.team == 1 and u.card_id == sup_id]
    spell = ARROWS if sup_id == MINIONS else FIREBALL
    if "hut" not in played:
        if GOBLIN_HUT in hand and el >= COST[GOBLIN_HUT]:
            played.add("hut")
            return hand.index(GOBLIN_HUT), *HUT_PULL
        if GOBLIN_HUT not in hand and BOMBER in hand and el >= COST[BOMBER] + 2 and "cycle" not in played:
            played.add("cycle")
            return hand.index(BOMBER), lane_x, 3.0  # cycle toward the Hut, ranged card in the back
    packed = tank is not None and sups and tank.y <= 18.5 and min(abs(s.y - tank.y) for s in sups) <= 3.5
    if "spell" not in played and packed and spell in hand and el >= COST[spell]:
        played.add("spell")
        s0 = min(sups, key=lambda u: u.y)
        return hand.index(spell), float(s0.x), float(s0.y)
    if "finish" not in played and sups and ("spell" in played or spell not in hand):
        for c in ((MEGA_MINION,) if sup_id == MINIONS else (MINI_PEKKA, MEGA_MINION)):  # Mini P.E.K.K.A can't hit air
            if c in hand and el >= COST[c]:
                s0 = min(sups, key=lambda u: u.y)
                if s0.y > 16.0:  # still across the river: wait for it to come to us
                    break
                played.add("finish")
                return hand.index(c), float(np.clip(s0.x, 0, 17)), float(np.clip(s0.y - 1.0, 0, 14))
    if "hut" not in played and "tank" not in played and tank is not None and tank.y < 14.0 and MINI_PEKKA in hand \
            and el >= COST[MINI_PEKKA] and GOBLIN_HUT not in hand:
        played.add("tank")  # no Hut came: the Mini P.E.K.K.A on the tank before it reaches the tower
        return hand.index(MINI_PEKKA), float(tank.x), float(max(tank.y - 2.0, 4.0))
    return 4, 0.0, 0.0


def _emergency_defence(env, spec, played) -> tuple[int, float, float]:
    """The yardstick for a big push already on the board: Cole's supported-push plan first (Hut pull, spell the
    support once it packs, finish), then keep answering every second while anything is on our side: Mini P.E.K.K.A on
    the nearest ground body, Mega Minion on the nearest, Bomber behind the tower, the Hut again in the middle."""
    k = _supported_defence(env, spec, played)
    if k[0] != 4:
        return k
    if env.situation.seconds - played_last(played) < 1.5:
        return 4, 0.0, 0.0
    hand, el = list(env.game.get_hand()), env.game.get_elixir()
    foes = [u for u in env.game.get_units() if u.team == 1 and not u.tower and not u.building and u.y < 18.0]
    if not foes:
        return 4, 0.0, 0.0
    near = min(foes, key=lambda u: u.y)
    ground = [u for u in foes if not u.flying]
    for c in (MINI_PEKKA, MEGA_MINION, GOBLIN_HUT, BOMBER):
        if c not in hand or el < COST[c]:
            continue
        if c == MINI_PEKKA and ground:
            t = min(ground, key=lambda u: u.y)
            pos = (float(np.clip(t.x, 0, 17)), float(np.clip(t.y - 1.5, 1, 14)))
        elif c == MEGA_MINION:
            pos = (float(np.clip(near.x, 0, 17)), float(np.clip(near.y - 1.5, 1, 14)))
        elif c == GOBLIN_HUT:
            pos = HUT_PULL
        elif c == BOMBER:
            pos = (float(np.clip(near.x, 1, 16)), 3.0)
        else:
            continue
        played.add(f"@{env.situation.seconds:.1f}")
        return hand.index(c), *pos
    return 4, 0.0, 0.0


def played_last(played) -> float:
    ts = [float(p[1:]) for p in played if p.startswith("@")]
    return max(ts) if ts else -9.0


def _spell_value_defence(env, spec, played) -> tuple[int, float, float]:
    """The yardstick for the spell situations: the spell once, on the densest group of threats, when at least two
    are within its radius of one point (or the threat is nearly at our tower)."""
    if "spell" in played:
        return 4, 0.0, 0.0
    spell = spec["answers"][0]
    hand, el = list(env.game.get_hand()), env.game.get_elixir()
    if spell not in hand or el < COST[spell]:
        return 4, 0.0, 0.0
    radius = 2.5 if spell == FIREBALL else 4.0
    pts = [(u.x, u.y) for u in env.game.get_units() if u.id in env.situation.threat]
    if not pts:
        return 4, 0.0, 0.0
    best = max(pts, key=lambda p: sum((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= radius ** 2 for q in pts))
    n = sum((best[0] - q[0]) ** 2 + (best[1] - q[1]) ** 2 <= radius ** 2 for q in pts)
    if n >= min(2, len(pts)) or min(q[1] for q in pts) < 10.0:
        played.add("spell")
        return hand.index(spell), float(best[0]), float(best[1])
    return 4, 0.0, 0.0


def scripted_defence(env) -> tuple[int, float, float]:
    """The exam's yardstick (2026-10-28): a plain sensible defence, not a tuned one. The best answer in hand by the
    template's priority order (answers lists are ordered: e.g. Goblin Hut first against a Hog, Mini P.E.K.K.A first
    against a Battle Ram), once: Goblin Hut in the middle (x 8.5, y 9) to pull, a troop in front of the threatened tower
    (lane, y 9). Then the second answer once, when there is one: a spell on the support unit where it stands, a troop on
    the nearest enemy on our side (the Ram's Barbarians, a support unit) or in front of the tower. Only affordable cards;
    otherwise wait. (First version picked by hand order and answered a Ram with a lone Mega Minion: Cole, "our scripted
    defense is ass?")"""
    spec = env.situation.spec if env.situation else None
    if spec is None:
        return 4, 0.0, 0.0
    if not hasattr(env, "defend_played") or env.defend_played[0] is not env.situation:
        env.defend_played = (env.situation, set())
    played = env.defend_played[1]
    if spec["name"] == "emergency":
        return _emergency_defence(env, spec, played)
    if spec["name"].startswith("supported"):
        return _supported_defence(env, spec, played)
    if spec.get("score") == "trade":
        return _spell_value_defence(env, spec, played)
    hand, el = list(env.game.get_hand()), env.game.get_elixir()
    lane_x = spec["units"][0][1]
    for group in ("answers", "answers2"):
        if group in played or not spec.get(group):
            continue
        if group == "answers2" and "answers" not in played:
            break
        for c in spec[group]:
            if c not in hand or COST[c] > el:
                continue
            i = hand.index(c)
            enemies = [u for u in env.game.get_units() if u.team == 1 and not u.tower and u.y < 17.5]
            if c == GOBLIN_HUT:
                played.add(group)
                return i, *HUT_PULL
            if c in (FIREBALL, ARROWS):
                sup = [u for u in env.game.get_units() if u.team == 1 and u.card_id == spec["units"][-1][0]]
                if not sup:
                    continue
                played.add(group)
                return i, float(sup[0].x), float(sup[0].y)
            if group == "answers2":
                if not enemies:
                    break  # nothing to clean up yet (the Ram's Barbarians appear when it breaks)
                tgt = min(enemies, key=lambda u: u.y)
                played.add(group)
                return i, float(np.clip(tgt.x, 0, 17)), float(np.clip(tgt.y + 1.0, 0, 14))
            played.add(group)
            return i, lane_x, 9.0
    return 4, 0.0, 0.0


def deal_hand(rng, deck: list[int], spec: dict) -> list[int]:
    """Four cards from the deck holding an answer we can afford now (and, for the supported push, a support answer)."""
    afford = [c for c in spec["answers"] if c in deck and COST[c] <= spec["own_elixir"]]
    first = [int(rng.choice(afford or [c for c in spec["answers"] if c in deck]))]
    if spec.get("answers2"):
        second = [c for c in spec["answers2"] if c in deck and c not in first]
        if second:
            first.append(int(rng.choice(second)))
    rest = [c for c in deck if c not in first]
    rng.shuffle(rest)
    hand = first + rest[: 4 - len(first)]
    rng.shuffle(hand)
    return hand


class Situation:
    """One running situation: set up on a game at reset, then stepped until it scores."""

    def __init__(self, game, spec: dict, deck: list[int], rng, dt: float = 1.0):
        self.game, self.spec = game, spec
        game.set_elixir_for_team(0, float(spec["own_elixir"]))
        game.set_elixir_for_team(1, float(spec["opp_elixir"]))
        self.hand = deal_hand(rng, deck, spec)
        self.hand_ok = bool(game.set_hand_for_team(0, self.hand))
        self.threat: set[int] = set()
        self.just_dropped = False
        if not spec.get("drop"):
            self._inject(deploy_ticks=0)  # already on the move (the supported push starts mid-push)
        self.tower0 = [max(0, game.get_tower_hp(0, s)) for s in range(3)]
        self.princess_max = float(game.get_tower_max_hp(0, 1))
        self.spent0 = float(game.get_elixir_spent(0))
        self.dt = dt  # seconds per env step (1.0, or 0.1 at a decision a tick)
        self.seconds = 0.0
        self.neutral_since: float | None = None

    def _inject(self, deploy_ticks: int) -> None:
        import clash_royale_env as cre
        before = {u.id for u in self.game.get_units()}
        for unit in self.spec["units"]:
            card, x, y = unit[:3]
            frac = unit[3] if len(unit) > 3 else 1.0
            hp = -1.0 if frac >= 1.0 else float(dict(cre.get_card_stats(int(card)))["hp"]) * frac
            self.game.inject(int(card), float(x), float(y), 1, hp, deploy_ticks)
        self.game.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)  # commit the injected units
        new = [u for u in self.game.get_units() if u.id not in before and u.team == 1]
        self.threat = {u.id for u in new}
        # value bookkeeping for trade scoring: elixir per body and the body's full HP
        self.threat_info = {}
        for u in new:
            st = dict(cre.get_card_stats(int(u.card_id))) if u.card_id >= 0 else {}
            bodies = max(1, int(st.get("bodies", 1) or 1))
            self.threat_info[u.id] = (COST.get(u.card_id, st.get("cost", 0)) / bodies, float(st.get("hp", u.hp) or u.hp), u.hp)

    def resolved(self) -> bool:
        """The injected threat is dead (anything it spawned, e.g. the Battle Ram's Barbarians, is on team 1 too and
        counts while it's on our side)."""
        units = self.game.get_units()
        alive = {u.id for u in units}
        if self.threat & alive:
            return False
        return not any(u.team == 1 and not u.tower and not u.building and u.y < 17.5 for u in units)

    def step(self, game_over: bool) -> tuple[bool, float, dict]:
        """After each env step: (finished, score, details). Finishes when the board is neutral (after MIN_SECONDS), at
        the timeout, or when the game ends."""
        self.seconds += self.dt
        self.just_dropped = False
        if self.spec.get("drop") and not self.threat and self.seconds >= 1.0 - 1e-6 and not game_over:
            # the board starts empty; the opponent drops the card now, with the normal 1 s deploy, as in a real game
            self._inject(deploy_ticks=-1)
            self.just_dropped = True
            return False, 0.0, {}
        if self.spec.get("drop") and not self.threat:
            return (True, 0.0, {}) if game_over else (False, 0.0, {})  # before the drop (the first second)
        neutral = self.resolved()
        if not neutral:
            self.neutral_since = None
        elif self.neutral_since is None:
            self.neutral_since = self.seconds
        # neutral for 2 s in a row: a Battle Ram's Barbarians appear ~1 s after it breaks, and the board looked empty
        # for that second (2026-10-28)
        steady = self.neutral_since is not None and self.seconds - self.neutral_since >= 2.0 - self.dt - 1e-6
        done = game_over or self.seconds >= self.spec["timeout"] - 1e-6 or (self.seconds >= MIN_SECONDS - 1e-6 and steady)
        if not done:
            return False, 0.0, {}
        lost = sum(max(0.0, a - max(0, self.game.get_tower_hp(0, s))) for s, a in enumerate(self.tower0))
        spent = float(self.game.get_elixir_spent(0)) - self.spent0
        unresolved = not self.resolved()
        survivors = sum(COST.get(u.card_id, 0) for u in self.game.get_units() if u.team == 0 and not u.tower and u.hp > 0)
        if self.spec.get("score") == "trade":
            # Cole's spell scoring: enemy elixir value removed (share of each body's max HP taken, capped at what it
            # had, plus ELIM_BONUS of its value if killed) against what we spent; towers still first
            alive = {u.id: u for u in self.game.get_units()}
            value = 0.0
            for uid, (worth, max_hp, hp0) in getattr(self, "threat_info", {}).items():
                hp1 = alive[uid].hp if uid in alive else 0.0
                value += worth * min(1.0, max(0.0, hp0 - hp1) / max(max_hp, 1.0)) + (ELIM_BONUS * worth if uid not in alive else 0.0)
            score = -lost / self.princess_max + TRADE_COEF * (value - spent) - (TIMEOUT_PENALTY if unresolved else 0.0)
        else:
            score = (-lost / self.princess_max - ELIXIR_COST * spent + SURVIVOR_BONUS * min(survivors, 10)
                     - (TIMEOUT_PENALTY if unresolved else 0.0))
        return True, float(score), {"tower_hp_lost": lost, "elixir_spent": spent, "unresolved": unresolved,
                                    "seconds": round(self.seconds, 1)}
