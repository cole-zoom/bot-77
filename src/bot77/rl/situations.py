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
VALKYRIE, CANNON = 10, 25

COST = {HOG: 4, BATTLE_RAM: 4, MINI_PEKKA: 4, KNIGHT: 3, PRINCE: 5, GIANT: 5, GOLEM: 8, MUSKETEER: 4, WIZARD: 5,
        EXECUTIONER: 5, MINIONS: 3, MEGA_MINION: 3, BOMBER: 2, GOBLIN_HUT: 4, FIREBALL: 4, ARROWS: 3,
        MINION_HORDE: 5, SKELETON_ARMY: 3, GOBLIN_GANG: 3, BATS: 2, ARCHERS: 3, WITCH: 5, FLYING_MACHINE: 4,
        VALKYRIE: 4, CANNON: 3}

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
COUNTER_TOLERANCE = 800  # defend_counter: tower HP the push may take and still unlock the attack credit
MIN_SECONDS = 3  # don't call the board neutral before the threat has had a chance to arrive
# The board at the end (Cole 2026-10-28, six play-page games that scored badly "due to numbers, but the situation on
# the board is good for us"): what we still have on the board refunds its elixir (cost a body x HP share: a troop that
# survives, e.g. a counter-push, wasn't elixir lost; half a refund barely moved his six), and what of theirs is still
# on our side costs us by its value instead of a flat 0.3 whenever anything is left
BOARD_REFUND = ELIXIR_COST
THREAT_LEFT_COEF = 0.08
THREAT_LEFT_CAP = 0.6
DEALT_COEF = 0.15  # tower damage we deal, a princess's worth = +0.15 (Cole: "slightly ... not so much that they just
#                    bum rush"; losing one is -1)
# Cole's takes as answer keys (Cole: "take all of my takes and use them as answer keys or positive signal"): on one of
# his starts, a play of the card he played within KEY_DT seconds and KEY_DIST tiles of his counts as matching; the
# training reward adds KEY_BONUS x the share of his take matched (the best of his takes there), not the score
KEY_BONUS = 0.3
KEY_DT = 1.5
KEY_DIST = 2.5
# Elixir is worth something on both sides (docs/plan-elixir-spells.md P2, 2026-10-29): what regen would have added
# while we sat at 10 is lost, at the same rate as spending it; a Fireball / Arrows that touches nothing of theirs costs
# its elixir again on top (the spell guard makes that rare in the sim; kept for the guard-off exam and the record).
# Both apply to every score kind (score v2; score_v1, the formula Cole played under, is unchanged).
LEAK_COEF = ELIXIR_COST
WASTED_SPELL_COEF = 0.04


_REGEN: list[float] = []


def regen_per_tick() -> float:
    """The engine's elixir regeneration per tick at 1x, measured once (no copy of the engine's constant here)."""
    if not _REGEN:
        import clash_royale_env as cre
        g = cre.ClashRoyaleEnv([0, 1, 2, 3, 5, 6, 7, 9], [0, 1, 2, 3, 5, 6, 7, 9], 3600,
                               cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
        g.seed(0)
        g.set_elixir_for_team(0, 0.0)
        g.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 100)
        _REGEN.append(float(g.get_elixir_for_team(0)) / 100.0 / float(cre.elixir_multiplier_at_tick(50)))
    return _REGEN[0]


def leak_step(game, team: int, el_before: float, spent_before: float, ticks: int) -> float:
    """Elixir regen would have added over the last `ticks` ticks that the cap threw away (for `team`)."""
    import clash_royale_env as cre
    now = int(game.get_current_tick())
    regen = sum(regen_per_tick() * float(cre.elixir_multiplier_at_tick(t)) for t in range(max(0, now - ticks), now))
    spent = float(game.get_elixir_spent(team)) - spent_before
    return max(0.0, el_before + regen - spent - float(game.get_elixir_for_team(team)))


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


# --- Cole's 1 / 8 / 9: a scripted opponent that reacts to us (2026-10-28; reviewed, in TEMPLATES since that day) ---
# The opponent plays through step_self_play in its own frame: x as on the board, y mirrored (33 - y), so its bridge
# drop at board y 18 is its y 15 and its back row at board y 30 is its y 3.

def safe_opening(rng) -> dict:
    """Cole's 1: an empty board, both on 5 elixir, and one of four opening plans for them; 30 s, towers first, no
    credit for damage we deal (the test is whether we answer, not whether we race)."""
    plan = ["tank_back", "bridge_rush", "support_back", "wait_defend"][int(rng.integers(4))]
    return {"name": f"opening_{plan}", "timeout": 30, "own_elixir": 5.0, "opp_elixir": 5.0, "units": [],
            "answers": [GOBLIN_HUT, MINI_PEKKA], "window": True, "score": "opening", "opp": plan,
            "opp_wait": rng.uniform(2, 5), "lane": _lane(rng),
            "tank": int(rng.choice([GIANT, GOLEM])), "rush": int(rng.choice([HOG, BATTLE_RAM])),
            "support": int(rng.choice([MUSKETEER, WIZARD]))}


def defend_counter(rng) -> dict:
    """Cole's 8: one of the supported pushes, 6-8 elixir, Giant in our deck; they keep 2-4 elixir back and defend
    what crosses. 45 s: our towers first, then damage to theirs and the elixir we end on, the attack bonus only once
    the push is beaten inside the tolerance."""
    spec = supported_push(rng)
    spec.update({"name": "counter_" + spec["name"].split("_", 1)[1], "timeout": 45, "own_elixir": rng.uniform(6, 8),
                 "opp_elixir": rng.uniform(2, 4), "window": True, "score": "counter", "opp": "defend"})
    return spec


def dont_overcommit(rng) -> dict:
    """Cole's 9: we're on 3, they're on 10, nothing on the board. Spend on attack and they punish in the other lane;
    otherwise they build a slower push that arrives anyway (so doing nothing can't solve it). 30 s."""
    return {"name": "deficit", "timeout": 30, "own_elixir": 3.0, "opp_elixir": 10.0, "units": [],
            "answers": [GOBLIN_HUT, MINI_PEKKA, BOMBER], "window": True, "score": "deficit", "opp": "punish",
            "lane": _lane(rng), "slow_at": rng.uniform(6, 12), "tank": int(rng.choice([GIANT, GOLEM])),
            "support": int(rng.choice([MUSKETEER, WIZARD])), "rush": int(rng.choice([HOG, BATTLE_RAM])),
            "punish_at": rng.uniform(3.5, 5.0)}


REVIEW_TEMPLATES = [safe_opening, defend_counter, dont_overcommit]
# the scripted opponent's deck in those (the engine only deals a team cards from its own deck)
SCRIPT_DECK = [GIANT, GOLEM, HOG, BATTLE_RAM, MUSKETEER, WIZARD, CANNON, MINI_PEKKA]
# What the scripted opponent plays when the card it wants isn't in its hand: the same job from the cards it holds, in
# this order, or nothing (it waits). It used to put the card it wanted into its hand (set_hand_for_team), skipping its
# cycle: three Mini P.E.K.K.As in a row on the play page (Cole, 2026-10-30, "how is this bot playing 3 Mini pekkas in
# a row??")
SCRIPT_SUBS = {CANNON: (CANNON, MINI_PEKKA, MUSKETEER, WIZARD), MINI_PEKKA: (MINI_PEKKA, CANNON, MUSKETEER, WIZARD),
               GIANT: (GIANT, GOLEM), GOLEM: (GOLEM, GIANT), HOG: (HOG, BATTLE_RAM), BATTLE_RAM: (BATTLE_RAM, HOG),
               MUSKETEER: (MUSKETEER, WIZARD), WIZARD: (WIZARD, MUSKETEER)}


def script_play(sit_, g, card: int, x: float, board_y: float, near=None):
    """The scripted opponent's play of `card` (or the card in its hand that does the same job, SCRIPT_SUBS) at (x,
    board_y) in board coordinates: (hand index, x, y in its own frame), or None to wait (not in hand, or short of
    elixir). A defender standing in for a Cannon goes in front of `near` (the unit it answers); a Cannon standing in
    for a defender goes in the middle."""
    hand, el = list(g.get_hand_for_team(1)), float(g.get_elixir_for_team(1))
    have = next((c for c in SCRIPT_SUBS.get(card, (card,)) if c in hand), None)
    if have is None or el < COST.get(have, 10):
        return None
    if have != card and CANNON in (card, have) and near is not None:
        x, board_y = (8.5, 25.0) if have == CANNON else (near.x, min(near.y + 2.5, 30.0))
    sit_.opp_state.setdefault("plays", []).append((round(sit_.seconds, 1), have, round(x, 1), board_y))
    return hand.index(have), float(x), float(33.0 - board_y)


def _ours_across(game) -> list:
    return [u for u in game.get_units() if u.team == 0 and not u.tower and not u.building and u.y > 17.5]


def opponent_script(sit_, env):
    """The scripted opponent's move this second for the review situations: (card, x, y) in its own frame, or None."""
    spec, st, g = sit_.spec, sit_.opp_state, env.game
    t = sit_.seconds

    def play(card, x, board_y, near=None):
        return script_play(sit_, g, card, x, board_y, near)

    def defend():
        # a defender in front of whatever of ours has crossed, once per crossing unit
        for u in _ours_across(g):
            if u.id in st.setdefault("answered", set()):
                continue
            card = CANNON if u.card_id in (GIANT, HOG, BATTLE_RAM, GOLEM) else MINI_PEKKA
            x = 8.5 if card == CANNON else u.x
            r = play(card, x, 25.0 if card == CANNON else min(u.y + 2.5, 30.0), near=u)
            if r:
                st["answered"].add(u.id)
            return r  # or save up for the answer
        return None

    plan = spec["opp"]
    if plan == "tempo":
        return _tempo_opponent(sit_, env)
    lane = spec.get("lane", 3.0)
    if plan == "defend":
        return defend()
    if plan == "wait_defend":
        return defend()
    if plan in ("tank_back", "support_back", "bridge_rush"):
        if "opened" not in st:
            if plan == "bridge_rush":
                st["opened"] = True
                return play(spec["rush"], lane, 18.0)
            if t >= spec["opp_wait"]:
                card = spec["tank"] if plan == "tank_back" else spec["support"]
                r = play(card, lane, 30.0)
                if r:
                    st["opened"] = True
                return r
            return defend()
        # follow up: support behind the tank when it reaches the bridge; otherwise defend
        tank = next((u for u in g.get_units() if u.team == 1 and u.card_id == spec.get("tank")), None)
        if plan == "tank_back" and tank is not None and tank.y < 22.0 and "followed" not in st:
            r = play(spec["support"], tank.x, min(tank.y + 3.0, 30.0))
            if r:
                st["followed"] = True
            return r
        return defend()
    if plan == "punish":
        spent = float(g.get_elixir_spent(0)) - sit_.spent0
        across = sum(COST.get(u.card_id, 0) for u in _ours_across(g))
        if "attacked" not in st and (across >= 3 or spent >= 4) and t >= 1.0:
            st["attacked"] = "punish"
            st["punish_from"] = t
            far = LANE_X[1] if (sum(u.x for u in _ours_across(g)) / max(1, len(_ours_across(g)))) < 8.5 else LANE_X[0]
            st["punish_lane"] = far if _ours_across(g) else lane
        if st.get("attacked") == "punish":
            # the punish: a win condition at the bridge, then support behind it, in the lane we left open
            pl = st["punish_lane"]
            if "p1" not in st:
                r = play(spec["rush"], pl, 18.0)
                if r:
                    st["p1"] = True
                return r
            if "p2" not in st and t - st["punish_from"] >= 1.0:
                r = play(spec["support"], pl, 20.5)
                if r:
                    st["p2"] = True
                return r
            return defend()
        if t >= spec["slow_at"] and "attacked" not in st:
            st["attacked"] = "slow"
        if st.get("attacked") == "slow":
            if "s1" not in st:
                r = play(spec["tank"], lane, 30.0)
                if r:
                    st["s1"] = True
                return r
            tank = next((u for u in g.get_units() if u.team == 1 and u.card_id == spec["tank"]), None)
            if "s2" not in st and tank is not None and tank.y < 24.0:
                r = play(spec["support"], tank.x, min(tank.y + 2.5, 30.0))
                if r:
                    st["s2"] = True
                return r
        return defend()
    return None


# --- Tempo situations: when to spend (docs/plan-elixir-spells.md P3, 2026-10-29) ---
# Matched pairs: the same board (drawn first, from the same seed), a different elixir picture or game state, opposite
# right answers. Cole: "it needs to know when to chill, and when to place a lot ... its knowing when to spend, and needs
# to be learnt." Scored over a fixed window with the tempo opponent below.
#
# Cole on when to press (2026-10-29, verbatim): "in a even game, +3 is a valid placeholder. But if were desperate, like
# late game, and were down a tower, we need to go all in, so even 0 or +1, we try for a hail mary. Or if we need like
# 10 damage, hail marry/spell their tower".
PRESS_LEAD = 3.0  # an even game: press at our elixir - theirs >= +3 (Cole's placeholder)
DESPERATE_LEAD = 0.0  # desperate: all in at 0 or +1 (Cole)
# OUR CUT-OFFS, open questions for Cole: "late" = the last minute of regulation (engine tick >= 1200, 2:00 elapsed,
# where double elixir starts) or overtime; "down a tower" = fewer of our towers standing than theirs
LATE_TICK = 1200
MID_TICKS = (600, 1150)  # the calm / pressure templates: 1:00-1:55 elapsed, single elixir
DESPERATE_TICKS = (1250, 1700)  # 2:05-2:50 elapsed, double elixir
CHILL_AT = 9.5  # chill: hold until about to leak, then one cheap or back play
TEMPO_WINDOW = 25  # seconds, the same for both halves of a pair
DEALT_COEF_PRESS = 1.0  # press / desperate: a hit point of theirs is worth one of ours (losing a Princess = -1)
_SPELL_TOWER: dict[int, float] = {}


def spell_tower_damage(card: int) -> float:
    """A spell's damage to a Princess Tower, measured by casting it on one (no copy of the engine's numbers)."""
    if card not in _SPELL_TOWER:
        import clash_royale_env as cre
        g = cre.ClashRoyaleEnv([0, 1, 2, 3, 5, 6, 7, 9], [0, 1, 2, 3, 5, 6, 7, 9], 3600,
                               cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
        g.seed(0)
        tw = next(u for u in g.get_units() if u.team == 1 and u.tower and u.x < 8)
        hp0 = g.get_tower_hp(1, 1)
        g.inject(int(card), float(tw.x), float(tw.y), 0, -1.0, -1)
        g.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 60)
        _SPELL_TOWER[card] = float(hp0 - g.get_tower_hp(1, 1))
    return _SPELL_TOWER[card]


def _calm_board(rng) -> dict:
    """Nothing is happening: an empty board, or a lone Cannon of theirs in the middle of their side."""
    units = [(CANNON, 8.5, rng.uniform(23.5, 25.5))] if rng.random() < 0.4 else []
    return {"units": units, "lane": _lane(rng), "tick": int(rng.integers(*MID_TICKS)),
            "tank": int(rng.choice([GIANT, GOLEM])), "support": int(rng.choice([MUSKETEER, WIZARD])),
            "rush": int(rng.choice([HOG, BATTLE_RAM])), "answers": [BATTLE_RAM, GIANT],
            "answers2": [GOBLIN_HUT, BOMBER, MEGA_MINION, MINI_PEKKA]}


def _pressure_board(rng) -> dict:
    """We just defended their push: our survivors (hurt) on our side in its lane, theirs dead or one nearly dead."""
    b = _calm_board(rng)
    x = b["lane"]
    own = [(MINI_PEKKA, x + rng.uniform(-0.8, 0.8), rng.uniform(9.0, 12.5), rng.uniform(0.35, 0.9))]
    if rng.random() < 0.6:
        own.append((MEGA_MINION, x + rng.uniform(-1.2, 1.2), rng.uniform(8.0, 11.5), rng.uniform(0.4, 1.0)))
    b["own_units"] = own
    b["units"] = [(MUSKETEER, x + rng.uniform(-0.6, 0.6), rng.uniform(11.0, 13.5), rng.uniform(0.05, 0.15))] \
        if rng.random() < 0.5 else []
    return b


# --- Full-game tempo (Cole, 2026-10-30): "These tempos should start out in these situatins, and last the rest of teh
# game to teach teh bot how to act in certain situations, and clear out the game." A tempo start plays on to the end of
# the game against the pro clone (no script: it played three Mini P.E.K.K.As in a row), and every step is rewarded by
# the tempo the game is in then, with the game's result on top (decisions.md, 2026-10-30):
#   chill / even  less damage on our towers and a higher elixir advantage (Cole: "goal is to get to press")
#   press         the damage differential, theirs - ours ("bigger damamge differential = higher reward")
#   desperate     pass / fail when regulation ends: overtime (or a win) passes ("simple as that")
#   spell finish  pass / fail on the start's window: a Princess Tower of theirs fell to our spell
# The tempo moves with the board: chill / even -> press at a lead of PRESS_LEAD; press -> even once the lead is gone and
# nothing of ours is across; anything -> desperate late with fewer towers than theirs; desperate -> even once paid.
FULL_GAME_MODES = {"chill": "chill", "press_calm": "press", "desperate_even": "even", "desperate": "desperate",
                   "spell_finish": "spell_finish"}
ADVANTAGE_COEF = ELIXIR_COST  # chill / even: an elixir of advantage (bank + board at HP share, ours - theirs)
PASS_FAIL = 1.0  # desperate and spell finish: +1 pass, -1 fail (a Princess Tower's worth)
RESULT_COEF = 1.0  # the game's result on top (+1 win, -1 loss, 0 draw; the trainer prices a draw like a loss)
REGULATION_END_TICK = 1800  # 3:00 in 0.1 s ticks (MatchRules::REGULATION_END_TICK)
PRESS_HOLD = 10.0  # s: a press lasts at least this long, to start the push it's for


def _tempo(name, board, own, opp, score, **kw) -> dict:
    spec = {**board, "name": name, "timeout": TEMPO_WINDOW, "own_elixir": own, "opp_elixir": opp, "window": True,
            "score": score, "opp": "tempo", "tempo": True, **kw}
    if name in FULL_GAME_MODES:
        # the rest of the game against the pro clone; spell_finish keeps its window for the pass / fail
        spec.update({"full_game": True, "mode": FULL_GAME_MODES[name], "opp": None, "window": False})
    return spec


def chill(rng) -> dict:
    """Calm pair, chill half: nothing happening, they're at about our elixir or more (8-10). Hold to 9-10, then one
    cheap or back play, keep elixir for what they drop. Dumping at the bridge loses to their counter; never playing
    loses to the leak and their push."""
    b = _calm_board(rng)
    own = rng.uniform(6.0, 9.0)
    return _tempo("chill", b, own, rng.uniform(max(8.0, own), 10.0), "tempo")


def press_calm(rng) -> dict:
    """Calm pair, press half: the same board, but they're at 0-3 (they just spent). Go in now, before they rebuild."""
    b = _calm_board(rng)
    own = rng.uniform(6.0, 9.0)
    return _tempo("press_calm", b, own, rng.uniform(0.0, 3.0), "press")


def punish(rng) -> dict:
    """Pressure pair, press half: we just defended (survivors on our side), ours 7-10, theirs 0-3: counter-push behind
    the survivors."""
    b = _pressure_board(rng)
    own = rng.uniform(7.0, 10.0)
    return _tempo("punish", b, own, rng.uniform(0.0, 3.0), "press")


def hold_after_defence(rng) -> dict:
    """Pressure pair, chill half: the same board, but they're at 7-10: one support play behind the survivors or
    nothing; keep elixir for their next push."""
    b = _pressure_board(rng)
    own = rng.uniform(7.0, 10.0)
    return _tempo("hold_after_defence", b, own, rng.uniform(7.0, 10.0), "tempo")


def desperate(rng) -> dict:
    """Desperate pair, press half (Cole: "late game, and were down a tower, we need to go all in, so even 0 or +1"):
    late in regulation (double elixir), one of our Princess Towers down, a lead of 0 to +1. All in."""
    b = _calm_board(rng)
    own = rng.uniform(5.0, 8.0)
    opp = max(0.0, own - rng.uniform(0.0, 1.0))
    down = 1 if b["lane"] < 8.5 else 2  # the tower on their planned lane's side fell
    return _tempo("desperate", b, own, opp, "desperate", tick=int(rng.integers(*DESPERATE_TICKS)), destroy_own=down)


def desperate_even(rng) -> dict:
    """Desperate pair, chill half: the same board and elixir (lead 0 to +1) in an even game, mid-game, no tower down:
    chill."""
    b = _calm_board(rng)
    own = rng.uniform(5.0, 8.0)
    opp = max(0.0, own - rng.uniform(0.0, 1.0))
    rng.integers(*DESPERATE_TICKS)  # the same draws as desperate() (the board is identical; the clock stays mid-game)
    return _tempo("desperate_even", b, own, opp, "tempo")


def spell_finish(rng) -> dict:
    """Single (Cole: "if we need like 10 damage, hail marry/spell their tower"): one of their Princess Towers within a
    spell's tower damage of falling, our elixir 4+, the spell in hand. Spell the tower, whatever the lead."""
    b = _calm_board(rng)
    spell = int(rng.choice([FIREBALL, ARROWS]))
    hp = rng.uniform(5.0, max(6.0, spell_tower_damage(spell) - 2.0))
    slot = 1 if b["lane"] < 8.5 else 2
    b.update({"answers": [spell], "answers2": [BATTLE_RAM, GIANT, GOBLIN_HUT, BOMBER]})
    return _tempo("spell_finish", b, rng.uniform(max(4.0, COST[spell]), 9.0), rng.uniform(2.0, 9.0), "press",
                  opp_tower_hp={slot: round(hp, 1)}, timeout=10)


TEMPO_PAIRS = [(chill, press_calm), (hold_after_defence, punish), (desperate_even, desperate)]  # (chill half, press half)
TEMPO_TEMPLATES = [chill, press_calm, desperate_even, desperate, spell_finish]
# Held out (2026-10-29): the pressure pair failed its reference ordering on the training seeds (scripts/tempo_check.py):
# against an opponent on 7-10, dumping behind the survivors (+0.32) beat holding (+0.15), so the scoring / scripted
# opponent can't judge hold_after_defence, and punish alone would teach only one half. Not in training, demos or the
# exam; on the play page so Cole can play it (open question: is a counter-push behind survivors right into 7-10?)
HELD_OUT_TEMPLATES = [hold_after_defence, punish]


def _ours_alive(game, team: int) -> int:
    return sum(1 for s in range(3) if game.get_tower_hp(team, s) > 0)


def _tempo_opponent(sit_, env):
    """The tempo situations' opponent (one script for both halves of a pair: only the elixir differs). In order: answer
    what crosses (a Cannon in the middle for a tank / win condition, a Mini P.E.K.K.A in front of anything else,
    retried until it can afford it); punish a commitment of ours in the other lane (their win condition at the bridge,
    support behind it) once we've spent 5+ or put 3+ elixir across; and at 9.5+ elixir build its own push (a tank at
    the back of the planned lane, support behind it at the bridge), as a player on a full bar does."""
    spec, st, g = sit_.spec, sit_.opp_state, env.game
    t, el = sit_.seconds, g.get_elixir_for_team(1)

    def play(card, x, board_y, near=None):
        return script_play(sit_, g, card, x, board_y, near)

    for u in _ours_across(g):
        if u.id in st.setdefault("answered", set()):
            continue
        card = CANNON if u.card_id in (GIANT, HOG, BATTLE_RAM, GOLEM) else MINI_PEKKA
        r = play(card, 8.5 if card == CANNON else u.x, 25.0 if card == CANNON else min(u.y + 2.5, 30.0), near=u)
        if r:
            st["answered"].add(u.id)
            return r
        return None  # save up for the answer
    spent = float(g.get_elixir_spent(0)) - sit_.spent0
    across = _ours_across(g)
    if "attacked" not in st and (sum(COST.get(u.card_id, 0) for u in across) >= 3 or spent >= 5) and t >= 1.0:
        st["attacked"] = "punish"
        st["punish_from"] = t
        st["lane_p"] = (LANE_X[1] if sum(u.x for u in across) / len(across) < 8.5 else LANE_X[0]) if across else spec["lane"]
    if st.get("attacked") == "punish":
        if "p1" not in st:
            r = play(spec["rush"], st["lane_p"], 18.0)
            if r:
                st["p1"] = True
            return r
        if "p2" not in st and t - st["punish_from"] >= 1.0:
            r = play(spec["support"], st["lane_p"], 20.5)
            if r:
                st["p2"] = True
            return r
        return None
    if "attacked" not in st and el >= CHILL_AT:
        st["attacked"] = "push"
    if st.get("attacked") == "push":
        if "s1" not in st:
            r = play(spec["tank"], spec["lane"], 30.0)
            if r:
                st["s1"] = True
            return r
        tank = next((u for u in g.get_units() if u.team == 1 and u.card_id == spec["tank"]), None)
        if "s2" not in st and tank is not None and tank.y < 24.0:
            r = play(spec["support"], tank.x, min(tank.y + 2.5, 30.0))
            if r:
                st["s2"] = True
            return r
    return None


def tempo_state(env) -> dict:
    """What the scripted tempo policy reads: the lead (our elixir - theirs: the script counts perfectly; the network
    has no input for theirs, P5), desperate (late and down a tower), and a spell finish if one is on."""
    g = env.game
    lead = float(g.get_elixir()) - float(g.get_elixir_for_team(1))
    late = int(g.get_current_tick()) >= LATE_TICK
    desperate_ = late and _ours_alive(g, 0) < _ours_alive(g, 1)
    return {"lead": lead, "desperate": desperate_, "threshold": DESPERATE_LEAD if desperate_ else PRESS_LEAD}


def _finish_tower(env):
    """(hand index, x, y) to spell an enemy Princess Tower that the spell in hand would bring down, or None."""
    g = env.game
    hand, el = list(g.get_hand()), g.get_elixir()
    for c in (ARROWS, FIREBALL):
        if c not in hand or el < COST[c]:
            continue
        for u in g.get_units():
            if u.team == 1 and u.tower and abs(u.x - 8.5) > 1.0 and 0 < u.hp <= spell_tower_damage(c):
                return hand.index(c), float(u.x), float(u.y)
    return None


def tempo_policy(env, played, mode=None) -> tuple[int, float, float]:
    """The scripted tempo policy (plan P3): defend on a threat; spell a tower that the spell finishes; press when the
    lead reaches the threshold (+3 in an even game, 0 when desperate; Cole), otherwise chill. mode "chill" / "press"
    forces one (the reference players)."""
    g = env.game
    foes = [u for u in g.get_units() if u.team == 1 and not u.tower and not u.building and u.y < 19.0]
    if foes:
        return _reactive_defence(env, played)
    fin = _finish_tower(env)
    if fin is not None and mode != "chill":
        return fin
    hand, el = list(g.get_hand()), float(g.get_elixir())
    ts = tempo_state(env)
    # a press starts only when calm (no troop of theirs anywhere on the board: their push building at the back is
    # theirs to commit, ours to defend first)
    calm = not any(u.team == 1 and not u.tower and not u.building for u in g.get_units())
    press = mode == "press" or (mode is None and calm and ts["lead"] >= ts["threshold"] - 1e-6)
    mine = [u for u in g.get_units() if u.team == 0 and not u.tower and not u.building]
    lane = env.situation.spec.get("lane", LANE_X[0])
    if mine:
        lead_u = max(mine, key=lambda u: u.y)
        lane = LANE_X[0] if lead_u.x < 8.5 else LANE_X[1]
    bridge = float(env.MAX_Y)
    now = env.situation.seconds
    if press or "pressing" in played:
        played.add("pressing")  # once in, follow through: go in hard (plan: "go in hard when it's ahead"), every
        #                         affordable troop behind the win condition, a second apart
        if "wincon" not in played:
            for c in (BATTLE_RAM, GIANT):
                if c in hand and el >= COST[c]:
                    played.add("wincon")
                    played.add(f"@{now:.1f}")
                    return hand.index(c), lane, bridge
            return 4, 0.0, 0.0  # save for the win condition
        n_sup = sum(1 for p in played if p.startswith("sup"))
        if now - played_last(played) >= 1.0:
            for c in (MINI_PEKKA, MEGA_MINION, BOMBER, GIANT, BATTLE_RAM):
                if c in hand and el >= COST[c]:
                    played.add(f"sup{n_sup}")
                    played.add(f"@{now:.1f}")
                    return hand.index(c), lane, max(bridge - 2.0, 1.0)
        return 4, 0.0, 0.0
    # chill: hold until about to leak, then one cheap or back play (behind the survivors if there are any)
    if el < CHILL_AT:
        return 4, 0.0, 0.0
    if mine:
        for c in (BOMBER, MEGA_MINION):
            if c in hand and el >= COST[c]:
                played.add(f"@{now:.1f}")
                return hand.index(c), float(np.clip(lead_u.x, 0, 17)), float(np.clip(lead_u.y - 2.5, 1, 14))
    for c, pos in ((GOBLIN_HUT, HUT_PULL), (GIANT, (lane, 1.0)), (BOMBER, (lane, 3.0)), (MEGA_MINION, (lane, 3.0)),
                   (MINI_PEKKA, (lane, 3.0))):
        if c in hand and el >= COST[c]:
            played.add(f"@{now:.1f}")
            return hand.index(c), float(pos[0]), float(pos[1])
    return 4, 0.0, 0.0


def tempo_dump(env, played) -> tuple[int, float, float]:
    """The "dump everything" reference: every affordable troop at the bridge of the board's lane, as soon as it can."""
    g = env.game
    hand, el = list(g.get_hand()), float(g.get_elixir())
    ok = [(COST.get(c, 9), i) for i, c in enumerate(hand) if COST.get(c, 9) <= el and c not in (FIREBALL, ARROWS)]
    if not ok:
        return 4, 0.0, 0.0
    return max(ok)[1], env.situation.spec.get("lane", LANE_X[0]), float(env.MAX_Y)


TEMPLATES = [hog_rider, battle_ram, melee, supported_push, arrows_value, fireball_value, emergency,
             *REVIEW_TEMPLATES]  # Cole's 1 / 8 / 9 in training from 2026-10-28 (reviewed on the play page)
DEFENCE_TEMPLATES = list(TEMPLATES)  # the defence gate's ten
TEMPLATES = TEMPLATES + TEMPO_TEMPLATES  # plan P3, 2026-10-29
# the scripted opponent's deck (SCRIPT_DECK) in these; the full-game tempo starts play the pro clone with a pro deck
SCRIPTED_OPP = REVIEW_TEMPLATES + HELD_OUT_TEMPLATES

BY_NAME = {"hog_rider": hog_rider, "battle_ram": battle_ram, "melee_mini_pekka": melee, "melee_knight": melee,
           "melee_prince": melee, "supported_giant": supported_push, "supported_golem": supported_push,
           "arrows_value": arrows_value, "fireball_value": fireball_value, "emergency": emergency,
           "safe_opening": safe_opening, "defend_counter": defend_counter, "dont_overcommit": dont_overcommit,
           # the names a situation logs itself under (the play page's log), for 1 / 8 / 9
           "opening_tank_back": safe_opening, "opening_bridge_rush": safe_opening, "opening_support_back": safe_opening,
           "opening_wait_defend": safe_opening, "counter_giant": defend_counter, "counter_golem": defend_counter,
           "deficit": dont_overcommit, **{t.__name__: t for t in TEMPO_TEMPLATES + HELD_OUT_TEMPLATES}}


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
    return _reactive_defence(env, played)


def _reactive_defence(env, played) -> tuple[int, float, float]:
    """Answer whatever enemy troop is on our side, one card every 1.5 s at most (the emergency and review yardstick)."""
    if env.situation.seconds - played_last(played) < 1.5:
        return 4, 0.0, 0.0
    hand, el = list(env.game.get_hand()), env.game.get_elixir()
    # from the moment a troop is at the bridge (Cole 2026-10-28, 1-2: the Mini P.E.K.K.A went in at 3 s at y 13.8 in front
    # of the Hog; "should be played earlier and lower, like 11")
    foes = [u for u in env.game.get_units() if u.team == 1 and not u.tower and not u.building and u.y < 19.0]
    if not foes:
        return 4, 0.0, 0.0
    near = min(foes, key=lambda u: u.y)
    ground = [u for u in foes if not u.flying]
    for c in (MINI_PEKKA, MEGA_MINION, GOBLIN_HUT, BOMBER):
        if c not in hand or el < COST[c]:
            continue
        if c == MINI_PEKKA and ground:
            t = min(ground, key=lambda u: u.y)
            pos = (float(np.clip(t.x, 0, 17)), float(np.clip(min(t.y - 1.5, 11.0), 1, 11)))
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


def _review_defence(env, spec, played) -> tuple[int, float, float]:
    """The yardstick in Cole's 1 / 8 / 9: no opening play; answer what crosses. In defend-then-counter, Cole's
    supported-push defence first, then once the push is dead a Giant in front of what survived if we can afford it."""
    if spec["score"] == "counter":
        k = _supported_defence(env, spec, played)
        if k[0] != 4:
            return k
        sit_ = env.situation
        hand, el = list(env.game.get_hand()), env.game.get_elixir()
        mine = [u for u in env.game.get_units() if u.team == 0 and not u.tower and not u.building]
        if sit_.push_beaten_at is not None and "counter" not in played and mine and GIANT in hand and el >= COST[GIANT]:
            played.add("counter")
            lead = max(mine, key=lambda u: u.y)
            return hand.index(GIANT), float(LANE_X[0] if lead.x < 8.5 else LANE_X[1]), 14.0
    return _reactive_defence(env, played)


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
    if spec.get("tempo"):
        return tempo_policy(env, played)
    if spec.get("opp"):
        return _review_defence(env, spec, played)
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
        # the tempo situations' game state (plan P3): the clock, a tower of ours already down, a tower of theirs low,
        # and our survivors from a defence just made (team 0, hurt)
        if spec.get("tick") is not None:
            game.set_current_tick(int(spec["tick"]))
        if spec.get("destroy_own") is not None:
            game.destroy_tower(0, int(spec["destroy_own"]))
        for slot, hp in (spec.get("opp_tower_hp") or {}).items():
            game.set_tower_hp(1, int(slot), float(hp))
        if spec.get("own_units"):
            import clash_royale_env as cre
            for card, x, y, frac in spec["own_units"]:
                game.inject(int(card), float(x), float(y), 0, float(dict(cre.get_card_stats(int(card)))["hp"]) * frac, 0)
            if not spec.get("units"):
                game.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
        if not spec.get("drop"):
            self._inject(deploy_ticks=0)  # already on the move (the supported push starts mid-push)
        self.tower0 = [max(0, game.get_tower_hp(0, s)) for s in range(3)]
        # any standing Princess Tower's max (ours on the left reads -1 once it's destroyed: the desperate template)
        self.princess_max = float(max(game.get_tower_max_hp(t, sl) for t in (0, 1) for sl in (1, 2)))
        self.etower0 = [max(0, game.get_tower_hp(1, s)) for s in range(3)]
        self.opp_state: dict = {}
        self.push_beaten_at: float | None = None  # defend_counter: when the starting push was dead
        self.lost_at_beaten = 0.0
        self.spent0 = float(game.get_elixir_spent(0))
        self.dt = dt  # seconds per env step (1.0, or 0.1 at a decision a tick)
        self.seconds = 0.0
        self.neutral_since: float | None = None
        self.key: list[list[tuple]] = []  # Cole's takes on this start: [(t, card, x, y), ...] each
        self.my_plays: list[tuple] = []  # ours, the same way
        self.leaked = 0.0  # P2: elixir lost at the cap
        self.wasted_spell_elixir = 0.0  # P2: cost of our Fireballs / Arrows that touched nothing (CloneEnv books it)
        self.wasted_spells = 0
        self._el = float(game.get_elixir())
        self._spent = self.spent0
        if spec.get("full_game"):
            self.mode = spec["mode"]
            self.modes = [(0.0, self.mode)]  # (seconds, tempo) each time it changes
            self.parts: dict[str, float] = {}  # reward so far by where it came from
            self.total = 0.0
            self.passed: dict[str, bool] = {}  # desperate / spell_finish: pass or fail, once paid
            self._lost_t, self._dealt_t, self._adv = 0.0, 0.0, self.advantage()

    def advantage(self) -> float:
        """Our elixir advantage: (bank + board) ours - theirs, the board at HP share (every unit and building of each
        side but towers, anywhere). Playing a card moves elixir from bank to board (no change); a good trade, or their
        bar leaking while ours fills, moves it our way."""
        import clash_royale_env as cre
        side = [float(self.game.get_elixir_for_team(0)), float(self.game.get_elixir_for_team(1))]
        for u in self.game.get_units():
            if u.tower or u.hp <= 0 or u.card_id < 0 or u.attached_to >= 0:
                continue
            st = dict(cre.get_card_stats(int(u.card_id)))
            worth = COST.get(u.card_id, st.get("cost", 0) or 0) / max(1, int(st.get("bodies", 1) or 1))
            side[u.team] += worth * min(1.0, u.hp / max(float(st.get("hp", u.hp) or u.hp), 1.0))
        return side[0] - side[1]

    def tempo_step(self, game_over: bool, result: float = 0.0) -> tuple[float, bool, dict]:
        """After each env step of a full-game tempo start: (this step's reward, the game is over, details). `result`
        is the game's own (+1 / -1 / 0) on its last step."""
        self.seconds += self.dt
        self.leaked += leak_step(self.game, 0, self._el, self._spent, int(round(self.dt * 10)))
        self._el, self._spent = float(self.game.get_elixir()), float(self.game.get_elixir_spent(0))
        g, pm = self.game, self.princess_max
        lost = self._lost()
        dealt = sum(max(0.0, a - max(0, g.get_tower_hp(1, s))) for s, a in enumerate(self.etower0))
        adv = self.advantage()
        d_lost, d_dealt, d_adv = lost - self._lost_t, dealt - self._dealt_t, adv - self._adv
        self._lost_t, self._dealt_t, self._adv = lost, dealt, adv
        got: dict[str, float] = {}
        if self.mode in ("chill", "even"):
            got[self.mode] = -d_lost / pm + ADVANTAGE_COEF * d_adv
        elif self.mode == "press":
            got["press"] = (d_dealt - d_lost) / pm
        tick = int(g.get_current_tick())
        if self.mode == "spell_finish":
            felled = any(a > 0 and g.get_tower_hp(1, s) <= 0 for s, a in enumerate(self.etower0) if s)
            spelled = any(c in (FIREBALL, ARROWS) for _, c, _, _ in self.my_plays)
            if felled and spelled:
                self.passed["spell_finish"] = True
            elif self.seconds >= self.spec["timeout"] - 1e-6 or game_over:
                self.passed["spell_finish"] = False
            if "spell_finish" in self.passed:
                got["spell_finish"] = PASS_FAIL if self.passed["spell_finish"] else -PASS_FAIL
                self._switch("even")
        if self.mode == "desperate" and (tick >= REGULATION_END_TICK or game_over):
            # overtime reached (the crowns level at 3:00), or the game won: pass
            self.passed["desperate"] = (not game_over) or result > 0
            got["desperate"] = PASS_FAIL if self.passed["desperate"] else -PASS_FAIL
            self._switch("even")
        if game_over:
            got["result"] = RESULT_COEF * result
        else:
            self._next_mode(tick)
        r = 0.0
        for k, v in got.items():
            self.parts[k] = self.parts.get(k, 0.0) + v
            r += v
        self.total += r
        return r, game_over, (self.details(result) if game_over else {})

    def _switch(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.modes.append((round(self.seconds, 1), mode))

    def _next_mode(self, tick: int) -> None:
        g = self.game
        if self.mode in ("desperate", "spell_finish"):
            return
        if tick >= LATE_TICK and tick < REGULATION_END_TICK and _ours_alive(g, 0) < _ours_alive(g, 1) \
                and "desperate" not in self.passed:
            self._switch("desperate")
            return
        lead = float(g.get_elixir()) - float(g.get_elixir_for_team(1))
        if self.mode in ("chill", "even") and lead >= PRESS_LEAD - 1e-6:
            self._switch("press")
        elif self.mode == "press":
            # a press holds while our push is on the board (spending on it drops the lead), or PRESS_HOLD s from its
            # start to begin one; it's over once the lead is gone and none of our troops is left
            mine = any(u.team == 0 and not u.tower and not u.building and u.hp > 0 for u in g.get_units())
            since = self.seconds - self.modes[-1][0]
            if lead < PRESS_LEAD - 1e-6 and not mine and since >= PRESS_HOLD:
                self._switch("even")

    def details(self, result: float = 0.0) -> dict:
        lost = self._lost()
        dealt = sum(max(0.0, a - max(0, self.game.get_tower_hp(1, s))) for s, a in enumerate(self.etower0))
        return {"name": self.spec["name"], "score": float(self.total), "full_game": True, "result": result,
                "parts": {k: round(v, 4) for k, v in self.parts.items()}, "modes": self.modes, "mode": self.mode,
                "passed": dict(self.passed), "tower_hp_lost": lost, "tower_hp_dealt": dealt,
                "elixir_advantage": round(self._adv, 2), "elixir_spent": float(self.game.get_elixir_spent(0)) - self.spent0,
                "elixir_leaked": round(self.leaked, 2), "seconds": round(self.seconds, 1),
                "clock": round(int(self.game.get_current_tick()) * 0.1, 1), "elixir_end": float(self.game.get_elixir()),
                "wasted_spells": self.wasted_spells, "wasted_spell_elixir": self.wasted_spell_elixir,
                "opp_plays": self.opp_state.get("plays", []), "unresolved": False,
                "key_match": self.key_match() if self.key else None}

    def _board(self) -> tuple[float, float]:
        """Elixir on the board at HP share: ours anywhere (buildings too), theirs on our side (towers aside)."""
        import clash_royale_env as cre
        ours = theirs = 0.0
        for u in self.game.get_units():
            if u.tower or u.hp <= 0 or u.card_id < 0 or (u.team == 1 and u.y >= 17.5):
                continue
            st = dict(cre.get_card_stats(int(u.card_id)))
            worth = COST.get(u.card_id, st.get("cost", 0) or 0) / max(1, int(st.get("bodies", 1) or 1))
            v = worth * min(1.0, u.hp / max(float(st.get("hp", u.hp) or u.hp), 1.0))
            if u.team == 0:
                ours += v
            else:
                theirs += v
        return ours, theirs

    def key_match(self) -> float:
        """The best share of one of Cole's takes on this start that our plays matched (each of ours used once)."""
        best = 0.0
        for take in self.key:
            used, hit = set(), 0
            for t, card, x, y in take:
                for i, (t2, c2, x2, y2) in enumerate(self.my_plays):
                    if i not in used and c2 == card and abs(t2 - t) <= KEY_DT and (x2 - x) ** 2 + (y2 - y) ** 2 <= KEY_DIST ** 2:
                        used.add(i)
                        hit += 1
                        break
            best = max(best, hit / max(1, len(take)))
        return best

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
        self.leaked += leak_step(self.game, 0, self._el, self._spent, int(round(self.dt * 10)))
        self._el, self._spent = float(self.game.get_elixir()), float(self.game.get_elixir_spent(0))
        if self.spec.get("drop") and not self.threat and self.seconds >= 1.0 - 1e-6 and not game_over:
            # the board starts empty; the opponent drops the card now, with the normal 1 s deploy, as in a real game
            self._inject(deploy_ticks=-1)
            self.just_dropped = True
            return False, 0.0, {}
        if self.spec.get("drop") and not self.threat:
            return (True, 0.0, {}) if game_over else (False, 0.0, {})  # before the drop (the first second)
        neutral = self.resolved()
        if self.spec.get("window"):
            # the review situations run the whole window (the opponent keeps playing); remember when the starting push
            # (if any) was beaten and what it had cost us by then
            if self.threat and self.push_beaten_at is None and not (self.threat & {u.id for u in self.game.get_units()}):
                self.push_beaten_at = self.seconds
                self.lost_at_beaten = self._lost()
            done = game_over or self.seconds >= self.spec["timeout"] - 1e-6
            return (self._finish() if done else (False, 0.0, {}))
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
        return self._finish()

    def _lost(self) -> float:
        return sum(max(0.0, a - max(0, self.game.get_tower_hp(0, s))) for s, a in enumerate(self.tower0))

    def _finish(self) -> tuple[bool, float, dict]:
        lost = sum(max(0.0, a - max(0, self.game.get_tower_hp(0, s))) for s, a in enumerate(self.tower0))
        spent = float(self.game.get_elixir_spent(0)) - self.spent0
        unresolved = not self.resolved()
        survivors = sum(COST.get(u.card_id, 0) for u in self.game.get_units() if u.team == 0 and not u.tower and u.hp > 0)
        kind = self.spec.get("score")
        el_end = float(self.game.get_elixir())
        dealt = sum(max(0.0, a - max(0, self.game.get_tower_hp(1, s))) for s, a in enumerate(self.etower0))
        ours, theirs = self._board()
        pm = self.princess_max
        # the board term every kind shares now (v2, 2026-10-28): half a refund for what we have left, their value left
        # on our side, a little for tower damage
        board = BOARD_REFUND * min(ours, 10.0) - min(THREAT_LEFT_CAP, THREAT_LEFT_COEF * theirs) + DEALT_COEF * dealt / pm
        if kind in ("press", "desperate"):
            # plan P3: over the window, damage dealt counts as much as damage taken (a tower of theirs that fell counts
            # as a whole Princess Tower: Cole's "spell their tower" finish), minus what we spent (not when desperate:
            # elixir has no later use) and the board. Leak and wasted spells come off below, as for every kind
            felled = sum(max(0.0, pm - a) for s, a in enumerate(self.etower0) if s and a > 0 and self.game.get_tower_hp(1, s) <= 0)
            gain = DEALT_COEF_PRESS * (dealt + felled) / pm
            score = gain - lost / pm - (0.0 if kind == "desperate" else ELIXIR_COST * spent) + board
            v1 = score
            unresolved = any(u.team == 1 and not u.tower and not u.building and u.y < 17.5 for u in self.game.get_units())
        elif kind in ("opening", "deficit", "counter", "tempo"):
            # towers first; then the elixir we end on and the board. In defend-then-counter the damage we deal counts
            # extra once the push was beaten inside the tolerance (Cole's 8)
            threat_left = any(u.team == 1 and not u.tower and not u.building and u.y < 17.5 for u in self.game.get_units())
            v1 = (-lost / pm + 0.02 * min(el_end, 10.0) + SURVIVOR_BONUS * min(survivors, 10)
                  - (TIMEOUT_PENALTY if threat_left else 0.0))
            score = -lost / pm + 0.02 * min(el_end, 10.0) + board
            if kind == "counter":
                ok = self.push_beaten_at is not None and self.lost_at_beaten <= COUNTER_TOLERANCE
                extra = (0.5 * dealt / pm) if ok else 0.0
                never = 0.0 if self.push_beaten_at is not None else TIMEOUT_PENALTY
                v1 += extra - never
                score += extra - never
            unresolved = threat_left
        elif kind == "trade":
            # Cole's spell scoring: enemy elixir value removed (share of each body's max HP taken, capped at what it
            # had, plus ELIM_BONUS of its value if killed) against what we spent; towers still first
            alive = {u.id: u for u in self.game.get_units()}
            value = 0.0
            for uid, (worth, max_hp, hp0) in getattr(self, "threat_info", {}).items():
                hp1 = alive[uid].hp if uid in alive else 0.0
                value += worth * min(1.0, max(0.0, hp0 - hp1) / max(max_hp, 1.0)) + (ELIM_BONUS * worth if uid not in alive else 0.0)
            v1 = -lost / pm + TRADE_COEF * (value - spent) - (TIMEOUT_PENALTY if unresolved else 0.0)
            score = -lost / pm + TRADE_COEF * (value - spent) + board
        else:
            v1 = (-lost / pm - ELIXIR_COST * spent + SURVIVOR_BONUS * min(survivors, 10)
                  - (TIMEOUT_PENALTY if unresolved else 0.0))
            score = -lost / pm - ELIXIR_COST * spent + board
        score -= LEAK_COEF * self.leaked + WASTED_SPELL_COEF * self.wasted_spell_elixir
        return True, float(score), {"tower_hp_lost": lost, "elixir_leaked": round(self.leaked, 2),
                                    "wasted_spells": self.wasted_spells, "wasted_spell_elixir": self.wasted_spell_elixir, "elixir_spent": spent, "unresolved": unresolved,
                                    "seconds": round(self.seconds, 1), "tower_hp_dealt": dealt, "elixir_end": el_end,
                                    "opp_plays": self.opp_state.get("plays", []), "score_v1": float(v1),
                                    "board_ours": round(ours, 2), "board_theirs": round(theirs, 2),
                                    "key_match": self.key_match() if self.key else None}
