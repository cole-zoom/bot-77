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

COST = {HOG: 4, BATTLE_RAM: 4, MINI_PEKKA: 4, KNIGHT: 3, PRINCE: 5, GIANT: 5, GOLEM: 8, MUSKETEER: 4, WIZARD: 5,
        EXECUTIONER: 5, MINIONS: 3, MEGA_MINION: 3, BOMBER: 2, GOBLIN_HUT: 4, FIREBALL: 4, ARROWS: 3}

LANE_X = (3.0, 14.0)  # the princess-tower lanes (engine x; bridges at 2.5 / 14.5)
RIVER_Y = 17.0  # inside the river band: a unit spawned here walks out of the bridge (the fork's scenarios do the same)

ELIXIR_COST = 0.04
SURVIVOR_BONUS = 0.02
TIMEOUT_PENALTY = 0.3
MIN_SECONDS = 3  # don't call the board neutral before the threat has had a chance to arrive


def _lane(rng) -> float:
    return float(rng.choice(LANE_X))


def hog_rider(rng) -> dict:
    """Cole's 2: a solo Hog Rider at the bridge; 4-6 elixir; a building pull or a troop answer in hand."""
    x = _lane(rng)
    return {"name": "hog_rider", "timeout": 20, "own_elixir": rng.uniform(4, 6),
            "opp_elixir": rng.uniform(0, 3), "units": [(HOG, x + rng.uniform(-0.4, 0.4), RIVER_Y + rng.uniform(-0.5, 0.5))],
            "answers": [MINI_PEKKA, GOBLIN_HUT, MEGA_MINION]}  # Hut first failed to pull (x 8.5, y 9): Cole to place it


def battle_ram(rng) -> dict:
    """Cole's 3: a Battle Ram from the bridge at different distances; 4-6 elixir; scored over the whole attack
    (the Barbarians it drops included)."""
    x = _lane(rng)
    return {"name": "battle_ram", "timeout": 26, "own_elixir": rng.uniform(4, 6),
            "opp_elixir": rng.uniform(0, 3), "units": [(BATTLE_RAM, x, rng.uniform(16.0, 19.5))],
            "answers": [MINI_PEKKA, GOBLIN_HUT, MEGA_MINION], "answers2": [BOMBER, MEGA_MINION]}


def melee(rng) -> dict:
    """Cole's 4: a Mini P.E.K.K.A, Knight or Prince approaching from the bridge; 3-5 elixir; cheap distraction in
    hand (the variant is kept in the name: one defence doesn't work equally against all three)."""
    card = int(rng.choice([MINI_PEKKA, KNIGHT, PRINCE]))
    x = _lane(rng)
    name = {MINI_PEKKA: "melee_mini_pekka", KNIGHT: "melee_knight", PRINCE: "melee_prince"}[card]
    return {"name": name, "timeout": 20, "own_elixir": rng.uniform(3, 5),
            "opp_elixir": rng.uniform(0, 3), "units": [(card, x + rng.uniform(-0.4, 0.4), rng.uniform(15.5, 18.5))],
            "answers": [BOMBER, MEGA_MINION, GOBLIN_HUT] if card == KNIGHT else [MEGA_MINION, GOBLIN_HUT, BOMBER]}


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
    sup_answers = [ARROWS, MEGA_MINION] if sup == MINIONS else [FIREBALL, MEGA_MINION, BOMBER]
    return {"name": name, "timeout": timeout, "own_elixir": rng.uniform(6, 8),
            "opp_elixir": rng.uniform(0, 2), "units": [(tank, x, ty), (sup, x + rng.uniform(-0.5, 0.5), ty + rng.uniform(2.0, 3.5))],
            "answers": [MINI_PEKKA, GOBLIN_HUT], "answers2": sup_answers}


TEMPLATES = [hog_rider, battle_ram, melee, supported_push]


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
                return i, 8.5, 9.0
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

    def __init__(self, game, spec: dict, deck: list[int], rng):
        self.game, self.spec = game, spec
        game.set_elixir_for_team(0, float(spec["own_elixir"]))
        game.set_elixir_for_team(1, float(spec["opp_elixir"]))
        self.hand = deal_hand(rng, deck, spec)
        self.hand_ok = bool(game.set_hand_for_team(0, self.hand))
        before = {u.id for u in game.get_units()}
        for card, x, y in spec["units"]:
            game.inject(int(card), float(x), float(y), 1, -1.0, 0)  # deploy_ticks 0: already on the move
        noop = 4
        game.step_self_play(noop, 0.0, 0.0, noop, 0.0, 0.0, 1)  # commit the injected units
        self.threat = {u.id for u in game.get_units() if u.id not in before and u.team == 1}
        self.tower0 = [max(0, game.get_tower_hp(0, s)) for s in range(3)]
        self.princess_max = float(game.get_tower_max_hp(0, 1))
        self.spent0 = float(game.get_elixir_spent(0))
        self.steps = 0
        self.neutral_streak = 0

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
        self.steps += 1
        neutral = self.resolved()
        self.neutral_streak = self.neutral_streak + 1 if neutral else 0
        # neutral for 2 s in a row: a Battle Ram's Barbarians appear ~1 s after it breaks, and the board looked empty
        # for that second (2026-10-28)
        done = game_over or self.steps >= self.spec["timeout"] or (self.steps >= MIN_SECONDS and self.neutral_streak >= 2)
        if not done:
            return False, 0.0, {}
        lost = sum(max(0.0, a - max(0, self.game.get_tower_hp(0, s))) for s, a in enumerate(self.tower0))
        spent = float(self.game.get_elixir_spent(0)) - self.spent0
        unresolved = not self.resolved()
        survivors = sum(COST.get(u.card_id, 0) for u in self.game.get_units() if u.team == 0 and not u.tower and u.hp > 0)
        score = (-lost / self.princess_max - ELIXIR_COST * spent + SURVIVOR_BONUS * min(survivors, 10)
                 - (TIMEOUT_PENALTY if unresolved else 0.0))
        return True, float(score), {"tower_hp_lost": lost, "elixir_spent": spent, "unresolved": unresolved,
                                    "seconds": self.steps}
