"""One simulated match, driven and read in our frame (docs/simulator.md S1, 2026-10-23). The only module that talks
to the engine (`clash_royale_env`, the ClashRoyaleEnv fork in third_party/clash-royale-env).

Teams are 'own' (the bottom / deciding player, engine team 0) and 'opp'. Positions are ground points (feet, taps) in
our frame (bot77.sim.geometry). The engine runs at 10 ticks a second; `t` is the game clock in seconds since the start.

Plays go through the engine's own playCard (placement rules, deploy time, elixir, card cycle). A replay's play may
not be playable as the engine sees it (the card not in its hand, elixir short, a spot the engine calls illegal):
`play(force=True)` makes it playable and says what it changed (`PlayResult`), so callers can count it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import clash_royale_env as cre

from bot77.sim import cards as sc
from bot77.sim.geometry import OPP_TEAM, OWN_TEAM, Y_SCALE, engine_to_ours, mirror_engine_y, ours_to_engine

TICKS_PER_S = 10
MAX_TICKS = 3000  # 3:00 regulation + 2:00 overtime
TEAM_NAME = {OWN_TEAM: "own", OPP_TEAM: "opp"}
TOWER_SLOTS = {"king": 0, "l": 1, "r": 2}  # board coordinates for both teams (left = our left)
# Tower stats (max hp, damage per shot, cooldown ticks) for (king, princess). The engine's cards are level 11 (Knight
# 1766 hp / 202 = RoyaleAPI cr-api-data level 11); its default towers are level 9. Chosen 2026-10-24: liquipedia_l11.
# Measured on replays (tower HP OCR): a lone Giant / Knight / Royal Ghost hit takes 0.0832 / 0.0665 / 0.0867 of a
# princess tower; engine damage / 3052 gives 0.0829 / 0.0662 / 0.0855, / 3584 gives 0.071 / 0.056 / 0.073.
TOWER_PRESETS = {
    "engine": None,  # King 4008 / 90 / 1.0 s, Princess 2534 / 90 / 0.8 s
    # RoyaleAPI cr-api-data (game-file extract) at the cards' level: KingTower / PrincessTower + their projectiles
    "royaleapi_l11": ((6144, 128, 10), (3584, 128, 8)),
    # Liquipedia's level-11 Tower Princess (3052 / 109 / 0.8 s); King scaled by cr-api-data's King/Princess ratio
    "liquipedia_l11": ((5232, 109, 10), (3052, 109, 8)),
}


def legal_deck(deck: list[int]) -> list[int]:
    """8 engine ids in an order the engine accepts (an Evolution in slot 0, a Champion / Hero in slot 1, one more of
    either in the wild-card slot 2). Raises if no order fits."""
    info = sc.engine_cards()
    evo = [c for c in deck if info[c]["evolution"]]
    special = [c for c in deck if info[c]["is_champion"] or info[c]["is_hero"]]
    plain = [c for c in deck if c not in evo and c not in special]
    slot0 = evo[:1] or plain[:1]
    slot1 = special[:1] or [c for c in plain if c not in slot0][:1]
    rest = [c for c in deck if c not in slot0 + slot1]
    wild = [c for c in rest if c in evo or c in special][:1] or rest[:1]
    order = slot0 + slot1 + wild + [c for c in rest if c not in wild]
    err = cre.validate_deck_slots(order)
    if err:
        raise ValueError(f"no legal slot order for {deck}: {err}")
    return order


def fill_deck(known: list[int], filler: list[int] | None = None) -> list[int]:
    """Pad a partly known deck to 8 with plain cards not in it (cheap ones first), so the engine can be built. The
    filler cards are never played; they only sit in the queue."""
    out = list(dict.fromkeys(known))[:8]
    pool = filler or sorted((i for i, v in sc.engine_cards().items()
                             if not v["evolution"] and not v["is_champion"] and not v["is_hero"]),
                            key=lambda i: (sc.cost(i), i))
    for c in pool:
        if len(out) == 8:
            break
        if c not in out:
            out.append(c)
    return out


@dataclass
class PlayResult:
    played: bool
    card: int  # engine id
    team: str
    t: float
    x: float  # where it went, our frame
    y: float
    moved: float = 0.0  # tiles the spot was shifted to the nearest legal one
    elixir_added: float = 0.0  # elixir given so the play was affordable
    hand_changed: bool = False  # the card was put into the hand
    reason: str = ""


@dataclass
class Sim:
    env: "cre.ClashRoyaleEnv"
    log: list[PlayResult] = field(default_factory=list)

    @classmethod
    def new(cls, own_deck: list[int], opp_deck: list[int], own_tower: str = "queen-tower",
            opp_tower: str = "queen-tower", max_ticks: int = MAX_TICKS, towers: str = "liquipedia_l11") -> "Sim":
        """`towers`: a TOWER_PRESETS name, applied to both sides' King and (default) princess towers after reset."""
        env = cre.ClashRoyaleEnv(legal_deck(fill_deck(own_deck)), legal_deck(fill_deck(opp_deck)), max_ticks,
                                 sc.TOWER_TROOPS[own_tower], sc.TOWER_TROOPS[opp_tower])
        env.seed(0)
        sim = cls(env)
        sim.towers_preset = towers
        sim.troops = (own_tower, opp_tower)
        return sim

    def reset(self) -> None:
        """Start the match (and level the towers per the preset)."""
        self.env.reset()
        preset = TOWER_PRESETS[getattr(self, "towers_preset", "liquipedia_l11")]
        if preset:
            for tm in (OWN_TEAM, OPP_TEAM):
                self.env.set_tower_stats(tm, 0, *preset[0])
                if sc.TOWER_TROOPS[self.troops[tm]] == cre.TowerTroopType.NONE:
                    for slot in (1, 2):
                        self.env.set_tower_stats(tm, slot, *preset[1])

    def snapshot(self) -> "Sim":
        s = Sim(self.env.snapshot(), list(self.log))
        s.towers_preset, s.troops = getattr(self, "towers_preset", "liquipedia_l11"), getattr(self, "troops", None)
        return s

    # --- clock, elixir, hand ---
    @property
    def tick(self) -> int:
        return self.env.get_current_tick()

    @property
    def t(self) -> float:
        return self.tick / TICKS_PER_S

    def set_clock(self, t: float) -> None:
        self.env.set_current_tick(round(t * TICKS_PER_S))

    def elixir(self, team: str) -> float:
        return self.env.get_elixir_for_team(_team(team))

    def set_elixir(self, team: str, value: float) -> None:
        self.env.set_elixir_for_team(_team(team), float(value))

    def hand(self, team: str) -> list[int]:
        return list(self.env.get_hand_for_team(_team(team)))

    def set_hand(self, team: str, hand: list[int]) -> bool:
        """Engine ids; cards missing from a short read are filled from the deck. The queue order (`next`) can't be
        set through the engine's API: the remaining cards keep their order."""
        cur = self.hand(team)
        want = [c for c in hand if c is not None][: len(cur)]
        for c in cur:
            if len(want) == len(cur):
                break
            if c not in want:
                want.append(c)
        return self.env.set_hand_for_team(_team(team), want)

    # --- towers ---
    def towers(self) -> dict[str, dict]:
        """{'own_king': {'hp': 4008, 'max': 4008, 'frac': 1.0}, ...}; a destroyed tower reads hp 0."""
        out = {}
        for team in (OWN_TEAM, OPP_TEAM):
            for name, slot in TOWER_SLOTS.items():
                hp, mx = self.env.get_tower_hp(team, slot), self.env.get_tower_max_hp(team, slot)
                hp = max(hp, 0)
                out[f"{TEAM_NAME[team]}_{name}"] = {"hp": hp, "max": mx, "frac": hp / mx if mx > 0 else 0.0}
        return out

    def set_tower(self, team: str, slot: str, frac: float) -> bool:
        """Tower HP as a fraction of the engine tower's (fixed-level) max: real accounts' towers differ by level, so
        only fractions carry over. 0 destroys it (crown, king wake-up, lane retargeting)."""
        tm, s = _team(team), TOWER_SLOTS[slot]
        if frac <= 0:
            return self.env.destroy_tower(tm, s)
        return self.env.set_tower_hp(tm, s, max(1.0, frac * self.env.get_tower_max_hp(tm, s)))

    def wake_king(self, team: str) -> bool:
        """Wake a sleeping King the way real damage does (hp just below max: Tower::update latches on hp < maxHp)."""
        tm = _team(team)
        mx = self.env.get_tower_max_hp(tm, 0)
        hp = self.env.get_tower_hp(tm, 0)
        return hp > 0 and self.env.set_tower_hp(tm, 0, min(hp, mx - 1))

    # --- units ---
    def inject(self, card: int, x: float, y: float, team: str, hp: float | None = None, deployed: bool = True) -> None:
        """Put a unit on the board as if it were already there (no elixir, no cycle). `deployed` skips the 1 s deploy
        time, for units already seen on the field; hp None = full."""
        ex, ey = ours_to_engine(x, y)
        self.env.inject(card, ex, ey, _team(team), -1.0 if hp is None else float(hp), 0 if deployed else -1)

    def units(self, towers: bool = False) -> list[dict]:
        """Live units in our frame: card (our name), engine_name, card_id (engine, -1 for spawned bodies), team, x, y
        (feet), hp, flying, building, tower, deploying, id, target."""
        out = []
        for u in self.env.get_units():
            if u.tower and not towers:
                continue
            x, y = engine_to_ours(u.x, u.y)
            out.append({"id": u.id, "card": sc.our_name(u.card_id) if u.card_id >= 0 else u.name,
                        "engine_name": u.name, "card_id": u.card_id, "team": TEAM_NAME[u.team], "x": x, "y": y,
                        "hp": u.hp, "flying": u.flying, "building": u.building, "tower": u.tower,
                        "spell": not u.targetable, "deploying": u.deploy_ticks > 0, "target": u.target_id})
        return out

    # --- playing and stepping ---
    def play(self, card: int, x: float, y: float, team: str, t: float | None = None, force: bool = True) -> PlayResult:
        """Play `card` (engine id) at our (x, y) at game time `t` (default: now; the sim first runs up to it), then
        advance one tick. With force, the card is put in the hand and elixir topped up when needed, and an illegal
        spot moves to the nearest legal one (within 2 of our tiles)."""
        if t is not None and t > self.t:
            self.run_to(t)
        tm = _team(team)
        ex, ey = ours_to_engine(x, y)
        res = PlayResult(False, card, team, self.t, x, y)
        hand = self.hand(team)
        if card not in hand:
            if not force:
                return self._skip(res, "not in hand")
            hand[-1] = card  # replace one slot; set_hand keeps the rest
            if not self.env.set_hand_for_team(tm, hand):
                return self._skip(res, "not in deck")
            res.hand_changed = True
            hand = self.hand(team)
        if not self.env.is_valid_placement(card, ex, ey, tm):
            spot = _nearest_legal(self.env, card, ex, ey, tm) if force else None
            if spot is None:
                return self._skip(res, "illegal spot")
            res.moved = ((spot[0] - ex) ** 2 + ((spot[1] - ey) * Y_SCALE) ** 2) ** 0.5  # our units
            ex, ey = spot
            res.x, res.y = engine_to_ours(ex, ey)
        short = sc.cost(card) - self.elixir(team)
        if short > 0:
            if not force:
                return self._skip(res, "elixir")
            self.set_elixir(team, self.elixir(team) + short)
            res.elixir_added = short
        spent = self.env.get_elixir_spent(tm)
        i = hand.index(card)
        a0 = (i, ex, ey) if tm == OWN_TEAM else (-1, 0.0, 0.0)
        a1 = (i, ex, mirror_engine_y(ey)) if tm == OPP_TEAM else (-1, 0.0, 0.0)
        self.env.step_self_play_fast(*a0, *a1, skip_frames=1)
        res.played = self.env.get_elixir_spent(tm) > spent
        if not res.played:
            res.reason = "engine refused"
        self.log.append(res)
        return res

    def step(self, ticks: int = 1) -> bool:
        """Advance with no plays; True if the match is over."""
        if ticks > 0:
            return self.env.step_self_play_fast(-1, 0.0, 0.0, -1, 0.0, 0.0, skip_frames=ticks).done
        return self.over

    def run_to(self, t: float) -> bool:
        return self.step(round(t * TICKS_PER_S) - self.tick)

    @property
    def over(self) -> bool:
        return self.env.is_game_over()

    def _skip(self, res: PlayResult, reason: str) -> PlayResult:
        res.reason = reason
        self.log.append(res)
        return res


def _team(team: str) -> int:
    return {"own": OWN_TEAM, "opp": OPP_TEAM}[team]


def _nearest_legal(env, card: int, ex: float, ey: float, tm: int, radius: float = 2.0) -> tuple[float, float] | None:
    """The closest legal engine point within `radius` of OUR tiles (engine rows are Y_SCALE of our units)."""
    best = None
    n = int(radius / 0.25)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx, dy = i * 0.25, j * 0.25  # our units
            d = dx * dx + dy * dy
            if d <= radius * radius and (best is None or d < best[0]) and \
                    env.is_valid_placement(card, ex + dx, ey + dy / Y_SCALE, tm):
                best = (d, ex + dx, ey + dy / Y_SCALE)
    return None if best is None else (best[1], best[2])
