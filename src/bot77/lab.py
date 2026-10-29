"""Lab battles (docs/plan-sim-1to1.md P0.4, 2026-10-29): friendly battles between Cole's phone and the BlueStacks
account, both sides placing scripted cards on scripted tiles at scripted times, recorded from the Battle Log and
compared with the simulator running the same placements.

A test is one YAML file in config/lab/ (`load_tests`):

  id: L01                       # file name starts with it; tests run in id order (priority)
  name: Mini P.E.K.K.A mirror
  mechanic: "P2 #2 attack cycle"
  settles: what the outcome decides
  lane: left | right            # the test's lane; the BlueStacks side cycles unwanted cards into the other one
  decks: {bluestacks: [8 names], cole: [8 names]}
  placements:                   # both sides, any order
    - {side: bluestacks | cole, card: name, tile: [x, y], t: 30, cue: "optional words for Cole"}
  watch: [what to look at afterwards]
  cole_notes: free text for the cheat sheet
  tail_s: seconds to keep watching after the last placement (default 45)

Tiles are **engine cells** in the BlueStacks player's frame (the play page's grid, scripts/situation_play.py): x 0-17
left to right, y 0-33 from the BlueStacks king (y 0) to Cole's (y 33); river rows 16-17; the BlueStacks side places on
y <= 15, Cole's on y >= 18 (the engine's halves, as the play page draws them). In a replay recorded on the BlueStacks account it is the bottom player, Cole the top.
`t` is game time in seconds since the battle clock started (the clock shows 3:00 - t).

Cole sees his own side at the bottom. Whether his screen is the BlueStacks view turned 180 degrees (his left lane is
the BlueStacks right lane) or flipped top-to-bottom (left stays left) is checked first thing in the session (test L00);
`cole_tile` takes either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

LAB_DIR = Path("config/lab")
SIDES = ("bluestacks", "cole")
ENGINE_W, ENGINE_H = 18, 34
OWN_MAX_Y = 15  # the BlueStacks (engine team 0) side's last legal row for troops / buildings (the engine and the play page)
COLE_MIN_Y = ENGINE_H - 1 - OWN_MAX_Y  # 18
BOMBER_DECK = ["Bomber", "Battle Ram", "Giant", "Mega Minion", "Goblin Hut", "Mini P.E.K.K.A", "Fireball", "Arrows"]
SPELLS = {"Fireball", "Arrows", "Zap", "The Log", "Rocket", "Lightning", "Poison", "Freeze", "Rage", "Tornado",
          "Earthquake", "Giant Snowball", "Barbarian Barrel", "Royal Delivery", "Goblin Barrel", "Graveyard", "Clone",
          "Mirror", "Void", "Goblin Curse"}
ELIXIR_S = 2.8  # seconds per elixir at 1x (the replays measured 2.90; budgeting only)


@dataclass
class Placement:
    side: str
    card: str
    tile: tuple[int, int]
    t: float
    cue: str = ""
    i: int = 0  # index in the test's placement list

    @property
    def spell(self) -> bool:
        return self.card in SPELLS


@dataclass
class LabTest:
    id: str
    name: str
    mechanic: str
    settles: str
    lane: str
    decks: dict[str, list[str]]
    placements: list[Placement]
    watch: list[str] = field(default_factory=list)
    cole_notes: str = ""
    tail_s: float = 45.0
    path: Path | None = None

    def side(self, side: str) -> list[Placement]:
        return sorted((p for p in self.placements if p.side == side), key=lambda p: (p.t, p.i))

    @property
    def end_t(self) -> float:
        return max(p.t for p in self.placements) + self.tail_s

    def dump_spots(self) -> dict[str, tuple[int, int]]:
        """Where the BlueStacks side cycles cards it doesn't need (engine tiles): troops and buildings at the back
        of the other lane, spells into its own back corner there (no tower or unit in reach)."""
        x = 15 if self.lane == "left" else 2  # row 0 is not legal in the engine; row 1 is
        return {"troop": (x, 1), "spell": (17 if x > 9 else 0, 2)}


def load_test(path: Path) -> LabTest:
    d = yaml.safe_load(Path(path).read_text())
    ps = [Placement(p["side"], p["card"], tuple(p["tile"]), float(p["t"]), p.get("cue", ""), i)
          for i, p in enumerate(d["placements"])]
    t = LabTest(str(d["id"]), d["name"], d["mechanic"], d["settles"], d.get("lane", "left"), d["decks"], ps,
                d.get("watch", []), d.get("cole_notes", ""), float(d.get("tail_s", 45.0)), Path(path))
    problems = check(t)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return t


def load_tests(lab_dir: Path = LAB_DIR) -> dict[str, LabTest]:
    out = {}
    for f in sorted(Path(lab_dir).glob("*.yaml")):
        t = load_test(f)
        out[t.id] = t
    return out


def check(t: LabTest) -> list[str]:
    """Static checks (no engine): sides, decks, tiles on the right half, cards in the side's deck."""
    out = []
    if t.lane not in ("left", "right"):
        out.append(f"lane {t.lane!r}")
    for s in SIDES:
        if len(t.decks.get(s, [])) != 8:
            out.append(f"{s} deck has {len(t.decks.get(s, []))} cards")
    if t.decks.get("bluestacks") and sorted(t.decks["bluestacks"]) != sorted(BOMBER_DECK):
        out.append("the BlueStacks deck is not the live Bomber deck")
    for p in t.placements:
        if p.side not in SIDES:
            out.append(f"side {p.side!r}")
            continue
        if p.card not in t.decks.get(p.side, []):
            out.append(f"{p.card} not in the {p.side} deck")
        x, y = p.tile
        if not (0 <= x < ENGINE_W and 0 <= y < ENGINE_H):
            out.append(f"tile {p.tile} off the board")
        if not p.spell and p.side == "bluestacks" and y > OWN_MAX_Y:
            out.append(f"{p.card} at {p.tile}: past the BlueStacks side's half")
        if not p.spell and p.side == "cole" and y < COLE_MIN_Y:
            out.append(f"{p.card} at {p.tile}: past Cole's half")
    return out


# --- coordinates ---

def cole_tile(tile: tuple[int, int], view: str = "rotate") -> tuple[int, int]:
    """An engine tile (BlueStacks frame) as Cole sees it on his phone, his side at the bottom, y 0 = his king's row.
    'rotate': his screen is the BlueStacks screen turned 180 degrees (left and right swap); 'flip': turned upside
    down only (left stays left)."""
    x, y = tile
    return ((ENGINE_W - 1 - x) if view == "rotate" else x), ENGINE_H - 1 - y


def own_tile_words(tile: tuple[int, int]) -> str:
    """A tile on one's own side in words, from the player's own view (tile in that player's frame, y 0 = own king
    row): the lane by the princess tower columns (3 and 14), the row counted back from the river."""
    x, y = tile
    if y >= COLE_MIN_Y:  # on the other side (spells)
        side = f"enemy side, row {ENGINE_H - y} from their back edge"
    elif y > OWN_MAX_Y:
        side = "at the river"
    else:
        side = f"row {OWN_MAX_Y - y + 1} back from the river"
    if x <= 8:
        d = x - 3
        lane = "left tower's column" if d == 0 else f"{abs(d)} {'tile' if abs(d) == 1 else 'tiles'} {'right' if d > 0 else 'left'} of the left tower's column"
    else:
        d = x - 14
        lane = "right tower's column" if d == 0 else f"{abs(d)} {'tile' if abs(d) == 1 else 'tiles'} {'right' if d > 0 else 'left'} of the right tower's column"
    return f"{lane}, {side}"


def clock_text(t: float) -> str:
    """Game time (s since the start) -> the clock as shown (3:00 counting down; overtime not handled)."""
    left = max(0, round(180 - t))
    return f"{left // 60}:{left % 60:02d}"


def bluestacks_ground_point(tile: tuple[int, int]) -> tuple[float, float]:
    """Engine tile (its centre) -> our tile frame (bot77.features.frames), the ground point a tap lands on; what
    bot77.live.device.deploy takes."""
    from bot77.sim.geometry import engine_to_ours
    return engine_to_ours(float(tile[0]), float(tile[1]))
