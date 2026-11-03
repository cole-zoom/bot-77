# Unit detector: what the decision maker needs (2026-11-03)

For whoever works on the unit detector / classifier. This is the contract between the detector and the live bot:
what goes in, what has to come out, what each field is used for, and what matters most. Everything downstream
(the screen -> engine mirror, the model) stays as it is; a detector that keeps this interface drops straight in.

Related: `docs/opponent-plays-spec.md` (the opponent-play log built on these detections) and
`docs/play-detector-spec.md` (the HUD readers: hand, elixir, clock). The detector is based on KataCR [Wu et al. 2025,
arXiv:2504.04783]; cite it for anything borrowed.

## 1. Where it sits

```
BlueStacks frame (1080 x 1920, portrait, BGR)
  ├─ HUD readers (not the detector's job): hand + Next (template match), elixir, clock
  └─ unit detector  ->  list[Detection]                     <- this document
        -> LiveReader: box centre -> arena tiles, class -> card name           (src/bot77/live/state.py)
        -> OpponentTracker: new enemy cards -> the opponent's plays              (same file)
        -> mirror(): units placed in a simulator engine                          (src/bot77/sim/mirror.py)
        -> the engine's observation -> the model -> a card + a spot -> a tap     (src/bot77/live/selfplay.py)
```

`scripts/dump_live_inputs.py <game>` writes every step of this for one recorded live game (what the detector found,
what the tracker counted, what went into the engine, what the model read and chose). Its output is the best example
of the data; ask Cole for a dump (game recordings and dumps stay private, under the gitignored `data/`).

## 2. The interface

```python
class Detector:
    def detect(self, frame: np.ndarray, layout: Layout) -> list[Detection]: ...

@dataclass
class Detection:                      # src/bot77/detect/units.py
    name: str                         # class name, e.g. "hog-rider", "mini-pekka-evo", "knight-hero", "queen-tower"
    team: str                         # "friendly" | "enemy"
    conf: float                       # 0-1
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 in FULL-FRAME pixels
```

- `layout` (`Layout.load("bluestacks_live")`) gives the arena crop: `layout.katacr_arena` scaled by
  `layout.scale(frame_width)`. The current detector crops that box, resizes it to 576 x 896 and runs YOLO at 896.
  How the new one sees the frame is up to you; boxes come back in full-frame pixels.
- **Class names**: the current detector (`data/models/bot77_detector/v6l`, adopted in the model registry as
  `detector_live`) has 219 KataCR-style classes. Names map to card names in `src/bot77/detect/names.py`
  (`class_to_card`): `-evo` / `-hero` suffixes are forms, a few classes are sub-units or summons (`SUBUNITS`, e.g.
  golemite -> Golem), and `NON_UNITS` (towers, bars, the clock, emotes, text, effects) are not units. A new class
  needs a line there, and a card the simulator doesn't have is dropped (`not in engine`).
- **Team**: the detector has no team output; team comes from the colour of the level badge above each unit (red
  enemy, blue friendly; `badge_team` in `src/bot77/detect/yolo.py`), else from which side of the river the box is
  on. A detector that predicts team itself is welcome; the field stays the same.
- Speed: in battle the reader calls `detect` every 0.2 s (`LiveReader.DETECT_EVERY`) and the model decides every
  0.1 s on the latest detections. The whole loop runs ~6 decisions a second on the M5 now; a slower detector makes
  every decision act on an older picture.

## 3. What each field is used for

| Field | Used for | Effect of an error |
|---|---|---|
| `name` -> card | which unit to put in the engine; a new enemy card = an opponent play (the model's "cards they have" input) | a wrong card is the wrong unit in the model's picture, and a play they never made in their cycle |
| `team` | own vs enemy unit; only enemy units count as opponent plays | **the worst error**: our unit read as theirs adds a fake play and an enemy unit where there's none |
| `box` centre | position: `x = (cx - arena_x0) / tile`, `y = (cy - arena_y0) / tile`, tile = arena width / 18. The unit's feet are 1.27 tiles below the box centre (`FEET_DY`); the engine places it there | a unit in the wrong place, or on the wrong side of the river |
| `conf` | everything >= 0.35 goes into the engine; the opponent tracker only counts >= 0.5 | flicker below the thresholds |

Not used today: unit HP (every unit goes in at full HP), and tower HP (towers count as full: not read live, see §5).
Spells are skipped (`SPELL_CLASSES` in `src/bot77/sim/scenario.py`). Bodies of one card within a few tiles are one
deploy (a Barbarians squad is one card).

## 4. What matters most, in order

1. **Team.** On 2026-11-03, game `bot_2026-11-03_113837`, by 1:15 the opponent tracker had counted 17 plays from 12
   different cards (a deck has 8), including three of our own cards (Battle Ram, Mini P.E.K.K.A, Mega Minion).
2. **The card at deploy.** The first 0.5 s of a new enemy unit decides which play gets logged. Getting the card right
   when it lands matters more than tracking it well afterwards.
3. **One play, counted once.** The tracker counts a card as a new play when it hasn't been seen for 6 s
   (`OpponentTracker(gap_s=6.0)`): a unit hidden behind a tower or missed for a few frames comes back as a second
   play. Stable detections, or track ids from the detector, would fix this.
4. **Units on our half**, where the bot has to defend: position and card, every frame.
5. Everything else: their side of the board, sub-units, effects.

## 5. Gaps the decision side has (if you want more scope)

- **Tower HP.** The model has inputs for all six towers' HP, and the pro training data had the real values, but live
  they always read full. A tower HP reader (the HP numbers over the towers, or the bar fill) would close that.
- **The opponent's elixir.** Not on screen. The model has no input for it today; the plan is to estimate it from
  their plays (5 at the start, regeneration by phase, minus each play's cost), so it is only as good as items 1-3.

## 6. Testing against the truth

- Battle Log replays of the bot's games can be watched from either player's side, so the opponent's hand is readable
  and gives their true plays (`data/ground_truth/opponent_truth_replay_<replay>.json`, read from the top player's
  side). `scripts/pair_bot_replays.py` pairs each live game with its replay. Compare the tracker's plays (or your
  detector's) against those.
- `bot77 benchmark` scores detections against the hand-checked sets (see `docs/opponent-plays-spec.md`).
- Before a detector goes live: the same game through `scripts/dump_live_inputs.py` with the old and the new one,
  and a 20-game live run.
