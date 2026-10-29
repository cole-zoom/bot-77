# Simulator vs replays (S2, 2026-10-24)

How close the simulator (ClashRoyaleEnv fork, `bot77.sim`) comes to our recorded games. No pass / fail here (Cole,
2026-10-23): S2 measures and flags; the go / no-go is S3. Calibration choices were made on train windows; the numbers
below are **held-back** (48 games), never test / held-out. Engine: fork commit in each `sim_validation` row.

`uv run python scripts/sim_validate.py --run s2_heldback --towers royaleapi_l11 --splits heldback`

## Method

- **Windows** (`bot77.sim.validate`): a play on a board with at most 2 units on it (one frame 1 s and 0.25 s before),
  then every play of both players for up to 12 s. Card and time are exact (`plays`, source hand); the spot is the
  unit's first detection (box centre + `FEET_DY`), from `plays` or recovered from the detections (`bot77.sim.spots`:
  agrees with the stored spots to median 0.00 / p90 1.12 tiles; fills 62% of missing unit spots, 41% of spells). A
  window ends at the first play with no spot or no engine card (only Minion Giant is missing). Units already on the
  board are injected deployed at full HP; towers start at their HP read off the video (`tower_hp`); a king whose HP
  bar was seen earlier is woken.
- **Compared every 0.25 s** with the detections (v5l_e15): per card and team, sim units matched to detected ones by
  distance (Hungarian). Spells are left out (no body). Distances are in our units (x tiles; y foreshortened, 0.73 per
  engine row).
- **Noise floor**: a standing building's detections lie a median **0.05** of our units from their own median position
  (p90 0.66, when units overlap it; 38,430 detections). The position errors below are well above it. Box centre vs
  feet: `FEET_DY` 1.27 for ground units; flying units sit ~2.0 above their shadow, spells ~0 (from both players'
  forward-most placements, which mirror about the river).

**Method changes, 2026-10-29** (decisions.md, "Why units outlive their real selves"): a replay unit's death is the
end of its last run of detections with gaps up to 1 s (was: consecutive frames); Tesla, Royal Ghost and Suspicious
Bush are left out of life times (they vanish alive); flying spells (Fireball, Rocket, Snowball, Goblin Barrel) get
no spot from detections (the first one is in flight). Held-back life mean +1.82 -> +1.49 s from these alone.

## Numbers (held-back)

| | empty start (105 windows, 98 with every play accepted) | 1-2 units at start (241, 196) |
|---|---|---|
| position error, +0-1 s | 0.30 (122 pairs) | 0.32 |
| +1-2.5 s | 0.73 | 0.56 |
| +2.5-5 s | 1.00 | 0.79 |
| +5-7.5 s | **1.14** | 0.99 |
| +7.5-10 s | 1.25 | 1.13 |
| +10-12.5 s | 1.17 | 1.41 |
| same lane (+2.5-10 s) | 90% | 98% |
| unit count off per card / frame | 0.7-1.4 | 0.8-0.9 |
| fight winner right (both played) | **57.6%** of 85 | **62.2%** of 156 |
| winner right (one side played) | 75.0% of 12 | 70.0% of 40 |
| life time, sim - replay (units played in the window) | median **+1.75 s** (n 69) | +2.25 s (n 156) |
| tower damage: "was it damaged" agrees | own 86%, opp 79% | own 72%, opp 67% |
| tower damage where the replay shows > 0.1 princess tower | sim / real median 0.02 (own, n 6), 0.07 (opp, n 14) | 0.30 (n 44), 0.55 (n 46) |
| plays the engine refused | 7 of 340 | 59 of 822 |

Rough expectations from the plan (not gates): tracks within ~1.5 tiles at +5 s — **met** (1.0-1.14); small-fight
winner right in ~80% — **missed** (58-62%); tower damage within ~20% — **missed** (the sim often misses big tower
damage, and over-damages in others).

## Calibration on train (557 windows of 150 games), and what was chosen

| knob | options | winner agreement (empty / near-empty) | chosen |
|---|---|---|---|
| tower level | engine (lvl 9) / Liquipedia lvl 11 / RoyaleAPI lvl 11 | 60.3 / 61.2 / 62.0% (all windows) | RoyaleAPI lvl 11 (same source and level as the cards) |
| sight | strict centre-to-centre / edge-to-edge | identical | strict (unchanged) |
| first-hit load time | off / on | 66.4 → 67.9% / 54.5 → 56.4%; life +1.75 → +2.12 s | **on** (sourced, and timing is the goal) |
| wake a king seen damaged | off / on | 1 of 75 empty-start windows changed | on |

## Engine changes (fork `cole-zoom/clash-royale-env`, each with a test)

- Card ids sorted (MSVC / libc++ order differed; the Python suite failed on the Mac).
- `get_units()` (read-only), `set_tower_stats()` (re-level a tower), `set_sight_edge_to_edge()` (switch, off).
- **Triple elixir from 4:00**, one minute into overtime (was 3:00): 586 replays show 2x from 120.0 s and 3x from
  240.7 s; 51 of 159,041 readings off. GAMEPLAY-AFFECTING.
- **First-hit load time per card** (98 cards, RoyaleAPI cr-api-data 2023-12): units hit 0.1-0.3 s after locking on;
  the game waits the load time (Knight 0.7 s, Mini P.E.K.K.A 1.1 s, Valkyrie 1.4 s). GAMEPLAY-AFFECTING.
- Measured, not changed: elixir regenerates 2.90 s / elixir in our replays (engine 2.857).

## Flags (fix before trusting the matching parts of the search)

1. **Units live longer in the sim** (+1.75-2.25 s). Not towers (level changes moved it 0.25 s), not sight, not kings.
   Partly the replay side: units vanish from the detector early (team flips, misclassifications, one-frame flickers
   are smoothed but not removed). 45 of 94 replay deaths happened while the sim's copy was untouched: kamikaze cards
   (spirits, Wall Breakers) and units killed by something the sim didn't have.
2. **Tower damage**: big real damage often isn't reproduced. Seen: an X-Bow at the bridge that reaches the tower in
   the replay lands ~1 engine row too far back in the sim and can't; buildings and units already on the board missed by
   the start frame. A 1-row map error near the river decides siege reach.
3. **Squad formations**: at deploy (0-1 s), Rascals 1.73, Goblins 1.61, Skeletons 0.97 of our units off vs ~0.5 for
   single troops; Royal Recruits 4.9 later. Spawn offsets in the engine don't match.
4. **Spots of fast units** come from a detection after they moved (Royal Hogs, Battle Ram, Balloon, Bandit: refused as
   illegal). Fixing needs the tap point, not the first detection.
5. **Map near the river**: both players' forward-most placements map 0.1-0.8 rows past the engine's deploy limit
   (partly units detected after they started walking).

## S2b: the bot's execution

- **Aim**: the bot's units land a median **0.55 of our units below** where it aimed (dx -0.08; 1,791 plays; |miss|
  median 0.95; Goblin Hut, which never moves, +0.75). **Fixed**: `bot77.live.device.AIM_DY = -0.55`.
- **Delay** (decision to unit on the field): not measurable from clock reads: the replay's clock runs 3-5 s ahead of
  the bot's reads (pairing), and the read difference is 2.8 s. Open: needs the bot's own recording (when the card
  leaves its hand).

## Tower HP and results

- `tower_hp` (`scripts/build_tower_hp.py`): HP numbers OCR'd off the tower bars (Apple Vision on this Mac) every 1 s,
  accepted when not above the last accepted value and the next read isn't above it. Example (tv_2026-10-12_r03): 4858
  max (level 16), 72 steps on one tower under steady fire.
- `matches.result` (`scripts/fill_results.py`): from the end screen ("Winner!" above the winner's name; OCR), with tower
  detections + tower HP as the fallback. Hand-checked 20 held-back games against their end screens: **20 / 20** right.
