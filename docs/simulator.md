# Simulator plan (handoff)

Written 2026-10-23 for an agent to pick up cold. Read this whole file first, then `ROADMAP.md`, then the "Ground rules"
section again before the first commit.

## Why

The bot (B) plays with hand-written rules from Cole's reviews plus a move predictor (A) trained on pro replays. Getting
better needs labels that say *what should have been played*, and right now only Cole can make them: about 150 plays
per review round, by hand. Claude's pre-review is not a substitute. On session 2's round 1 Claude agreed with
Cole on 26 of 44 plays (59%); of Cole's 30 mistakes Claude had called 17 OK; "high confidence" was right 5 of 13
times. What it misses needs to see the future: "the Giant is already dead, let it go", "Minions outrun the Giant and
die", "kite the Valkyrie to the middle with Skeletons".

A simulator sees the future. From any moment of a real game, try every card × spot (and waiting), play the next ~10 s
forward, and score what happened. That gives:
1. **Search labels**: the best play at a moment, for free, at every moment we have (161k).
2. **Review triage**: Cole only reviews plays where the search and the bot disagree.
3. **Scenarios**: set-ups we choose ("Giant on our 148 HP tower, left lane"), to test and train on.
4. **Later**: Cole playing in it from the data browser (S5), and RL (S6).

The catch is **sim-to-real**: a bot trained in a simulator that is slightly wrong gets good at the wrong game. So the
simulator is checked against our real replays before anything it says is trusted. That check is the core of this
plan, not an afterthought.

## The base: ClashRoyaleEnv (itzik123), forked

Two open simulators were evaluated on 2026-10-23 (code read only, not run):

| | [itzik123/ClashRoyaleAi](https://github.com/itzik123/ClashRoyaleAi) "ClashRoyaleEnv" | [Jason-XII/clash-royale-simulator](https://github.com/Jason-XII/clash-royale-simulator) |
|---|---|---|
| License | **MIT** | **none** (all rights reserved: can't copy code) |
| Engine | C++17 header-only, pybind11 module `clash_royale_env`, 10 ticks/s | Python, 20 fps |
| Speed (their numbers) | ~5 µs/tick, ~10 ms/match, `snapshot()` fork 7-20 µs | ~0.26 s/match |
| Cards | 132 + 41 evo/hero; all 8 of ours + common opponents | ours except **Goblin Hut** (no spawner logic) |
| Tests | 734 Catch2 + ~100 Python, CI | none |
| State control | `inject(card, x, y)`, `set_tower_hp`, `set_hand_for_team`, `set_elixir_for_team`, `set_current_tick`, `snapshot` | none built in |
| Already has | 10 s lookahead "teacher", Gymnasium env, PPO | RL env |
| Problems | builds on Windows/MSVC for Python 3.11 only; no general first-hit delay; retargeting re-scans every tick; one fixed collision radius, no mass; "engine tile is not a real tile" | no license; no tests; collision push ratio looks inverted |

**Decision: fork ClashRoyaleEnv, use its engine only.** Search needs speed (8 cards × ~600 spots × 10 s per moment):
that rules out a Python engine. Its state-control API is exactly what building a scenario from one of our moments
needs. Jason-XII stays a design reference only (its A* lane costs and target rules are worth reading; copy no code).

Not used from the fork: `perception/` (screen capture + ADB taps into the real game; we have our own bot, and nothing of
theirs goes near the game) and their PPO / teacher (we build our own search on our data; read theirs for ideas). Keep
their LICENSE, credit them in `README.md` and `decisions.md`, cite like KataCR.

## Ground rules (from the project, not optional)

- **The repo is public.** Commit and push after every working step; before each push scan the diff for secrets, local
  paths (`/Users/...`) and binaries. Game frames, videos, `data/` tables and review pages stay local and gitignored.
  The engine fork goes in as a git submodule at `third_party/clash-royale-env` (or a vendored copy with its LICENSE);
  build output (`build/`, `.so`) is gitignored.
- **Nothing touches the game client.** No game memory, network, client files or replay-file decoding. Card stats come
  from the fork (sourced from Supercell's balance notes and RoyaleAPI) or other public stat sites, never from game
  files.
- Never open or touch `client_secret_*.json`.
- No game frames to third parties.
- Keep `decisions.md` (one dated entry per phase: what was done, numbers, what was decided) and `ROADMAP.md` current.
- Report numbers honestly, including the ones that don't pass. A gate that fails is a result, not something to tune
  away.
- Ask Cole for anything only he can judge (Clash mechanics, whether a scenario plays right). He is a 1800 Ultimate
  Champion player: trust his mechanics corrections over any source.
- Python 3.12 + `uv` (`uv run python ...`), LanceDB at `data/lancedb`. Read `docs/training-architecture.md`
  (tables, conventions) and `docs/features.md` before writing tables.

## What we have to check against

All in LanceDB (`data/lancedb`). The coordinates are **arena tiles in the bottom player's frame**: x 0-18 left to
right, y down from the top of the arena, river at y 14.9 (`src/bot77/features/frames.py`). Unit positions are **box
centres**: a unit's feet are `FEET_DY` = 1.27 tiles below its box centre. A tap lands at the feet.

| Table | Rows | What | Use |
|---|---|---|---|
| `matches` | 2,273 | one per match and side; `source` (replay_tv / replay_bl / replay_bot / bot / ...), `split`, `pov_side`, `pair_match_id` | pick replays; respect the splits |
| `plays` | 80k | every play: `card`, `t_match` (game clock s), `x`, `y` (tile), `side`, `source` (`hand` = read from a replay's hand, exact) | the inputs to replay in the sim |
| `detections` | 33M | every unit box every 0.25 s: `run_id`, `card`, `team`, `x`, `y`, `conf` | the truth to compare unit tracks with (use `run_id = 'v5l_e15'`) |
| `moments` | 162k | one player at one time: `units` (card, team, x, y), `towers` (alive flags only), `hand`, `next`, `elixir`, `multiplier`, `t_match` | start states for search |
| `labels` | 162k | per moment: what was played (`replay`, `bot`) and Cole's verdicts (`review_cole`: 64 bad / 43 ok so far, with the fix card or "wait" and a note) | the go / no-go test |
| `bot_pairs`, `bot_play_moments` | 104 / 2,197 | the bot's games linked to their replays (exact opponent plays) | scoring the bot's real decisions |
| `cards` | 127 | card ids / names / elixir | name mapping |

Replays (both hands visible, so every play of both sides is exact): 390 TV Royale pro games (`replay_tv`), 81 Battle
Log games (`replay_bl`), and 115 of the bot's own games (`replay_bot`: never use them as imitation targets; they are
fine for validation).

**Gaps** the plan has to fill: no tower HP (only alive flags) and no unit HP anywhere. Tower HP is printed on screen in
every replay frame, so it can be read off the recordings (`videos.path`; S2). Unit HP is not visible; the simulator
assumes full HP at injection, so only use start states from right after the injected units were played, or compare
outcomes in a way that's robust to it (who survives, tower damage).

Splits: `matches.split` is `train` / `dev` / `heldback` / `test` / `heldout`. Calibrate on train/dev; report on
held-back; **never tune on test or heldout**.

## Phases

Each phase ends with a commit, a `decisions.md` entry and a short report to Cole (what passed, what didn't, numbers).
Time estimates are guesses; if a phase runs 2× over, stop and report instead of pushing on.

### S0 · Build the fork on this Mac (spike, ≤ half a day)

- Fork, add as a submodule, and build the pybind11 module with Apple clang for **Python 3.12 arm64** (the M5 laptop).
  Their CMake fetches pybind11 and Catch2 with FetchContent; read `CMakeLists.txt` and `CMakePresets.json` before
  building and drop MSVC-only flags. Build it from our side with `scikit-build-core` or a small script, so
  `uv run python -c "import clash_royale_env"` works.
- Run their Catch2 suite and their Python tests. Record pass / fail counts.
- Time one full match and one `snapshot()` fork. Write the numbers down.
- **Gate:** module imports under 3.12, ≥ 95% of their tests pass, a match runs in < 50 ms. If the build can't be made
  to work in the time box, stop and report: the fallback is a Python engine using the fork's stat rows and rules
  (MIT), which is much slower and changes S3's budget.

### S1 · Adapter `bot77.sim` (1 day)

`src/bot77/sim/`: everything between our data and the engine, so nothing else in the repo imports the engine
directly.
- **Coordinates:** a function pair `ours_to_engine(x, y, team)` / `engine_to_ours(...)`. Their arena is 18 × 34 with
  the river at y 15.5-17.5 and bridges at x 2.5 / 14.5; ours is 18 wide with the river at 14.9. Fit the map from
  fixed points (both towers' positions, the river, the bridges, deploy zones), check it on unit tracks (a Giant walking
  a lane should walk the same lane), and remember feet vs box centre. Test it.
- **Cards:** map our card names / `cards.id` to their card ids; list what's missing (`docs/simulator-cards.md`).
- **Teams:** the bottom player is our "own"; work out which engine team is which and keep it in one place.
- **State from a moment:** `scenario_from_moment(moment_id) -> Engine` sets the clock (`set_current_tick`), both sides'
  elixir (opponent elixir is unknown: estimate it from the opponent's plays since the last full bar, or assume a
  default and record it), our hand and next card, tower HP when known (S2), and injects every unit on the board at its
  position. Note what can't be carried over (unit HP, units mid-attack, spell effects in flight).
- **Events in and out:** `play(card, x, y, team, t)`; read back per-tick unit lists (card, team, x, y, hp), tower HP,
  elixir, kills.
- Tests: round-trip coordinates, a Giant injected at the bridge reaches the princess tower in about the real time,
  an Arrows drop on a Skeletons group kills them.

### S2 · Check it against the replays (2-3 days, the important one)

`scripts/sim_validate.py` → LanceDB `sim_validation` (one row per replay window and metric) and a summary in
`docs/simulator-validation.md`.

- **Windows:** take fights from human replays: start just before a play on an empty or near-empty board (so unit HP
  isn't a problem), feed the sim the *actual* plays of both sides from `plays` (source `hand`, exact) at their game
  clock times, run ~10-15 s.
- **Compare to `detections`** at each 0.25 s:
  - **Unit tracks:** match sim units to detected units by card + team (Hungarian on distance); median position error
    in tiles over time, per card.
  - **Life times:** when did each unit disappear in the replay vs the sim (did the Mini P.E.K.K.A win that fight?).
  - **Towers:** tower falls (and, once read, tower HP).
- **Tower HP from the recordings:** read the HP numbers over each tower in replay frames (OCR on the video pixels:
  allowed, it's the screen) → LanceDB `tower_hp` (match_id, t, tower, hp). Check it against the alive flags (a tower
  at 0 HP must be dead). Then compare tower damage over each window.
- **Fix what's off, in the fork, with a test per fix.** Known gaps to look at first: first-hit delay / load time,
  retarget behaviour, collision radius and mass, Goblin Hut spawn count and interval (their values aren't sourced),
  spell damage to crown towers. Ask Cole when the replays don't settle it. Offer fixes upstream.
- **No pass / fail here (Cole, 2026-10-23).** What matters is whether the sim ranks plays right, which is S3's
  question; a sim with 2-tile errors can still rank plays right, and one that meets every number can still rank them
  wrong. S2 *measures* (on held-back windows) and *flags*: units walking a different lane or target, fight winners
  consistently backwards, tower damage far off. Fix flags before S3. Report alongside each number the replay's own
  noise floor (detector jitter on a unit standing still, box centre vs feet ≈ 1 tile), so errors read against it.
  Rough expectations, not gates: unit tracks within ~1.5 tiles at +5 s, small-fight winner right in ~80%, tower damage
  within ~20%.

### S3 · Search for the best play (2 days) — the go / no-go

`src/bot77/sim/search.py`, `scripts/sim_search.py` → LanceDB `sim_runs` and labels.

- At a moment: candidates = each affordable card in hand × a placement grid (start coarse: a few dozen legal spots per
  card from `is_valid_placement`, refine around the best) + **wait**. For each, `snapshot()` the scenario, play it,
  step ~10 s.
- **The opponent during the 10 s:** for a replay (or a bot game paired with its replay) use the opponent's *actual*
  next plays from `plays`, at their real times and spots. Simple, real, and it doesn't need an opponent model.
  (Their plays were answers to what really happened, which is a bias; note it. A model opponent is S6.)
- **Score = change in win chance, not hand-set points (Cole, 2026-10-23):** "there is no right answer, it's what
  will win you the game." The same damage is worth different amounts in different games: down a tower, the king
  tower's extra HP is there to be spent ("we can allow as much damage as it will take to get even with our other tower
  that's alive"), and time left changes everything. So:
  - **Win model** (`bot77.sim.value`, a role in the `models` registry): P(bottom player wins | towers alive, tower HP
    (`tower_hp`), elixir both sides (estimated for the opponent), time left, multiplier, units on board summary),
    trained on the replays' states over time and each match's result (human replays, train split; calibration
    checked on held-back). **`matches.result` is empty for all 471 human replays (checked 2026-10-23):** fill it
    first, from the replay's end (towers left / crowns on the last frames, plus `tower_hp` at the end; overtime ends
    on the first tower) into `matches.result`, spot-checked by hand on ~20. Cheap to start (gradient boosting on a handful of features).
  - A candidate's score = win chance after the ~10 s window − before. Waiting can win: elixir kept is in the state.
  - Cole's rule is a scenario (S4) the score must get right: down a tower, soaking king-tower damage to save elixir
    for the even-up push should score above spending it to defend.
- **The test, on Cole's reviewed plays** (`labels` source `review_cole`, mapped to their replay moments through
  `bot_play_moments`):
  - verdict agreement: the search calls a play bad when the bot's play scores well below the search's best;
    compare with Cole's ok / bad;
  - on Cole's mistakes, is Cole's fix card (or "wait") in the search's top 2?
  - baseline to beat: Claude's pre-review, 59% verdict agreement (26 / 44).
- **Cole's own ceiling first:** re-show him ~20 round 1 plays blind (no earlier verdict) and measure his agreement
  with himself. That's the ceiling the search can reach; set the gate from it and write it in `decisions.md` before
  running the search on the gate set.
- **Gate (the only one in S0-S3):** verdict agreement ≥ 70% (adjust to the ceiling: e.g. ~10 points under it) and
  Cole's fix in the top 2 on ≥ 50% of his mistakes. If it fails, report where
  it disagrees (with the frames: Cole will say whether the sim or the review is wrong) before changing anything.
- If it passes: label every moment of the human replays and the bot's games: `labels` rows with `source = 'sim_search'`,
  `action`, `card_id`, `x`, `y`, `xy_kind = 'tap'`, the score margin, and the engine commit + score weights in a
  `sim_search` row of the `models` registry (`bot77.models.register`). The next review page shows Cole only the
  plays where the search and the bot disagree by a wide margin (`scripts/build_bot_review.py` already takes `--games`;
  add a filter).

### S4 · Scenarios (1 day)

- `config/scenarios.yaml`. **First 10 = Cole's round 1 mistakes** (`data/ground_truth/reviews/bot_review_2_cole.json`),
  built from their moments with Cole's fix as the expected answer:

  | Play | Bot did | Cole's fix |
  |---|---|---|
  | `bot_2026-10-22_234548:14` | Musketeer right while a Giant walks left on our 148 HP tower | Goblin Hut in the centre to pull the Giant |
  | `bot_2026-10-22_234548:4` | Skeletons onto a Valkyrie (spun instantly) | Skeletons / Hut in the centre to pull it off the tower |
  | `bot_2026-10-22_234548:6` | Minions "supporting" the Giant | they outrun the Giant and die: let the Giant tank first |
  | `bot_2026-10-22_234548:8` | Musketeer after our Giant died | nothing: let it go and reset |
  | `bot_2026-10-22_234548:18` | Giant in the back | Giant at the bridge in front of their big Spear Goblin group: counter-push |
  | `bot_2026-10-23_001115:5` | Goblin Hut to save the king tower | let it go: king has HP, save 5 elixir for the counter |
  | `bot_2026-10-23_001115:9` | Mini P.E.K.K.A into Goblins | Minions on the Goblins, Mini P.E.K.K.A on the Giant after |
  | `bot_2026-10-23_001115:19` | Mini P.E.K.K.A straight on their Mini P.E.K.K.A | Skeletons to pull it to the middle, then Mini P.E.K.K.A |
  | `bot_2026-10-23_015212:3` | Skeletons | Arrows ("we lost the tower because we refused to play arrows") |
  | `bot_2026-10-23_015212:6` | Mini P.E.K.K.A after losing a tower | Musketeer opposite lane to snipe the Valkyrie and counter-push |

  Plus Cole's rule: down a tower, soak king damage to save elixir for the even-up push.
- LanceDB `sim_scenarios` (scenario_id, from = yaml / moment, the state as JSON) and a runner that scores the bot's
  current policy (`bot77.live.policy`, offline) and the search on each. A regression suite: every rule change or new
  model gets run on it.

### S5 · In the data browser (later, only after S3 passes)

- Browser page (`bot77.browser` + `ui/`): watch a simulated window next to the real replay (same 0.25 s board view), see
  the search's candidates ranked with their scores.
- Then playable: Cole sets a scenario and plays it in the browser against scripted or model opponents; his plays
  become `labels` (`source = 'sim_cole'`).

### S6 · Training on it (later)

- A v5 trained with `sim_search` labels next to the replay labels (weights chosen like A v4's review weight: held-out
  agreement with Cole).
- RL / self-play only once the search labels show the simulator is right enough. The fork's Gymnasium env is the
  starting point.
- **The only number that counts is the real ladder**: win rate in bot sessions, as in `ROADMAP.md`. Never report a
  simulator win rate as progress.

## Decks change: nothing assumes ours

Cole changes the deck between sessions until we get Hog 2.8 ("it's always good to learn new cards"). The rules in
`config/playbook_rules.yaml` are written per deck; the simulator and search must not be. No card list anywhere in
`bot77.sim` except what comes from the moment (hand, deck) or the scenario. A new deck should get search labels from its
first session's replays without anyone reviewing first, which is the biggest win over the review loop. Scenarios name
situations; when a scenario's cards aren't in the current deck it is skipped, not rewritten.

## LanceDB: yes, as the data layer only

The engine runs in memory and never reads LanceDB. Everything that goes in or comes out is a table, so it shows up in
the data browser and joins with what we have:

| Table | Key | Written by | Holds |
|---|---|---|---|
| `tower_hp` | match_id, t, tower | S2 OCR | tower HP read from the replay video |
| `sim_validation` | run, match_id, window, metric | S2 | sim vs replay errors, engine commit |
| `sim_scenarios` | scenario_id | S4 | start states (from yaml or a moment) |
| `sim_runs` | run_id, moment_id / scenario_id, candidate | S3 | each candidate's outcome, win chance before / after |
| `labels` (existing) | label_id | S3 | `source = 'sim_search'` best plays |
| `models` (existing) | role `sim_search` | S3 | engine commit, score weights, gate results |

Same conventions as the other tables (`docs/training-architecture.md`): our tile frame everywhere, `match_id` /
`moment_id` keys, `optimize(cleanup_older_than=...)` after big appends, scalar indexes on the keys
(`scripts/build_indexes.py`). Table writes go through one module (`bot77.sim.store`).

## Done means

S0-S3 done = the simulator builds here, its error against real replays is measured and written down, and the search
either beats Claude's review agreement on Cole's plays (then it starts labelling) or a clear report says why not.
Report the numbers that missed their gate too.

## Open questions for Cole

Answered 2026-10-23: S2 measures and doesn't gate (the gate is S3, set from Cole's own blind re-review agreement);
score = change in win chance from a model trained on the replays, not hand weights; the first scenarios are the 10
round 1 mistakes above; nothing deck-specific. Still open:
1. Cole's blind re-review of ~20 plays (to set the S3 gate).
