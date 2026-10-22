# Training architecture (plan, 2026-10-22)

**Status: proposal for Cole to review — nothing here is built yet.** Once agreed, it becomes the build plan in
`ROADMAP.md`.

Why: the data that trains A and the placement model is spread over ad-hoc tables, JSON files and config lists, each
built by its own script with its own conventions. It works, but nobody can see what the training data looks like,
and every new model starts with another one-off join. This plan puts all of it in a small number of LanceDB tables
with one schema, one feature library, one split, and a local UI to look at any of it.

---

## 1. What we have today

| Layer | Where it lives today | Size | Problem |
|---|---|---|---|
| Sources | `videos` (1,196), `matches` (2,031) | ok | no split / source / league / result columns: those are in 5 config JSONs and runner logs |
| Raw observations | `hud_states` (4.2M), `events` (70k), `unit_detections_*` (**33 tables**), `opponent_events_*` (**373 tables**), `unit_crops` (1.9M) | 79 GB | one table per experiment; which one is "current" lives in people's heads and scripts |
| Ground truth | `data/ground_truth/*.json` (~600 files) | 7.5 MB | answer keys and reviews are JSON files, not queryable with the rest |
| Training samples | `move_samples`, `board_states`, `placements`, `interactions`, `playbook` | 24k–97k rows | each built by its own script; board stored as JSON strings; two different mirror conventions for the top player |
| Bot games | `bot_actions`, `bot_games`, frames + mkv in `data/raw/live_games/` | 57 games | different schema from replay data, so the bot's games can't feed training without another converter |
| Features | computed inside `train_moves.py`, `train_place.py`, `bot77.board`, `bot77.moves` | — | partly shared with live play, partly duplicated; no list of what exists |
| Models | `data/models/*` (30 dirs) | 5.9 GB | no record of which data / features / metrics produced which model, or which is in use |
| Inspection | review HTML pages, contact sheets, ad-hoc scripts | — | no way to browse the data itself |

**Storage today:** the laptop holds LanceDB (79 GB), models (5.9 GB), hot videos and the bot's games (5.6 GB). The
LaCie (`/Volumes/LaCie/bot-77-data`, 323 GB used of 1.8 TB) holds the cold videos (everything except the test /
check / benchmark / dev games) and Cole's original .mov recordings; `data/raw/...` keeps 2,536 symlinks to them, so
paths in `videos` work only while the drive is plugged in. Nothing records which file is where.

What's solid and stays: `videos`, `matches`, `hud_states`, `events` (the replay answer keys), the detector output,
the hand / elixir / clock readers, and the split discipline (train / held-back / test / held-out, test scored once).

---

## 2. Principles

1. **LanceDB is the single source of truth** for everything a model trains or is scored on. Files on disk only for
   media (videos, frames) and model weights.
2. **One table per concept, versioned, not one table per experiment.** A detector run is a `run_id` column in
   `detections`, not a new table. Lance keeps table versions; important ones get a tag (`git`-style).
3. **Moments, not plays.** The unit of training data is a *moment*: one player, one point in time, the full state
   they saw. Plays, waits, bot decisions and reviews all hang off moments.
4. **Labels are separate from state,** with a `source` (replay key, bot, Cole's review, Claude's review) and a
   `weight`. A moment can have several labels (what the pro did, what the bot did, what Cole says it should have done).
5. **One feature library** (`bot77.features`), used by training *and* by the live bot, with a registry: every
   feature has a name, group, description and version. Training sets are moments ⨝ labels → features.
6. **One frame convention:** every position stored in the deciding player's frame, measured once and written down.
7. **Splits are a column on `matches`,** assigned once, never re-derived in scripts.
8. **Every model is registered:** dataset version, feature version, params, held-back / test metrics, adopted or not.
9. **Everything is inspectable in the UI** before it's trained on.

---

## 3. The tables

```
 videos ─┐
         ├─ matches ──┬── observations ─────────────┐
 (bot games)          │   hud, detections, plays     │
                      │                              ▼
                      │                         moments  ──── labels ◄── reviews
                      │                      (state at t)     (what was / should be done)
                      │                              │
                      │                              ▼
                      │                     features_<version>   (materialised, optional)
                      │                              │
                      └── split ─────────────────────▼
                                              training sets → models registry
```

### Sources
- **`matches`** (extend the existing table): add `source` (`replay_tv` / `replay_bl` / `pov_cole` / `bot`),
  `split` (`train` / `heldback` / `test` / `heldout`), `league`, `result` (per side), `deck_bottom`, `deck_top`,
  `player_bottom`, `player_top`, `has_key` (an exact replay answer key exists). Bot games become rows here too.

- **`media`**: one row per file (video, recording, frame folder): `path`, `kind`, `bytes`, `tier` (`laptop` /
  `lacie` / later `cloud`), `sha` (first MB), `available` (checked at UI start). The UI and build scripts look files
  up here, so a cold video says "on the LaCie, plug it in" instead of failing.

### Storage tiers
| Tier | What | Why |
|---|---|---|
| Laptop SSD | LanceDB (current tables), models, hot videos (split test / held-back / dev), the last few bot sessions | fast random reads; LanceDB never on the spinning drive |
| LaCie (HDD) | cold videos and recordings, older bot sessions, **the archived experiment tables** (read-only LanceDB copy, phase 0) | big and cheap; sequential reads are fine for video |
| Cloud (later) | only on the triggers in memory: repeated pod uploads, server access for B, off-site backup, outgrowing both | — |

Archived tables on the HDD are read rarely (old experiments), so the "never LanceDB on the HDD" rule is about the
*working* database only.

### Observations (per time, per match)
- **`hud`** = today's `hud_states` (both sides for replays).
- **`detections`**: one table, columns `run_id` (detector + version), `match_id`, `t`, `cls`, `card`, `team`, `conf`,
  box, `x`, `y` (tiles, bottom frame). Today's 33 `unit_detections_*` tables become `run_id`s; one run is marked
  current in a small `runs` table.
- **`plays`**: every play with its source: `key` (from the replay answer key: exact card and time),
  `detected` (C's opponent pipeline, one `run_id` per pipeline version — replaces the 373 `opponent_events_*`),
  `bot` (what the bot played). Card, time, `x`, `y` (where it landed, when matched), side.

### Moments (the core table)
- **`moments`**: one row per (match, side, t). Taken at **every play** and at **sampled non-play times** (e.g. every
  1 s when the player could have played but didn't — these teach "wait"). Columns, all native Lance types (no JSON
  strings):
  - ids: `moment_id`, `match_id`, `side`, `t`, `t_match`, `kind` (`play` / `idle`), `source`
  - player view: `hand` (4 ids), `next`, `elixir`, `multiplier`, `own_recent` / `opp_recent` (last plays: card, t,
    x, y)
  - board: `units` (list of struct {card, cls, team own/opp, x, y, conf}), `towers` (struct of 4 bools), in the
    player's frame
  - pointers: `detections_run`, `frame_ref` (video path + timestamp, for the UI)
- Built by one script from the observations; bot games produce moments directly (the bot already logs the same
  state at each play).

### Labels and reviews
- **`labels`**: `moment_id`, `source` (`replay` / `bot` / `review_cole` / `review_claude`), `action` (card or
  `wait`), `x`, `y` (spot, if a card), `verdict` (for reviews: ok / bad), `note`, `weight`.
  - For a replay moment: what the player did (or `wait` at an idle moment).
  - For a bot moment: what the bot did, plus a review label if reviewed (Cole's fix = the correct action, the bot's
    choice marked bad).
- **`reviews`**: the review sessions themselves (who, when, which page, raw export) so nothing lives only in
  `data/ground_truth/reviews/*.json`.

### Features
- **`bot77.features`**: functions `moment → features` for the moment, and `(moment, candidate) → features` for each
  candidate action (4 cards + wait) and candidate spot. Groups: `hand`, `elixir/clock`, `history`, `board`,
  `card_vs_board`, `spot`. A registry (`FEATURES`, with name, group, description, added in version) is generated into
  the docs and shown in the UI.
- **`features_<version>`** (optional): the features materialised per moment / candidate when computing them on the fly
  is too slow. Rebuilt when the feature version changes.

### Models
- **`models`** table: `name` (`card`, `place`, later `critic`), `version`, `dataset_version` (Lance tag),
  `feature_version`, `params`, `metrics` (held-back / test, by split and subset), `path`, `adopted` (bool), `notes`.
  The bot loads "the adopted version of each model" instead of hardcoded paths.

---

## 4. What the models become on top of this

| Model | Trained on | Output |
|---|---|---|
| **Card + timing** ("A v3") | play and idle moments × (4 cards + wait); replay labels + review labels (weighted) | play now? which card? |
| **Placement** | play moments × candidate spots; replay spots + Cole's clicked spots | where |
| **Critic** (later) | reviewed bot moments | "would Cole call this a mistake?" |
| Rules | — | guard rails; each rule retired once the models follow it ~95% on unseen reviews |

Review labels go into training with a high weight (they're few but exactly on the bot's deck and mistakes); replay
labels teach general play.

---

## 5. The UI ("bot-77 data browser")

A local web app (Python server reading LanceDB directly; never published — frames stay local). Pages:

1. **Overview:** row counts per table and split; label sources; cards in the data; which detector run / model
   versions are current.
2. **Matches:** filter by source, split, league, deck, result; click into one.
3. **Match timeline:** the video with a scrubber; at any time the frame with the detected units drawn (team colours,
   card names), the hand, elixir, clock; the plays of both players as marks on the timeline; moments and labels
   listed under it.
4. **Moment inspector:** the frame, the state as the model sees it (board drawn in the player's frame, so you can
   check the mirroring), every feature value, each candidate's score from each model, and the label(s).
5. **Training set view:** a sample of rows exactly as a model receives them, with feature distributions per split
   (catches leaks and broken features).
6. **Models:** registered models and their metrics side by side; examples where two versions disagree.
7. **Reviews:** today's review pages moved in here: queue of unreviewed bot moments, Claude's pre-reviews with
   Agree / change, clicks for the right spot — written straight to `labels`.

Tech choice to discuss: plain FastAPI + a small HTML/JS front end (fast, no build step, matches the current review
pages) vs Streamlit (quicker to write, clunkier for video scrubbing and clicking on frames).

---

## 6. Build plan (each phase ends with a check before the next)

| Phase | Work | Done when |
|---|---|---|
| 0. Inventory + archive | list every table / file / model and the `media` table (laptop vs LaCie); move experimental tables (`opponent_events_*`, old `unit_detections_*`, `mined_smoke*`, …) to an archive LanceDB on the LaCie (copied, checked, then removed from the laptop) | the main DB has only current tables; every file has a tier; nothing lost |
| 1. Sources + observations | `matches` columns (split, source, league, result), `detections` with `run_id`, `plays` with source; bot games as matches | split counts match today's config files; plays match today's keys |
| 2. Moments + labels + reviews | `moments` (play + idle), `labels`, `reviews`; bot games and the 144 reviews imported | **A v1 retrained from the new tables reproduces 46.6% held-back**; board positions agree with `board_states` |
| 3. Feature library + registry | `bot77.features` replacing the code in `bot77.moves` / `bot77.board` / `bot77.place`; live bot switched to it | A v1 and the placement model reproduce their numbers through it; the live dry run is unchanged |
| 4. UI | pages 1–5 first, then models and reviews | Cole can open any match / moment and see what the model sees |
| 5. Models registry | `models` table, bot loads adopted versions | the current models registered with their metrics |
| 6. New models | card + timing with "wait", review labels weighted, then the critic | measured on held-back, then test once, then a bot session |

Phases 0–3 are plumbing: they shouldn't change any number. That's the check that the migration is correct.

---

## 7. Open questions for Cole

1. **Archive vs delete** the ~400 experimental tables (79 GB DB)? Plan says archive them to the LaCie (copied and
   checked before they leave the laptop).
2. **UI tech:** FastAPI + HTML (recommended) or Streamlit?
3. **Idle-moment sampling:** every 1 s between plays, or only when the player had enough elixir to play something?
4. **Review weight:** how much one of your reviews should count vs a pro replay play (start at 20×, tune on held-back
   reviews?).
5. **Bot sessions during the rebuild:** keep playing with the current code (data still logs in the old format and gets
   imported in phase 2), or pause until phase 3?
