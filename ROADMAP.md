# bot-77 roadmap

The plan we follow from here (written 2026-10-22). Update it when a step is done or the plan changes; the history of
*why* lives in `decisions.md`, how to run things in `README.md`.

**Goal:** a Clash Royale bot that wins: it climbs trophies on its own account (bot-77) in BlueStacks, built in the
order C (analysis) → A (move predictor) → B (the bot), for Cole's video.

**The metric that matters:** the bot's **win rate and trophies** in real games. Offline numbers (how often a model
copies a player's card or spot) are only proxies: use them to choose between versions, never as the goal.

---

## Where we are (2026-10-22)

| Piece | What it does | In the bot? | Evidence | Status |
|---|---|---|---|---|
| Screen reading (hand, elixir, clock, unit detector) | screen → data | yes | 0 wrong cards in hand reads; 55 games without getting stuck | ✅ working |
| Playbook rules (`config/playbook_rules.yaml`, `bot77.live.rules`) | when to play, which answer, where | **yes, decides most plays** | would change 70/73 reviewed mistakes (checked on the plays it came from) | ⚠️ not yet tested in real games |
| Card model A v1 (`data/models/moves_v1`) | ranks the 4 hand cards | yes, breaks ties the rules leave | 44% exact vs 25% random (copying players) | 🟡 small role |
| Card model + board, A v2 (`moves_v2`) | same, sees the units | no | test 44.0% → 44.0% | ❌ no gain, parked |
| Placement model (`bot77.place`, `place_v1`) | where to put a card | shadow: logs its spot only | test: 4.1 tiles from the real spot, right lane 77% (best of all methods) | ✅ offline, not tested live |
| Playbook table (mined answers) | answer spots | old policy only | beaten by the placement model | 🟡 replaced |
| Review loop (review pages + `review-bot-plays` skill) | finds mistakes → rules | how the rules are made | 73 mistakes in 144 plays | ✅ best source of improvement |

**Baseline to beat:** 17 W / 30 L (36%) in the 4-hour session before the rules (2026-10-22 morning).

---

## The loop (repeat)

1. **Play** a session (Cole runs it: `README.md` → "Running the bot", or the command below).
2. **Measure:** win rate and trophies for the session vs the previous one (`bot_games`, `data/live/game_results.json`).
3. **Review:** Claude pre-reviews the losses with the `review-bot-plays` skill → review page → Cole double-checks.
4. **Fix:** new mistakes → `config/playbook_rules.yaml` (+ a test in `tests/test_live_rules.py`); principles Cole adds
   → the skill. Check with `scripts/eval_rules_on_reviews.py`.
5. Commit + push, update this file.

```
caffeinate -i -s uv run python scripts/live_session.py --hours 2
```

---

## Now: get organised (bot sessions paused, Cole 2026-10-22)

We have hundreds of GB of data and it's glued together. Before more sessions or models: rebuild the training data
infrastructure per **[docs/training-architecture.md](docs/training-architecture.md)**: LanceDB tables for
matches / observations / moments / labels, one feature library, a models registry, storage tiers (laptop vs LaCie),
and a data browser (TypeScript + React + Tailwind front end, FastAPI back end). Phases 0–3 must reproduce today's
numbers before anything new is trained.

- [x] Phase 0 — inventory + cleanup (2026-10-22): [docs/data-inventory.md](docs/data-inventory.md) (`scripts/inventory.py`),
      `media` table (`scripts/build_media.py`: 775 files, 388 GB, laptop vs LaCie; every `videos` row resolves).
      401 experiment tables → macOS Trash (Cole: bin them; recoverable until the Trash is emptied). Old table
      versions dropped: LanceDB 80 GB → 18 GB (`unit_crops` had 58 GB of version manifests), row counts unchanged.
      Then (Cole): `data/upload` RunPod bundles (26 GB) and old review pages (0.9 GB) → Trash; the two broken links
      on the LaCie (`cole_2026-10-11_a/b.mov`, recordings gone) removed. Old model versions kept as the models'
      history: they become non-adopted entries in the models registry (phase 5).
- [x] Phase 1 — sources + observations (2026-10-22): `matches` gets source / split / pov side / pair / truth set /
      league / result (`build_sources.py`; split agrees with A's samples on all 96,568 rows; the 57 bot games are
      matches); `detections` + `runs` (`build_detections.py`: 6 runs, 32.4M rows, counts identical); `plays`
      (`build_plays.py`: 70,099 hand plays = events, 1,381 bot plays, 14,080 with a position). The old
      `unit_detections_*` tables stay until their readers move over (phases 2-3). C's detected plays: only the old
      default output (359 rows) survived the cleanup; C's current-default outputs on the evaluation sets were restored
      from the Trash into `plays` (runs c_test 30 games, c_heldback 48, c_dev 26, c_live 45; 4,100 plays). **The Trash
      can be emptied now.**
- [x] Phase 2 — moments + labels + reviews (2026-10-22): `moments` (`build_moments.py`: 26,467 replay plays,
      116,575 could-have-played waits, 1,381 bot plays; board from `detections`), `labels` (replay / bot / reviews),
      `reviews` (`import_reviews.py`: Cole's 60 + Claude's approved 84). **Check passed:** A's rows rebuilt from
      moments + plays (`check_moments.py`) are identical to `move_samples` (96,568 rows, no field differs) and A v1
      retrains to 46.6% held-back. Found on the way: the bot's own replays (53 games) were about to become imitation
      data -- they're now `replay_bot` and kept out of training by default.
- [x] Phase 3 — feature library + registry (2026-10-22): `bot77.features` (frames, hand, rates, board, spots,
      datasets, registry; the old `bot77.moves / board / place` are thin wrappers). 88 features registered
      ([docs/features.md](docs/features.md), `tests/test_features.py` keeps it in step). **Checks:** A's rows identical,
      A v1 46.6% held-back, A v2 and placement reproduce exactly with the roles file they were trained on (the newer
      roles moved placement 4.1 → 3.9 tiles: retrained as `place_v2`, the bot's shadow placer now uses it), rules on
      reviews unchanged (70/73), dry run clean.
- [x] Phase 4 — data browser (2026-10-22): `uv run python -m bot77.browser` → http://127.0.0.1:8077 (README → Data
      browser). FastAPI over LanceDB (`bot77.browser`) + React / TypeScript / Tailwind (`ui/`, types generated from the
      API). Pages: overview, matches, match timeline (video or frames, detector boxes, both players' plays, waits),
      moment inspector (board as the model sees it, card scores + features, placement spots, labels), training data
      (feature distributions per split, rows as the model gets them), features. Scalar indexes
      (`build_indexes.py`) took a match page from ~5 s to 0.1 s. Fixed on the way: bot play times were session time
      (now video time: game start + a measured 1.5 s recording lag); the bot's .mkv recordings are remuxed to .mp4 on
      first view so they can be seeked. Reviews page (phase 4b) still to move in.
- [x] Phase 5 — models registry (2026-10-22): `models` table (`bot77.models`): every model version with its role,
      status (adopted / superseded / not adopted / reference), what it was trained on, features, params and scores.
      Seeded with all 39 versions on disk and their history from decisions.md (`scripts/seed_models.py`); one adopted
      per role (card moves_v1, place place_v2, detector_live v6l, spell_pov mix25, spell_replay mix256, fake_filter
      v4, crop_clf). The bot loads the adopted model of each role (today's files as fallback); `train_moves.py` /
      `train_place.py` register what they save in data/models; `scripts/models.py list / show / adopt`; Models page
      in the browser.
- [x] Phase 6 — new models (2026-10-22): **A v3** = card + timing (the 4 cards and "wait" ranked at every play and
      could-have-played moment; `train_timing.py`). Test (once): card right at plays 44.0% (A v1) → **50.2%**;
      play-vs-wait AUC 0.78 (elixir alone 0.62); at the rules' false-play rate it catches **60%** of real plays vs the
      rules' 40%. **A v4** = A v3 + Cole's reviews as labels, read on the bot's paired replays (`pair_bot_replays.py`:
      49 of 53 replays paired, 94 usable review moments), weight 50 chosen by holding out reviewed games: agreement
      with Cole 33% → 53%, replay card accuracy -1.4 pts (test 48.8%). In the bot: **shadow** (`bot_shadow`: once a
      second what A v4 would do next to the rules); the rules still play.
- [x] **Session 2** (2026-10-22/23, rules play, A v4 + placement in shadow; new deck: Giant, Mini P.E.K.K.A,
      Musketeer, Goblin Hut, Fireball, Arrows, Skeletons, Minions): 61 games, **28 W / 32 L (47%)** vs 36% before the
      rules; **trophies 400 → 606**. 61 replays recorded (two pull bugs fixed on the way). 4,966 shadow rows.
- [ ] Next: process session 2 (replays → keys, pair with the bot games, moments), Claude pre-reviews the losses,
      Cole double-checks, then rules vs A v4 on those reviews -> decide whether A v4 takes over timing / cards.

## Next steps (after the rebuild)

- [ ] **Session 2 (rules on), 2 h** — first real test of the playbook rules. Every play now also logs the whole
      board and the placement model's spot.
- [ ] **Compare win rate** with the 36% baseline.
- [ ] **Review session 2's losses** (Claude first, Cole double-checks) → new rules.
- [ ] **Placement: rules vs model on real plays.** From session 2's logs and reviews: where the rules tapped vs where
      the model would have. If the model wins, let it place where the rules have no principle, with the rules as guard
      rails (no Giant at the bridge, ranged troops away from the attacker).
- [ ] **Card roles:** Cole corrects the guesses for the newer cards in `config/card_roles.yaml`.
- [ ] Repeat the loop until the win rate stops improving, then look at what's still losing games.

## Simulator (planned 2026-10-23, handoff: `docs/simulator.md`)
Cole's reviews don't scale and Claude's pre-review agreed with him on only 59% of plays. Plan: fork the MIT
ClashRoyaleEnv C++ engine (itzik123), build it on the Mac (S0), adapter to our tables (S1), check it against the
replays (S2), search for the best play at each moment and test it on Cole's reviews (S3: go / no-go), then scenarios,
a browser view, training. LanceDB is the data layer (`tower_hp`, `sim_validation`, `sim_scenarios`, `sim_runs`,
`labels` source `sim_search`).
- [x] S0 — fork builds on the Mac (2026-10-23): `uv sync` builds it; Catch2 735 + 1 expected fail / 736, Python
      1,007 / 1,012 (2 PPO-only failures); 2.3 ms per match, `snapshot()` 2.4 µs. One portability fix in the fork.
- [x] S1 — adapter `bot77.sim` (2026-10-23): our y is foreshortened (0.73 of our units per engine row, measured);
      all cards but Minion Giant map; `scenario_from_moment`; 7 tests. Found: the engine's 3x elixir starts at 3:00,
      our replays' at 4:00.
- [x] S2 — check against the replays (2026-10-24, docs/simulator-validation.md): tracks 1.0-1.25 of our units at
      +2.5-10 s, lanes 90-98%, fight winner 58-62% (expected ~80%), tower damage often missed. Fixed in the engine:
      triple elixir at 4:00, first-hit load times. `tower_hp` (OCR), `matches.result` (end screen, 20/20 by hand).
      S2b: bot aim offset 0.55 fixed.
- [x] S3 — search + gate (2026-10-24): **failed**. Verdict agreement 37.4% (gate 65%, Claude 59%), fix in top 2
      38.3% (gate 50%). Win model weak (held-back AUC 0.58). Stopped; disagreements page for Cole. No `sim_search`
      labels written.

## Later (only if the loop stalls)

- Retrain the card model on the bot's own reviewed games (needs a few hundred reviewed plays first).
- Learned "when to play" from replays + the bot's games, to replace the elixir thresholds.
- More decks / a deck the bot plays better.

## Parked (decided not to do now)

- **Similar-moment search** (LanceDB nearest moments) — held back by Cole, 2026-10-22.
- **Tuning the card model's accuracy** — A v2 with the board didn't move test accuracy; copying players' cards isn't
  the bottleneck.
- More feature sets for A.

## Known caveats

- The rules were tuned on the same 144 plays they're checked on; the real test is the next session's games.
- The placement model only learned from the 45% of replay plays whose spot was matched to a detection.
- The top player's board is mirrored about y 27.5 and taps are 1.27 tiles below a unit's box centre: both measured
  from data (2026-10-22), not exact.
- `config/card_roles.yaml` was written from general knowledge; newer cards are guesses.
- The bot plays a throwaway account only (automated play is against Supercell's ToS).
