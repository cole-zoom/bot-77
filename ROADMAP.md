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
      default output (359 rows) survived the cleanup; the current pipeline is rerun into `plays` when needed.
- [ ] Phase 2 — moments + labels + reviews (A v1 reproduces 46.6% held-back from the new tables)
- [ ] Phase 3 — feature library + registry (live bot switched to it, dry run unchanged)
- [ ] Phase 4 — data browser
- [ ] Phase 5 — models registry
- [ ] Phase 6 — new models (card + timing with wait, reviews as weighted labels)
- [ ] Then resume the loop below with session 2.

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
