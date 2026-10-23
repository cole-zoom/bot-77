# Data inventory

Generated 2026-10-22 by `scripts/inventory.py` (phase 0 of [training-architecture.md](training-architecture.md)). Regenerate after any data change.

## Storage

| Tier | Used | Free |
|---|---|---|
| Laptop SSD | 701.6G | 224.7G |
| LaCie HDD (`/Volumes/LaCie/bot-77-data`: 323.3G) | 468.9G | 1.4T |

## LanceDB tables (`data/lancedb`, laptop)

| Table | Layer | What it is | Rows | Size | Versions | Written by | Read by |
|---|---|---|---|---|---|---|---|
| `fakes_now` | analysis (C) | current fake plays snapshot (fake-filter work) | 3,014 | 256K | 2 | — | — |
| `live_outcomes` | analysis (C) | play outcomes on Cole's live (POV) games | 667 | 76K | 1 | — | — |
| `play_outcomes` | analysis (C) | per detected play: real / fake / missed vs the key; trains the fake filter | 16,909 | 1.0M | 2 | — | build_failures.py, mine_failures.py |
| `pov_fakes` | analysis (C) | fake plays in Cole's POV games | 1,228 | 140K | 2 | — | — |
| `bot_actions` | bot | every bot play: card, spot, reason, hand, elixir, threats, full board, model spot | 1,381 | 216K | 1 | loop.py | build_bot_review.py, build_moments.py, build_plays.py, eval_rules_on_reviews.py, import_reviews.py, pair_bot_replays.py, |
| `bot_games` | bot | every bot game: start / end, plays, result, deck | 57 | 32K | 1 | loop.py | build_bot_review.py, build_moments.py, build_plays.py, build_sources.py, import_reviews.py |
| `bot_pairs` | bot | each bot game and its Battle Log replay (pair_bot_replays.py) | 49 | 32K | 2 | pair_bot_replays.py | — |
| `bot_play_moments` | bot | each bot play -> the replay moment it happened on (exact hands, full board) | 1,022 | 56K | 2 | pair_bot_replays.py | train_timing.py |
| `reviews` | ground truth | each exported bot-play review (who, when, approved by, raw export) | 2 | 76K | 3 | import_reviews.py | __init__.py, after_batch4.sh, after_batch5.sh, after_batch6.sh, after_clicks8.sh, after_cole_1001.sh, after_cole_1001bcd |
| `playbook` | model input | interactions aggregated per opponent card: common answers and spots | 4,718 | 176K | 1 | build_playbook.py | eval_playbook_place.py, policy.py, train_place.py |
| `models` | models | the models registry: every model version, role, status, data, features, scores | 41 | 1.1M | 98 | — | app.py, autolabel.py, build_failures.py, build_real_v3.py, crop_classifier.py, cropclf.py, ingest_live_replays.py, inges |
| `detections` | observation | every detector run in one table (run_id): boxes, class, card, team, arena tiles | 32,407,283 | 1.8G | 10 | build_detections.py | app.py, build_failures.py, build_indexes.py, build_moments.py, eval_v2.py, mine_v4_chainA.sh, mine_v4_chainB.sh, mine_v4 |
| `hud_states` | observation | what the HUD showed, every frame read: hand, next, elixir, clock, multiplier | 4,194,766 | 157.3M | 4 | — | build_indexes.py, build_moments.py, build_moves.py, opponent.py, pairing.py, pipeline.py, render_game.py, train_fake_fil |
| `opponent_events` | observation | C's detected opponent plays (default output table) | 359 | 10.2M | 1 | — | build_plays.py, eval_v2.py, opponent.py, review.py, slices.py, train_moves.py |
| `runs` | observation | what each detector run is and what it's used for | 6 | 16K | 1 | build_detections.py | app.py, build_train_v3.py, mine_v4_chainA.sh, pod_v3.sh, pod_v4.sh, pod_v5.sh, pod_v6.sh, pod_v8.sh |
| `unit_detections` | observation | detector output (older default run) used by mining, slices, autolabel | 14,071,127 | 278.9M | 1 | — | autolabel.py, build_detections.py, mine.py, opponent.py, run.py, slices.py |
| `unit_detections_mine4` | observation | detections from mining run 4 (RunPod), source of autolabels | 3,862,466 | 76.7M | 1 | — | build_detections.py |
| `unit_detections_v5l_e15` | observation | detector V5 on all 524 keyed replays, every 0.25 s: the board for A / placement | 9,529,520 | 189.4M | 1 | — | build_board_states.py, build_detections.py, build_placements.py, build_playbook.py, eval_playbook_place.py, ingest_live_ |
| `unit_detections_v6l_best` | observation | detector V6 (live bot's detector) on the bot's replays | 2,466,065 | 48.9M | 1 | — | build_detections.py, ingest_live_replays.py |
| `unit_detections_v7l_best` | observation | detector V7 on a check set (30-ish games) | 1,229,471 | 24.4M | 1 | — | build_detections.py |
| `unit_detections_v8l_best` | observation | detector V8 on a check set | 1,248,634 | 24.7M | 1 | — | build_detections.py |
| `events` | observation / ground truth | plays and abilities read from the hand (exact for the player whose hand is shown; replay answer keys) | 70,099 | 3.4G | 4 | pipeline.py | __init__.py, add_next_card_plays.py, after_session_f.sh, after_session_g.sh, analyze_dev.py, autolabel.py, autolabel_pai |
| `plays` | observation / ground truth | every play: from the hand (exact), detected by C, or made by the bot | 75,939 | 23.8M | 12 | build_plays.py | app.py, build_bot_review.py, build_click_short.py, build_indexes.py, build_moments.py, datasets.py, eval_rules_on_review |
| `cards` | reference | the card catalogue: names, elixir, type, rarity, art (from the Clash API) | 127 | 87.3M | 3 | cards.py | __init__.py, add_next_card_plays.py, analyze_dev.py, app.py, autolabel.py, autolabel_paired_spells.py, build_board_state |
| `matches` | source | one row per match in a video: start / end, opponent, deck. Split / source / league not here yet | 2,088 | 1.4M | 8 | build_sources.py | add_next_card_plays.py, analyze_dev.py, app.py, autolabel.py, autolabel_clock_spots.py, autolabel_paired_spells.py, auto |
| `media` | source | every video / recording file: where it really lives (laptop / LaCie), size, available now | 773 | 352K | 3 | build_media.py | app.py |
| `videos` | source | one row per video file: creator, layout, mode, path (laptop or LaCie symlink), size | 1,253 | 976K | 8 | build_sources.py | after_session_f.sh, after_session_g.sh, app.py, autolabel.py, build_board_states.py, build_click_abilities.py, build_cli |
| `labels` | training data | what was / should have been done at a moment: replay, bot, review_cole, review_claude | 144,567 | 31.6M | 16 | build_moments.py | after_batch3.sh, app.py, auto_sprites.py, autolabel.py, build_indexes.py, build_real_v3.py, build_train_mined.py, build_ |
| `moments` | training data | one player at one point in time with everything they saw: plays + could-have-played waits + bot plays | 144,423 | 13.6M | 5 | build_moments.py | app.py, build_indexes.py, datasets.py, import_reviews.py, pair_bot_replays.py, train_timing.py |
| `board_states` | training data (A) | the board at each of A's decisions (JSON strings for now) | 24,142 | 1.6M | 1 | build_board_states.py | build_placements.py, train_moves.py |
| `move_samples` | training data (A) | A's samples: one row per (decision, hand card), both players of every keyed replay | 96,568 | 6.1M | 2 | build_moves.py | build_board_states.py, check_moments.py, seed_models.py, train_moves.py |
| `move_samples_m` | training data (A) | A's rows rebuilt from moments + plays (identical to move_samples; phase 2 check) | 96,568 | 9.4M | 3 | check_moments.py | app.py |
| `mined_candidates` | training data (C) | mined failure candidates for review / labelling | 140 | 2.6M | 3 | — | build_mined_review.py, label_mined.py, mine_failures.py |
| `real_crop_index` | training data (C) | index of real crops used by the crop classifier | 5,051 | 7.4M | 1 | — | autolabel.py, crop_classifier.py |
| `slice_index` | training data (C) | index of video slices for labelling | 4,844 | 7.2M | 1 | — | — |
| `unit_crops` | training data (C) | unit image crops + embeddings: train the card classifier, feed mining | 1,892,719 | 13.7G | 1 | mine.py | dedupe_crops.sh, mine_all.py, mine_failures.py |
| `placements` | training data (placement) | every play with its spot and the board before it | 14,080 | 1.2M | 1 | build_placements.py | build_plays.py, seed_models.py, train_place.py |
| `interactions` | training data (playbook) | play -> the other player's answer within 8 s, with positions | 20,643 | 848K | 1 | build_playbook.py | — |
| `opponent_events_dev_dev_s6_m25` | ? | **undocumented** | 636 | 648K | 51 | — | build_plays.py |
| `opponent_events_dev_lt_s6_m25` | ? | **undocumented** | 830 | 1.2M | 89 | — | build_plays.py |
| `opponent_events_dev_rt6_v5l_e15` | ? | **undocumented** | 1,057 | 804K | 59 | — | build_plays.py |
| `opponent_events_dev_va6_m256` | ? | **undocumented** | 1,577 | 1.4M | 95 | — | build_plays.py |

Total: 42 tables, 19.9G.

## Data folders

| Folder | What it is | On laptop | Symlinks (→ LaCie) |
|---|---|---|---|
| `data/raw/replays` | replay recordings from the Battle Log (bot + Cole's account) | 17.0G | 16 (16) |
| `data/raw/tv_royale` | TV Royale replay recordings (top ladder) | 4.7G | 364 (364) |
| `data/raw/live_games` | the bot's games: screen recording + a frame at / after every play | 5.6G | 0 (0) |
| `data/raw/replay_links` | symlinks to replay videos by match id | 0B | 1048 (0) |
| `data/raw/live_ui` | screens the bot didn't recognise between games | 25.7M | 0 (0) |
| `data/raw/live_calib` | BlueStacks calibration frames | 2.1M | 0 (0) |
| `data/raw/youtube` | YouTube download metadata | 44K | 0 (0) |
| `data/raw/*` (files at the top) | Cole's POV recordings + YouTube videos (top level) | 12.0G | 154 (152) |
| `data/external` | datasets for the detector: KataCR, synthetic and real crop sets | 9.7G | 954 (0) |
| `data/upload` | bundles made for RunPod uploads (tarballs and unpacked copies) | 0B | 0 (0) |
| `data/reviews` | generated review pages (HTML) | 112K | 0 (0) |
| `data/ground_truth` | answer keys and Cole's reviews (JSON) | 7.5M | 0 (0) |
| `data/models` | model weights | 5.9G | 0 (0) |
| `data/calibration` | layout calibration images | 189.3M | 0 (0) |
| `data/cards` | card art + overrides | 30.4M | 0 (0) |
| `data/slices` | video slices for labelling | 56.7M | 0 (0) |
| `data/logs` | run logs | 17.7M | 0 (0) |
| `data/opponent` | opponent-pipeline debug images | 1.2M | 0 (0) |
| `data/live` | the bot's runtime files (STOP flag, game results) | 4K | 0 (0) |

## Models (`data/models`)

| Model | Size | Referenced in code |
|---|---|---|
| `bot77_detector` | 4.6G | build_real_v3.py, ingest_live_replays.py, ingest_replays.py, live_calibrate.py, live_dryrun.py |
| `crop_clf.pt` | 1.2M | crop_classifier.py, cropclf.py, seed_models.py |
| `crop_clf_nocand.pt` | 1.1M | crop_classifier.py, seed_models.py |
| `fake_filter` | 580K | opponent.py, playrules.py, seed_models.py, train_fake_filter.py |
| `fake_filter_v2` | 776K | seed_models.py |
| `fake_filter_v3` | 824K | seed_models.py |
| `fake_filter_v4` | 804K | playrules.py, seed_models.py |
| `katacr` | 334.1M | build_train_mined.py, build_train_v3.py, crop_classifier.py, generate_synth.py, mix_spell_probs.py |
| `moves_v1` | 1.0M | app.py, moves.py, policy.py, seed_models.py, train_moves.py |
| `moves_v2` | 1.3M | seed_models.py |
| `moves_v3` | 2.3M | train_timing.py |
| `moves_v4` | 1.8M | moves.py, policy.py |
| `place_v1` | 968K | seed_models.py |
| `place_v2` | 1.1M | app.py, place.py, policy.py, seed_models.py, train_place.py |
| `probs_new_v2` | 11.8M | seed_models.py |
| `probs_new_v3` | 11.9M | seed_models.py |
| `probs_new_v5` | 11.8M | seed_models.py |
| `sam` | 74.5M | autolabel.py, seed_models.py, slices.py |
| `spell_mix` | 18.1M | build_failures.py, mix_spell_probs.py, seed_models.py |
| `spell_mix235` | 14.7M | seed_models.py |
| `spell_mix25` | 18.1M | playrules.py, seed_models.py |
| `spell_mix256` | 18.1M | playrules.py, seed_models.py |
| `spell_mix25ab` | 14.7M | seed_models.py |
| `spell_mix25avg` | 14.7M | seed_models.py |
| `spell_mix26` | 18.1M | seed_models.py |
| `spell_v1` | 125.4M | seed_models.py |
| `spell_v2` | 134.3M | mix_spell_probs.py, seed_models.py |
| `spell_v3` | 134.3M | mix_spell_probs.py, seed_models.py |
| `spell_v4` | 129.9M | seed_models.py |
| `spell_v5` | 134.3M | seed_models.py |
| `spell_v6` | 134.2M | seed_models.py |
| `yolo11s.pt` | 18.4M | seed_models.py, train_detector.py, train_v2.py |

## Ground truth (`data/ground_truth`)

| Group | Files |
|---|---|
| `opponent_truth_replay_tv` | 390 |
| `opponent_truth_replay_bl` | 134 |
| `opponent_truth_live_cole` | 55 |
| `opponent_truth_auto_cole` | 16 |
| `opponent_truth_auto_friend` | 12 |
| `truth` | 4 |
| `opponent_truth` | 1 |
| `reviews/` | 64 |

## Config (`config/`)

| File | Referenced in code |
|---|---|
| `benchmark_holdout.json` | analyze_dev.py, autolabel_clock_spots.py, autolabel_paired_spells.py, autolabel_v4.py |
| `card_roles.yaml` | board.py, frames.py |
| `classes_v2.json` | auto_sprites.py, autolabel.py, build_real_v3.py, build_train_v2.py |
| `heldback_replays.json` | build_failures.py, build_moves.py, build_placements.py, build_playbook.py |
| `layouts` | __init__.py, layout.py, manual_layouts.py, prepare_batch.py |
| `live_test_games.json` | build_sources.py, holdout.py |
| `play_timing.json` | play_timing.py, playrules.py |
| `play_timing_replay.json` | playrules.py |
| `playbook_rules.yaml` | build_bot_review.py, frames.py, rules.py |
| `recording_decks.json` | autolabel.py, session_click_page.py |
| `replay_test_games.json` | autolabel_clock_spots.py, build_failures.py, build_moves.py, build_placements.py |
| `spell_clip_relabels.json` | spell_clips.py |
| `v5_check_games.json` | autolabel_clock_spots.py, build_train_mined.py, build_train_v5.py, label_mined.py |

