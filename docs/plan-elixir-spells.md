# Plan: elixir sense and spell use (2026-10-29)

Cole, after watching the defence-school model (data/models/bc_defence_v2/epoch03_cal20.pth) play real games: "it defends
better, but it doesn't know it has an elixir advantage. we should teach it to count elixir, and to go in once it has a
substantial lead", "sometimes it freezes", "it almost plays too passive, like it hesitates", and "sometimes it fireballs
or arrows nothing like it's cycling it in the back. we needa fix that too". He asked for one plan for both, handed to
another agent.

## What we measured (decisions.md, 2026-10-28 night / 2026-10-29)

- **Live test, school at log-50 calibration:** 6-12 clean; the bum rush is gone (2 of 19 openings at the bridge, v4 13
  of 14). But it's passive: the first play came at 9-10 elixir in 13 of 19 games, and the log shows full elixir on
  154 lines.
- **The freeze is the model, not the bot.** In the log-20 test, game 2 (bot_2026-10-29_011219) went 34 s (clock
  2:18-1:47) with 233 decisions. It chose wait 226 times at p(wait) 0.95-0.99, at 10 elixir, with no threats. The
  unreadable-card slot was picked only 5 times. The defence school saw only defence, so it learned that nothing to
  defend means wait.
- **It can't see its lead.** The observation (fork include/core/ClashEnv.h, scalars 1-2) has its own elixir and both
  sides' *cumulative spend / 280*, not the opponent's elixir. One elixir of lead moves those inputs by 0.0036.
- **Wasted spells.** In the log-50 test, about 26 of 70 Fireballs and Arrows landed behind our own towers or in the
  centre of our side, with nothing there as far as the logs show (v4: about 9 of 120). The situation score charges
  0.04 per elixir, so a wasted Fireball costs about 0.16, which is small next to the tower terms.

## Phases

Each phase ends with a commit + push (scan the diff first: no secrets, no `/Users/...`, no binaries), a decisions.md
entry with its numbers, and a stop check: if a phase runs twice its estimate, stop and report.

### P1. Guards: same rule live and in the sim (est. 2 h)

1. **Spell guard.** Fireball or Arrows only go down if at least one enemy non-tower unit, or an enemy tower, is within
   the spell's radius of the target. Take the radii from bot77.sim.cards / the engine card stats, not from memory.
   Otherwise the action becomes wait.
   - Live: src/bot77/live/selfplay.py `decide()`, from `st.units` (team "enemy"), after the placement is sampled.
   - Sim: the same check in CloneEnv.step (src/bot77/rl/clone_env.py) on our action, so training, the exam and
     live play agree. Put the rule in one shared function (e.g. bot77/rl/guards.py), used by both.
   - Log every veto: live in decisions.json (a `guard` field), sim as a counter in `info`.
2. **Full-elixir guard (live only, a band-aid).** If elixir is at least 9.8 for 2 s and the model keeps choosing wait,
   play its highest-probability affordable non-wait, non-blind card at its sampled cell (the spell guard still
   applies). Mark the play as `guard` in the log and in the console line ("full-bar guard"), so these are counted
   apart from the model's own plays.
3. Tests: unit tests for the guard function (spell on empty grass is vetoed; Arrows on a Skeleton Army isn't; Fireball
   on an enemy tower isn't). The exam with the guard on, against epoch03_cal50 at `--n 50 --reference`, so the number
   is comparable with -0.90.

### P2. Scoring: leaks and spell value (est. 2 h)

In src/bot77/rl/situations.py (`_finish` and the step accounting) and CloneEnv (full games):
1. **Leak.** Track the seconds at 10 elixir during a situation; subtract `LEAK_COEF * elixir_leaked` (elixir_leaked =
   seconds at full × the current regen rate). Start LEAK_COEF at the same 0.04 per elixir as spending: a leaked elixir
   should cost as much as a spent one. Apply it to every score kind, and report `elixir_leaked` in the details dict.
2. **Spells by what they hit.** For every score kind (not only "trade"), a spell that touches no enemy unit or tower
   costs `WASTED_SPELL_COEF * cost` on top of its elixir. Start at 0.04, so a whiffed Fireball costs about 0.32. The
   guard makes this rare in the sim, but keep it for the guard-off exam and for the record.
3. **Full games (RL):** in CloneEnv.step, a small per-step leak penalty for team 0 when at 10 elixir, scaled so a whole
   game of leaking costs about as much as one princess tower. Behind a flag (`RL_LEAK_COEF`, default 0) so v1-v5 stay
   reproducible.
4. Re-run the exam (`situation_exam.py --n 50 --reference`) for nothing / rush / defend / the school. The scripted
   defence must still beat both references on every template (Cole's check). Rescore Cole's play-page games
   (`scripts/rescore_cole.py`) and report his mean before and after.
5. Don't tune these coefficients on the exam seeds (10_000+). If one needs changing, change it from training-seed
   numbers and say so.

### P3. "Press the lead" situations (est. 3 h)

New templates in situations.py, same shape as the existing ones (a generator taking rng, returning a spec), a score
kind `press`, added to TEMPLATES, to the play page (scripts/situation_play.py) and to the exam:
1. **punish:** we've just beaten their push. Our elixir is 7-10, theirs 0-3 (opp_elixir in the spec), their units are
   dead or nearly dead, and a survivor of ours may be at the bridge. The right answer is to go in, in the lane where
   they're weak or behind a survivor.
2. **full_bar:** an empty board, we're at 9-10 elixir and they're at 4-7. The right answer is to play something
   (a tank at the back, a Battle Ram at the bridge, or a Goblin Hut), not wait.
3. **spell_finish:** an enemy princess tower at or below Fireball's tower damage (see the fork's measured 158 at
   Crown Tower level) or with units clustered in front of it, and our elixir at 4+. The right answer is Fireball on
   the tower and the units.
- **Score `press`:** `DEALT_COEF_PRESS * dealt / princess_max - lost / princess_max - ELIXIR_COST * spent - LEAK_COEF
  * leaked + board`, over a fixed window (about 15 s). The opponent script defends (the existing "defend" opp plan) so
  a lone overcommit is punished. Check the references:
  - "nothing" must score badly (leaks + no damage);
  - "rush" (everything at the bridge) must score worse than a sensible press when they have elixir to defend;
  - a new scripted `press` policy (the lead check: our elixir − their estimated elixir ≥ 3) must beat both.
  If no reference ordering comes out right, report it and stop: don't ship a template the scoring can't judge.
- Add the scripted press policy to `scripted_defence` so build_defence_demos.py produces demos for the new templates.
- Cole plays the new templates on the play page. That's his step: list the commands for him and don't play them.

### P4. School v3 and its exam (est. 3 h, mostly compute)

1. Rebuild the demos: `scripts/build_defence_demos.py` (all templates, now including press) into a new folder
   data/cache/bc/school_v3_t01 (leave defence_t01 alone).
2. Train like v2 (`scripts/train_bc.py`, same flags as data/models/bc_defence_v2/train.log: demo share 0.7, lead 3)
   into data/models/bc_school_v3. Exam every epoch, with the guard on, at log-50 and log-20 calibration.
3. Report per template against v2 epoch 3 (-0.90 overall, log 50), with elixir spent, elixir leaked and wasted spells,
   and Cole's 142 starts.
4. **Stop there.** No RL round and no live games: Cole runs the live games, and RL comes after he's seen this.

### P5. Show the model the count (gated: write the proposal, don't edit)

The fork's observation gets the opponent's estimated elixir (a regen clock minus their seen spend, clipped 0-10,
divided by 10) next to scalar 0, or the lead (ours − theirs) / 10. The live bot already computes the estimate
(src/bot77/live/selfplay.py `_mirror`: `5.0 + regen - spent`).
- Write the exact C++ diff (include/core/ClashEnv.h, the observation size constant, every consumer of the
  observation size in python_ai/), what it invalidates (every checkpoint's input layer: the imitation and RL nets
  would need retraining, or the new input zero-initialized), and how the live mirror would fill it.
- Put the proposal in docs/plan-elixir-spells.md under P5 and **stop for Cole's OK.** The fork's .claude/CLAUDE.md
  says no C++ change without confirming the exact diagnosis and edit with him. Gameplay physics are unchanged, so
  win-rate history stays valid.

## Ground rules (from docs/simulator.md; they apply here)

- The repo is public. Commit and push after every working step. Scan each diff for secrets, local paths and binaries.
  data/ stays gitignored.
- Nothing touches the game client. Don't run the live bot or touch BlueStacks; Cole runs live games.
- third_party/clash-royale-env: python_ai/, include/ and src/ are read-only (P5 is a proposal only).
- Python 3.12 + uv (`uv run --no-sync --with gymnasium==1.3.0 ...`, as the existing scripts' docstrings show).
- Report numbers honestly, including misses. Never tune on the exam seeds. Log every measured finding in
  decisions.md as it happens. Commit attribution: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ask Cole about Clash mechanics whenever the data doesn't settle something. Don't guess spell radii, tower
  damage or lead thresholds a player would know.
