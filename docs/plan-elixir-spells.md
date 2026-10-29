# Plan: knowing when to spend, and spell use (2026-10-29)

Cole, after watching the defence-school model (data/models/bc_defence_v2/epoch03_cal20.pth) play real games:
- "it defends better, but it doesn't know it has an elixir advantage. we should teach it to count elixir, and to go
  in once it has a substantial lead"
- "sometimes it freezes", "it almost plays too passive, like it hesitates"
- "sometimes it fireballs or arrows nothing like it's cycling it in the back. we needa fix that too"
- "its actually really good at defending though, when its not freezing. so lets make sure we don't get rid of its
  defesnive capabilities"
- "we also need to teach it to take a chill pill when nothing is happening ... it needs to know when to chill, and
  when to place a lot. this shouldnt be determined by willingness to play elixir, its knowing when to spend, and
  needs to be learnt."

**The goal:** the model learns *when* to spend from the state it sees: the elixir count on both sides, the board,
and the clock.
- Chill when nothing is happening and spending would only hand the opponent a counter-push. Chilling means sitting at
  9-10 elixir, then a cheap or back play, not leaking for 30 s.
- Defend when they push, as it does now.
- Go in hard when it's ahead on elixir or a tower is low.

The spend rate must come from the network, not from a knob:
- **Calibration** (the play/wait bias on card_head.bias[4]: log 50, log 20) sets one willingness to play for every
  state. It can't tell calm from pressure, so it's not the fix. The aim is a model that plays right at one fixed
  calibration, chosen once on training seeds. If it only looks right at a different calibration, that counts as a
  miss.
- **The live full-bar guard** below is a temporary band-aid for Cole's games. It's never in training or the exam,
  and it comes out once the model stops freezing on its own.

## What we measured (decisions.md, 2026-10-28 night / 2026-10-29)

- **Live test, school at log-50 calibration:** 6-12 clean; the bum rush is gone (2 of 19 openings at the bridge, v4 13
  of 14). But it's passive: the first play came at 9-10 elixir in 13 of 19 games, and the log shows full elixir on
  154 lines. At log 20 it was 3-1 after four games (test still running when this was written).
- **The freeze is the model, not the bot.** In the log-20 test, game 2 (bot_2026-10-29_011219) went 34 s (clock
  2:18-1:47) with 233 decisions. It chose wait 226 times at p(wait) 0.95-0.99, at 10 elixir, with no threats. The
  unreadable-card slot was picked only 5 times. The defence school saw only defence, so it learned that nothing to
  defend means wait. It has never seen a calm state where the right answer was to spend, so it can't tell "chill"
  from "press".
- **It can't see its lead.** The observation (fork include/core/ClashEnv.h, scalars 1-2) has its own elixir and both
  sides' *cumulative spend / 280*, not the opponent's elixir. One elixir of lead moves those inputs by 0.0036.
- **Wasted spells.** In the log-50 test, about 26 of 70 Fireballs and Arrows landed behind our own towers or in the
  centre of our side, with nothing there as far as the logs show (v4: about 9 of 120). The situation score charges
  0.04 per elixir, so a wasted Fireball costs about 0.16, which is small next to the tower terms.

## Don't lose the defence

The defence is the thing that works, so every phase protects it:
- **The baseline stays.** data/models/bc_defence_v2/epoch03.pth (and its _cal50 / _cal20 copies) and
  data/cache/bc/defence_t01 are read-only. New work goes in new folders.
- **A defence gate** for every new checkpoint or scoring change: the exam on the 10 existing defence templates at
  `--n 50`, with the same calibration, compared per template with v2 epoch 3. It fails if the defence-template mean
  drops more than 0.05 below v2 epoch 3's, or if any single template drops more than 0.15. A failed gate is reported
  as it stands, not tuned away.
- **The new data adds; it doesn't replace.** The new demos join the defence demos. No defence template gets fewer
  demo windows than in v2, and the new tempo templates (chill + press) are capped at about a third of the windows.
- **The guards never touch a defence play except a wasted spell.** The full-bar guard only fires with no enemy unit
  on our side of the river.

## Phases

Each phase ends with a commit + push (scan the diff first: no secrets, no `/Users/...`, no binaries), a decisions.md
entry with its numbers, and a stop check: if a phase runs twice its estimate, stop and report.

### P1. Spell guard, and the live band-aid (est. 2 h)

1. **Spell guard (a real rule, live and in the sim).** Fireball or Arrows only go down if at least one enemy
   non-tower unit, or an enemy tower, is within the spell's radius of the target. Take the radii from
   bot77.sim.cards / the engine card stats, not from memory. Otherwise the action becomes wait.
   - Put it in one shared function (e.g. src/bot77/rl/guards.py), used by:
     - src/bot77/live/selfplay.py `decide()` (from `st.units`, team "enemy"), after the placement is sampled;
     - CloneEnv.step (src/bot77/rl/clone_env.py) on our action, so training, the exam and live play agree.
   - Log every veto: live in decisions.json (a `guard` field), sim as a counter in `info`.
2. **Full-bar guard (live only, temporary).** If elixir is at least 9.8 for 3 s, no enemy unit is on our side, and the
   model keeps choosing wait, play its highest-probability affordable non-wait, non-blind card at its sampled cell.
   The spell guard still applies. Behind a `--full-bar-guard` flag on scripts/live_play.py, off by default. Mark these
   plays `guard` in the log and the console line so they're counted apart from the model's own plays. Not in
   CloneEnv, not in the exam.
3. Tests: unit tests for the spell-guard function (spell on empty grass is vetoed; Arrows on a Skeleton Army isn't;
   Fireball on an enemy tower isn't). Then the exam with the spell guard on against epoch03_cal50 at
   `--n 50 --reference`, comparable with -0.90, plus the defence gate.

### P2. Scoring: elixir is worth something on both sides (est. 2 h)

The score has to make "chill" and "press" both correct in their states. Leaking must cost something, and so must
spending into an opponent who has the elixir to punish it. In src/bot77/rl/situations.py (`_finish` and the step
accounting) and CloneEnv (full games):
1. **Leak.** Track the elixir that regen would have added while we sat at 10 (seconds at full × the current regen
   rate) and subtract `LEAK_COEF * elixir_leaked`. Start at 0.04 per elixir, the same as spending. Apply it to every
   score kind and report `elixir_leaked` in the details dict.
2. **Spells by what they hit.** For every score kind (not only "trade"), a spell that touches no enemy unit or tower
   costs `WASTED_SPELL_COEF * cost` on top of its elixir. Start at 0.04, so a whiffed Fireball costs about 0.32. The
   guard makes this rare in the sim, but keep it for the guard-off exam and the record.
3. **Overcommit stays costly.** The existing `deficit` kind (dont_overcommit) already punishes spending into their
   elixir. Check that leak + deficit together still make sitting at 9-10 and then making one cheap or back play beat
   both "dump everything" and "never play" on the calm templates (P3).
4. **Full games (RL):** a small per-step leak penalty for team 0 at 10 elixir in CloneEnv.step, scaled so a whole game
   of leaking costs about one princess tower. Behind `RL_LEAK_COEF`, default 0, so v1-v5 stay reproducible.
5. Re-run the exam (`situation_exam.py --n 50 --reference`) for nothing / rush / defend / the school. The scripted
   defence must still beat both references on every template (Cole's check), and the defence gate applies. Rescore
   Cole's play-page games (`scripts/rescore_cole.py`) and report his mean before and after.
6. Don't tune these coefficients on the exam seeds (10_000+). If one needs changing, change it from training-seed
   numbers and say so.

### P3. Tempo situations: chill vs press, as matched pairs (est. 4 h)

The model has to learn the difference from the state, so the templates come in **pairs**: the same board, a
different elixir picture, opposite right answers. Only the count separates them, which is exactly what we want it
to learn (and what P5 makes visible). All are generators taking rng and returning a spec. Add them to TEMPLATES, the
play page (scripts/situation_play.py) and the exam.

**Calm pair** (nothing is happening: an empty board or a lone cheap unit of theirs at the back, mid-game, single
elixir):
- **chill:** they're at about our elixir or more (8-10). Right answer: hold to 9-10, then one cheap or back play
  (Goblin Hut, a tank behind the king), and keep elixir to defend what they drop. Score kind `tempo`: leak cost +
  their counter-push's damage + `deficit`-style elixir-at-end. "Dump everything at the bridge" must lose to their
  counter; "never play" must lose to the leak.
- **press_calm:** the same board, but they're at 0-3 (they just spent). Right answer: go in now (Battle Ram or a
  Giant + support at the bridge), before they rebuild. Score kind `press` (below).

**Pressure pair** (we just defended their push and have survivors):
- **punish:** our elixir 7-10, theirs 0-3, their units dead or nearly. Right answer: counter-push behind the
  survivors, in the weak lane.
- **hold_after_defence:** the same board, but they're at 7-10. Right answer: one support play behind the survivors,
  or nothing; keep elixir for their next push.

**Single:**
- **spell_finish:** an enemy princess tower at or below Fireball's tower damage (the fork's measured 158 at Crown
  Tower level) or with units clustered in front of it, and our elixir at 4+. Right answer: Fireball on the tower and
  the units.

**Score `press`:** over a fixed window (~15 s, the opponent script defending with its elixir):
`DEALT_COEF_PRESS * dealt / princess_max - lost / princess_max - ELIXIR_COST * spent - LEAK_COEF * leaked + board`.

**The reference check** (each pair is useless if the scoring can't tell its halves apart):
- **Scripted tempo policy:** a new policy using the lead (our elixir − their estimated elixir). Press at a lead of +3
  or more, chill otherwise, defend on a threat. Add it to `scripted_defence` so build_defence_demos.py produces demos
  for the new templates.
- **The ordering it must produce:**
  - chill beats dump and never-play on the chill halves;
  - press beats chill and never-play on the press halves;
  - the same scripted policy wins both halves.
- If an ordering doesn't come out right, report it and stop. Don't ship a template the scoring can't judge.
- The +3 threshold is a guess: ask Cole what lead he goes in at (and whether it depends on the clock or double
  elixir) before it goes into demos.

Cole plays the new templates on the play page; they become his answer keys like the defence ones. That's his step:
list the commands for him and don't play them.

### P4. School v3, and does it know when to spend (est. 3 h, mostly compute)

1. Rebuild the demos with `scripts/build_defence_demos.py`: all templates, now including the tempo ones, and Cole's
   new play-page games if he's played them by then. They go into a new folder, data/cache/bc/school_v3_t01; leave
   defence_t01 alone.
2. Train like v2 (`scripts/train_bc.py`, same flags as data/models/bc_defence_v2/train.log: demo share 0.7, lead 3)
   into data/models/bc_school_v3. Every epoch: the exam with the spell guard on, at one calibration (log 50 unless
   training-seed numbers say otherwise, chosen once), and the defence gate.
3. **The tempo report**, beyond the exam mean, for v3 vs v2 epoch 3:
   - per tempo pair, both halves' scores and whether it did the right thing in each (spent within ~3 s on press
     halves; held to 9+ and made at most one cheap play on chill halves);
   - the play rate by state on the held-out exam seeds: plays per 10 s when calm vs under a threat vs ahead by 3+
     (the model should vary this; one flat rate means it learned a knob, not a state);
   - elixir leaked and wasted spells per template;
   - Cole's 142 starts.
4. **Stop there.** No RL round and no live games: Cole runs the live games, and RL comes after he's seen this. The
   later leashed RL round gets the leak penalty on and the tempo templates in its situation mix.

### P5. Show the model the count (gated: write the proposal, don't edit)

The fork's observation gets the opponent's estimated elixir (a regen clock minus their seen spend, clipped 0-10,
divided by 10) next to scalar 0, or the lead (ours − theirs) / 10. The live bot already computes the estimate
(src/bot77/live/selfplay.py `_mirror`: `5.0 + regen - spent`).
- Without it, the P3 pairs can only be told apart through a 0.0036-per-elixir difference of two spend totals, so
  this is likely needed for P3 to work. Measure that in P4 first: if v3 can't separate the halves of a pair, that's
  the evidence.
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
- Ask Cole about Clash mechanics whenever the data doesn't settle something (the lead he presses at, when he chills,
  spell radii). Don't guess what a player would know.
