# Plan: the simulator as a 1:1 copy of the battle (2026-10-29)

Cole, after 51 ladder games went 21-30 and 142 play-page games "felt off": *"before we train again, can we focus on
making the sim closer to a 1:1 representation of the actual game?"* His three goals:

1. Interactions as close to 1:1 as possible.
2. The play page shows every tile, separated: "rn I cannot tell what tile I am placing things on."
3. Every troop behaves the same, does the same damage, has the same stats.

No training (imitation or RL) until this plan's gate (section 7) passes. Graphics are out of scope. The battle logic
is in scope: everything that decides who wins a fight and how much a tower takes.

## 1. Where we start (measured 2026-10-29, held-back replays, fork 5e7711b)

| | empty board | 1-2 units on board | target (section 7) |
|---|---|---|---|
| fight winner right | 60% | 67% | 90% lab, 80% replays |
| unit life time, sim - real (median) | +1.75 s | +1.50 s | within ±0.3 s |
| big tower damage, sim / real (median) | 0.14-0.34 | 0.34-0.60 | 0.9-1.1 |
| position error at +5 s (our units) | 0.96 | 0.87 | ≤ 0.6 |
| same lane | 90% | 98% | 98% |

Units outlive their real selves on nearly every card (Knight +1.75 s, Hog +2.4, Valkyrie +4.1; Ice Spirit 0.0). A bias
that broad is a systemic cause, not per-card stats (those already match the client on 101 of 101 cards). An
investigation agent is tracing it now.

## 2. Sources of truth, best first

| Source | What it settles | Cost |
|---|---|---|
| **Client logic tables** (`data/external/cr_client/logic`, pulled from BlueStacks 10-28, no root) | every number: stats, LoadTime, DeployTime, Pushback, Mass, CollisionRadius, projectile Speed / Homing, AreaDamageRadius, CrownTowerDamagePercent, buffs, death spawns, towers | have it |
| **Lab battles** (friendly battles, Cole's phone vs the BlueStacks account, scripted placements, the replay recorded from the Battle Log) | how the numbers are *used*: exact, repeatable, both hands known, frame-accurate | Cole's evenings |
| **The client's code** (the native battle library, read with REA (github.com/morluto/rea) on top of Ghidra) | the rules themselves: tick length, attack cycle, retarget, pushing, pathing, when LoadTime applies | slow; aimed at one question at a time |
| **Replays already recorded** (586+ pro / ladder, 42 of our own) | the regression yardstick (S2 metrics) at scale | have it |
| **Cole** (1800 UC) | which behaviour is real when two sources disagree | ask concretely |

Rule: a mechanic is settled when two sources agree (client value + lab, or code + lab). One source is a hypothesis.

## 3. Phases

### P0 · Groundwork (today, while Cole is at work)

- **P0.1 Tile grid on the play page** (goal 2). Every tile outlined (18 x 34 engine cells; the river and bridges
  shaded), the tile under the cursor highlighted with its coordinates, the legal placement area tinted (own side,
  plus the pocket once a princess tower falls), and a ghost of the card's footprint (buildings 2x2 / 3x3). Clicks snap
  to the tile centre the engine uses. Also: unit HP numbers on hover, both elixir bars, the clock.
- **P0.2 Field coverage table.** Every behaviour field in the client tables against the engine: modelled? reads the
  client value? Ranked by how much it plausibly explains the misses. (Asked of the investigation agent.)
- **P0.3 The lifetime cause.** The investigation agent's report: each candidate (load time per retarget, tick
  quantisation, hit timing, projectile travel, deploy, how the replay side's death is read) measured, fixes with a test
  and a held-back re-run.
- **P0.4 Lab tooling**, ready for tonight:
  - a lab script format (`config/lab/*.yaml`): per test, both decks, and a timed list of `card, tile, side` placements;
  - a cheat sheet for Cole per test (which card, which tile, when) and a tile overlay he can match on his phone;
  - after the session: `replay_runner.py` pulls the friendly battles' replays from the BlueStacks Battle Log (both
    hands visible, so placements are exact), the existing pipeline reads units, tower HP and deaths;
  - `scripts/lab_compare.py`: the same placements in the sim, then per test the outcomes (who died, when, tower HP,
    positions) side by side, and a frame-pair viewer (real frame | sim board at the same tick).
- **P0.5 Native library feasibility.** Pull the arm64 battle library from BlueStacks (`adb pull` of the split APK, no
  root). Check: are the logic field names ("LoadTime", "HitSpeed") readable strings, and can their loader be found
  from them? That decides whether code reading is cheap (hours a mechanic) or not (then lab only). Needs REA
  installed (its setup edits the agent config, so Cole's OK) and Ghidra (free, Java present).

### P1 · Data parity: every number from the client (goal 3)

Driven by the P0.2 table. For every card in the pool (101 cards plus evolutions) and the towers:

- every stat and behaviour field read from the client tables at the level the battle uses (friendly / ladder level
  caps: confirm in the lab), replacing hand-entered values in the engine's card registry;
- a generated test per card: the engine's loaded values equal the client's, so a game update shows up as a failing
  test, not silent drift;
- the parts the tables hold beyond units: projectiles (speed, homing, radius), area effects (slow, freeze, rage,
  durations), death spawns and death damage, spawner buildings (interval, count), building life and decay, tower
  stats (princess / king, level 11, damage, hit speed, range), spell crown-tower percentages.

Order: our deck first (Bomber, Battle Ram, Giant, Mega Minion, Goblin Hut, Mini P.E.K.K.A, Fireball, Arrows), then the
cards we meet most on the ladder (from the 51 games' replays), then the rest.

### P2 · Mechanics parity: how the numbers are used (goal 1)

One mechanic at a time. Each gets: what the client value is; what the code does (if P0.5 says it's readable); a lab
test that isolates it; the engine fix; a C++ test pinned to the lab measurement; a held-back S2 re-run.

| # | Mechanic | Why it matters | Lab test |
|---|---|---|---|
| 1 | **Tick length and time** (engine 0.1 s; the client's unknown), elixir rate (engine 2.857 s, real 2.8), double / triple timings | every delay rounds up a tick; a few per hit cycle add up to the +1.5 s | elixir: time 0 to 10 from the replay HUD |
| 2 | **Attack cycle**: first hit after LoadTime, then HitSpeed; does LoadTime reset on retarget, after moving, after pushback | the lead suspect for units living too long | Knight vs Knight, Mini P.E.K.K.A vs Knight, Bomber vs Skeletons |
| 3 | **Targeting**: sight, lock-on, retarget when a closer target appears, building-only units, air / ground, tower pulls | who hits whom decides fights and pulls | Giant past a Goblin Hut; Hog vs a building in the centre; Mega Minion vs Bomber vs Skeletons |
| 4 | **Projectiles**: travel time, homing, what happens when the target dies mid-flight, splash centre vs edge | Bomber, Mega Minion, towers all shoot projectiles | Bomber into a clump; tower vs Skeletons |
| 5 | **Splash and area effects**: radius from the edge or the centre, slows / freezes through splash, spell waves (Arrows), Fireball knockback | spell value and the Bomber | Arrows on Skeletons / Goblins; Fireball on a Bomber + Knight |
| 6 | **Deploy**: deploy time, the delay before the first move, spawn positions of squads, deploy on a unit | the start of every fight | Skeletons and Goblins squads placed on a tile, positions at landing |
| 7 | **Collision, mass, pushing** | blocking, kiting, Battle Ram stopping | Giant pushing Skeletons; Battle Ram into a Knight |
| 8 | **Movement and pathing**: speeds, bridges, river, turning, ranged units walking around towers, ram charge | lanes and arrival times | Hog / Battle Ram from the back to the tower, clocked |
| 9 | **Charge units** (Battle Ram, Prince): charge distance, charge damage, reset on hit | our Battle Ram | Battle Ram into a Knight, into the tower |
| 10 | **Towers**: princess / king damage and hit speed, King activation (hit, or a princess falls), the targeting radius, crown tower damage reduction | tower damage numbers | Giant to the tower alone, tower HP over time |
| 11 | **Buildings**: decay, spawners (Goblin Hut interval), death spawns | our Goblin Hut | Goblin Hut alone for its life |
| 12 | **Placement rules**: legal tiles, pocket placement, spells anywhere, the tile grid's real alignment | goal 2 and live play | placements at the edges of the legal area |

### P3 · Validation that keeps it honest

- **Lab regression suite:** every lab test becomes a fixture: the sim must reproduce its outcome (who died, death
  times within 0.2 s, tower HP within 5%). Run on every engine change.
- **S2 on held-back replays** after every fix, with the per-card lifetime and winner tables, so a fix that helps
  one card and hurts another shows.
- **Side-by-side viewer** for Cole: a replay window next to the sim's version, frame by frame, so he can say
  "that's wrong" with his eyes.
- Test and held-out splits stay untouched until the final measurement.

### P4 · Back to training

Only after section 7's gate: re-run the exam baselines (scripted defence, the school) on the new engine; rebuild the
situation templates and demos; then Cole's play-page examples on the new page (goal 2), then training.

## 4. Tonight's lab session (Cole, ~1-2 h)

Needed first: the BlueStacks account as a friend of Cole's (or both in one clan) for friendly battles. Then ~20 short
scripted tests, about 3 min each, in priority order: the attack cycle (Knight vs Knight; Mini P.E.K.K.A vs Knight;
Bomber vs Skeletons), a tower test (Giant alone to the princess tower), a spell test (Arrows on Skeletons; Fireball on
Bomber + Knight), Battle Ram into a Knight and into the tower, Goblin Hut alone, Hog vs a centre building, squad
deploy positions, the elixir clock. The cheat sheet says exactly what to place, where (tile) and when. The BlueStacks
side places its half from a script over ADB, so Cole only plays his half.

## 5. What could go wrong

- **The client's code may be unreadable** (stripped and obfuscated). Then P2 runs on lab tests and the tables only.
  Slower, but it still converges.
- **The friendly battle level may differ from ladder.** Measure which level the client uses there and set the
  engine to match per test.
- **The engine's architecture may not fit a mechanic** (e.g. its tick length). Changing the tick changes every
  checkpoint's timing; if so, we decide it with Cole before doing it.
- **Our own reading of the replays** (detections, HP OCR) has its own error. Every number is reported next to the
  replay's noise floor, and lab tests are judged on HP bars and death frames, not detector boxes.
- **Game updates** change values. The generated per-card tests catch them.

## 6. Ground rules (unchanged)

The fork's python_ai/ is read-only. C++ changes go to the main session for review (Cole delegated the C++ review),
each with a test. Client files never go in the repo (data/ is gitignored). Every measured finding goes in decisions.md
as it happens. Cite sources: client field, lab test id, or code address.

## 7. Done means

The lab suite reproduces every test (death times ±0.2 s, tower HP ±5%, same winner). On held-back replays: fight
winner ≥ 80%, median life time difference within ±0.3 s, big tower damage sim / real 0.9-1.1, position error at +5 s
≤ 0.6. Cole plays 10 play-page games on the new grid and says it feels like the game. Then P4.
