# Client code notes: how the battle library uses its numbers (2026-10-29)

What the game's native battle library (`libg.so`, arm64, dumped decrypted from a rooted BlueStacks; see
decisions.md 2026-10-29) does with LoadTime, HitSpeed, targeting and the King's wake-up, read from the code.
Everything here is in my own words: logic, constants and addresses only. The dump, the rebuilt image, the Ghidra
projects and all decompiler output stay under `data/` (gitignored) and never go in the repo.

Build analysed: BlueStacks 5.21.790.7505, load base `0x71fee9c000`. Addresses below are absolute at that base
(subtract it for the file offset in the rebuilt image). They move with every game update; the *method* below finds
them again.

Confidence words: **sure** = read directly in the code path, no gap; **likely** = code read, one link inferred
(a name, a caller, a struct identity); **open** = not found in code, a lab test settles it.

## How to repeat this after a game update

1. Dump `libg.so` from memory again (the 2026-10-29 decisions entry has the root + `/proc/<pid>/mem` steps) into
   `data/external/cr_client/libg_dump/` with `maps.txt`.
2. `python3 -I scripts/client_re/rebuild_image.py` places each region at its load address and rewrites the program
   headers so file offset = virtual address (section headers are not mapped; they are dropped). Output
   `data/work/ghidra/libg_rebuilt.so`.
3. `python3 -I scripts/client_re/ehframe_funcs.py` reads `.eh_frame_hdr`: every function's start and size (the
   library is stripped; this is the only symbol-free way to get boundaries, ~77k functions).
4. Ghidra 12.1.4 headless, two projects under `data/work/ghidra/`:
   - `proj` (full analysis, ~15 min on the M5): `analyzeHeadless data/work/ghidra/proj libg -import
     libg_rebuilt.so -loader ElfLoader -loader-imageBase <base>`;
   - `proj2` (same with `-noanalysis`): the big table loaders decompile cleanly there (full analysis wrongly marks
     the string destructor as no-return and truncates them).
   - The macOS arm64 build of Ghidra ships no native decompiler: build it once with `make ghidra_opt` in
     `Ghidra/Features/Decompiler/src/decompile/cpp` and copy the binary to `Ghidra/Features/Decompiler/os/
     mac_arm_64/decompile` (it builds as x86_64 and runs under Rosetta).
5. `scripts/client_re/probe.sh <proj|proj2> <out> fn:ADDR callers:ADDR dis:ADDR:N str:TEXT` decompiles / lists
   callers into a file under `data/`; `annot.py` puts the referenced strings back in.
6. The way in:
   - `string_xrefs.py LoadTime HitSpeed ...` finds the table loaders (ADRP+ADD to the field-name string);
   - decompile the loader, `fields.py` gives field -> struct offset (CharacterData, buffs, area effects;
     `--globals` for the globals loader, which looks names up as strings);
   - `ldscan.py w:48c w:418 b:3df` finds the functions reading those offsets together (the combat component);
   - `blscan.py` for direct callers; virtual calls: find the vtable whose slots hold the function (the combat
     component's vtable has its per-tick update in slot 3);
   - game-tag bit masks (`NO_ATTACK`, `INACTIVE`, ...) live in .bss and are filled at start-up from names:
     `gstore.py <mask address>` finds the initialiser, which names each mask.
7. Re-check each answer below at its new address; the logic should be recognisable by its 50 ms constants and the
   offsets from step 6.

## Struct offsets (this build)

CharacterData (buildings, troops and towers share it; loader `0x71ffd5e6fc`): HitSpeed 0x418, Range 0x420,
MinimumRange 0x424, SpecialRange 0x428, SpecialLoadTime 0x430, SightRange 0x434, LoadTime 0x48c, DeployTime 0x490,
DeployTimerDelay 0x494, LoadFirstHit 0x3df, LoadAfterRetarget 0x3e0, CollisionRadius 0x4e4, SightClip 0x4f0,
SightClipSide 0x4f4, TargetOnlyBuildings 0x577, AttacksAir/Ground 0x589/0x58a, AttackFinishTime 0x6b4,
TargetLowestHP 0x6fc, ResetHitTimerWhenNoTarget 0x83d, DeployTimeAffectedByCharacterSpeed 0x83f,
KeepTargetWithPendingDamage 0x840 (default true).

Globals (loader `0x71ffdb7a38`): EXTRA_SIGHT_RANGE_TO_BUILDING 0xe0, EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS 0xe4,
ATTACK_FINISH_TIME_MS 0x2c0 (250), ALLOW_AOE_ATTACKS_WITHOUT_TARGET 0x2e9, ADD_CHARACTER_RANGE_TO_RADIUS 0x378,
LOGIC_LOAD_FIRST_HIT_RESET_TIMER_AFTER_ATTACK 0x3da, ..._WHEN_ZAPPED 0x3db, ..._KEEP_LOADED_AFTER_DISCARD 0x3de,
LOGIC_PENDING_DAMAGE_IGNORE_IF_DURATION_LESS 0x42c (600), LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET 0x430 (25),
LOGIC_PRESERVE_TARGET_IF_HIT_STARTED 0x44c, LOGIC_XPOS_BASED_TOWER_TARGETING 0x44e,
CURRENT_TARGET_IGNORES_PENDING_DAMAGE 0x770, COMBAT_CMP_USE_HIT_STARTED 0x7cc, USE_TARGET_QUEUEING 0x85d.

Combat component (one per attacking object; constructor `0x71fff54bd0`, per-tick update `0x71fff599e4`):
current target +0x10, attack timer +0x24 (ms, counts up), load remaining +0x28 (ms, counts down), forced-target
lock timer +0x1c, special-attack load +0x44, attack-finish timer +0x50.

## 1. The attack cycle

**Functions:** per-tick update `0x71fff599e4`; attack-timer advance `0x71fff5ac3c`; set target `0x71fff558e0`;
target removed (died) `0x71fff591f8`; reset attack `0x71fff5af20`; hit-speed scaling by buffs `0x71fff543e4`;
per-tick target drop when frozen `0x71fff11980`; constructor `0x71fff54bd0`.

How it works (sure, except where marked):
- **Two timers.** An *attack timer* runs only while the unit is attacking, advancing 50 ms each tick scaled by its
  hit-speed buffs. A hit lands each time the timer crosses a multiple of HitSpeed, so a unit that keeps attacking
  hits exactly HitSpeed apart. A *load timer* holds how much load is still missing. It counts down 50 ms every tick,
  unscaled and regardless of what the unit is doing (walking, idle, stunned).
- **Starting an attack.** When an attack starts (timer at 0, target in reach), the attack timer starts at
  LoadTime minus the missing load, and the load timer is set back to the full LoadTime (reloading starts at
  the attack's start, not at the hit).
  - First hit = HitSpeed - LoadTime for a loaded unit; HitSpeed - (load gathered) for a partly loaded one.
  - Load is counted from the start of the previous attack: a unit is fully reloaded LoadTime after its last
    attack *started*.
  - LoadTime >= HitSpeed (Inferno Tower, Sparky-like lock-on cards): the attack waits until no more than HitSpeed
    of load is missing, then starts from 0 (first hit HitSpeed after that).
- **Spawn.** Units spawn loaded (both timers 0), except LoadFirstHit units (only Sparky), which spawn with the full
  LoadTime missing.
- **Retarget:**
  - target dies with another enemy in reach: the target is cleared and an *attack-finish* window of
    ATTACK_FINISH_TIME_MS = 250 ms opens (a unit's own AttackFinishTime if OverrideAttackFinishTime is set). A new
    target found in reach keeps the running timer: the next hit comes HitSpeed after the previous one (full cycle,
    as Cole believed);
  - no target in reach after 250 ms: the attack timer is reset (the swing is lost, load keeps counting);
  - switching to a target out of reach: attack timer reset at once, load untouched;
  - switching to another target already in reach (ordinary unit): the timer keeps running, no reset.
- **Lock / leaving reach.** The current target is kept while within Range + own radius + target radius + 0.025
  tiles (LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET 25).
  - An attack that has started (attack timer past the first 50 ms of a cycle) continues even when the target
    steps out of reach.
  - Units with a projectile also keep a target up to 0.5 tiles past reach when the hit has started
    (LOGIC_PRESERVE_TARGET_IF_HIT_STARTED).
  - Otherwise, out of reach means: attack reset, walk.
- **Stun / freeze (Zap, Freeze, E-Wiz):**
  - The buffs are HitSpeedMultiplier -100 (and SpeedMultiplier -100). The scaling function
    (`0x71fff543e4`) multiplies 50 ms by the strongest boost and by (100 - the strongest slow), so a stun makes the
    attack timer's step 0: **the swing pauses, it is not reset.**
  - In parallel, the character drops its target every tick while its hit speed is 0 (`0x71fff11980`), so the
    combat update re-acquires the nearest valid target each tick. The same target in reach: the timer resumes
    where it was when the stun ends. A different target out of reach: reset.
  - That target drop is the "Zap resets targeting" behaviour.
  - The load timer keeps counting through the stun.
  - LoadFirstHit units (Sparky), with LOGIC_LOAD_FIRST_HIT_RESET_TIMER_WHEN_ZAPPED: the drop resets the
    attack and sets the full LoadTime missing.
  - Likely: the frozen test is "hit-speed factor 0", so pure slows (Ice Spirit, Freeze's tail) do not drop the
    target.
- **Pushback / knockback / being moved:** no special case in the combat code. A started swing continues; if the
  unit is pushed out of reach before the swing starts, it is an out-of-reach target (reset, walk back).
  - Dash and jump states (state 3, 5) reset the attack.
  - The hit itself is cancelled when the target is more than 1.5 tiles past reach
    (LOGIC_CANCEL_HIT_FROM_LONG_DISTANCE_RANGE): likely, that check sits in the hit function `0x71fff57ddc`,
    not read line by line.
- **LoadAfterRetarget** (no card sets it): every retarget would reload fully. **LoadFirstHit** (Sparky): reloads the
  full LoadTime after each attack (LOGIC_LOAD_FIRST_HIT_RESET_TIMER_AFTER_ATTACK).
- **SpecialLoadTime** is the windup of a *special* attack.
  - When the target is between SpecialMinRange and SpecialRange (and NO_SPECIAL_ATTACK is not set), the unit
    enters special mode, counts SpecialLoadTime down by the scaled 50 ms step, then fires the special.
  - Only Fisherman (1.3 s hook) and Fisherbarrel (1.0 s) have one.
- **HitSpeedOffset** is not a troop field: it is in the area-effect schema (offset 0x104 of area-effect data).
  - Values: Poison 250, Lightning 500, Earthquake 1000, Tornado 50, ...
  - By its name and data, the delay before the first pulse. **Open**: its use was not read in code.

**Engine differences** (CombatEntity.h preload model):
- same: first hit HitSpeed - LoadTime when loaded, hits HitSpeed apart while attacking, full cycle after a kill
  with a target in reach;
- **reload is counted from the hit in the engine, from the attack start in the client**:
  - the engine is loaded again HitSpeed after the start; the client LoadTime after it;
  - after a kill or a walk to a new target, the client can strike sooner: Knight up to 0.5 s, Musketeer
    0.7 s, Valkyrie 0.1 s;
- **stun:** the engine cancels the windup and makes the unit wind up the full HitSpeed - LoadTime again (cooldown
  held). The client pauses and resumes the swing;
- no 250 ms attack-finish window in the engine (a kill with no target in reach keeps the cooldown as is);
- range keep +0.025 tiles and the projectile-units' +0.5 keep window are missing (the engine keeps a started hit
  to +1.5 tiles for everyone);
- Fisherman: the engine hooks at once when ready; the client loads 1.3 s first;
- towers: see section 4.

## 2. Tick length and elixir

- **Logic step: 50 ms (likely; per-update step is sure).** Every per-tick timer adds or subtracts 50 ms per
  update: attack timer, load, attack-finish, forced-target lock, deploy timer, special load, King wake-up.
  - Places: `0x71fff599e4`, `0x71fff5ac3c`, the character tick `0x71fff11b58`.
  - Speed scaling takes 50 as its base (`0x71fff543e4`).
  - Not traced: the battle loop calling one update per 50 ms of battle time. The 10-29 video-clock slope (1.000)
    and the 0.1 s replay granularity are consistent with 20 updates a second.
- **Elixir (data sure, code open).** battle_timelines.csv `Default` (Ladder, Friendly, Tournament):
  - ElixirFullBarMS 28000 / 14000 / 9300 for the sections ElixirRateLength 120 / 120 / 60 s: 2.8 s per elixir,
    then 1.4 s, then 0.93 s;
  - StartingElixir **6**.
  - The accrual code was not found (the timeline's accessors are inlined).
- Engine:
  - 0.1 s tick (half the client's resolution);
  - 0.035 elixir a tick = 2.857 s per elixir (2% slow; HUD-measured 2.8 agrees with the table);
  - 2x at 120 s, 3x at 240 s, which matches the table;
  - starts at 5.0, where the table says 6. **Open**: read the HUD at 0:00 on a lab replay (or the elixir reader on
    the first frames) before changing it.

## 3. Target selection

**Functions:** find target `0x71fff56878`; target-valid check `0x71fff55c48` → `0x71fff26670`; range test
`0x71fff27864`; attack range `0x71fff5673c`; priority `0x71fff264e8`; default (blind) target `0x71fff56fd8`; queued
target requests `0x71fff55d3c` / `0x71fff55e80`.

- **When it runs.** Every tick while the unit has no target in reach (including while walking toward one, and
  every tick while stunned), unless a forced-target lock (+0x1c, set by hooks / dashes / taunt-like abilities to
  99999) or the LOCK_TARGET tag holds it.
  - A target in reach is locked: no rescans while it stays in reach (+0.025).
  - Data flags at 0x39d / 0x39c make a unit rescan every tick, or drop its target after each hit. These are
    likely RetargetEachTick / RetargetAfterAttack (both are schema fields; set on the Goblin Cannon, Fisherbarrel,
    Neutral Rage Barbarians), but their offsets were not tied to the names.
- **Which candidates.** Enemies the unit can hit (air / ground, TargetOnlyBuildings / Troops / Towers, not
  UNTARGETABLE, ...), inside a sight circle.
  - Sight circle: SightRange + own CollisionRadius (ADD_CHARACTER_RANGE_TO_RADIUS) + the candidate's
    CollisionRadius, + 2.0 tiles if the candidate is a crown tower, + 0 for other buildings.
  - Sight is therefore **edge to edge**, tested on squared centre distance (sure).
  - Units with SightRangeForCharacters use that (at least their Range) against troops.
- **SightClip / SightClipSide** (Hog Rider, Royal Hogs 4.0 / 4.0; Giant, Royal Giant 2.0 / 2.0; Balloon side
  2.0): the candidate's sideways distance |dx| must be within sight minus SightClipSide, and its distance along the
  lane within sight minus SightClip on one side (sure). Which side, ahead or behind, is open: the facing flag's
  meaning was not resolved.
- **Pending damage** (overkill avoidance): a candidate is skipped if damage already in flight will kill it *and*
  lands within 600 ms (LOGIC_PENDING_DAMAGE_IGNORE_IF_DURATION_LESS).
  - This applies only to attackers that have a Projectile (ranged troops, buildings, towers). Melee units never
    skip.
  - The current target is kept despite pending damage (CURRENT_TARGET_IGNORES_PENDING_DAMAGE +
    KeepTargetWithPendingDamage, default true). Sure, except "all projectile units": the flag is "data has a
    Projectile".
- **Choice:** highest *priority* first, then nearest by squared centre-to-centre distance.
  - Priority 1: buildings for TargetOnlyBuildings units; targets without the buff named by
    DeprioritizeTargetsWithBuff. Everything else 0.
  - Squad members spawned in a ring carry a small distance bias.
  - TargetLowestHP units pick by HP instead.
  - Ties keep the first found.
- **Building placed in front of a building-targeter** (Cannon pulling a Hog): the Hog rescans every tick until
  its current target is in reach. The Cannon wins the moment it is valid, in the clipped sight box, and nearer
  (centre to centre) than the tower. Once the Hog is in reach of the tower it is locked and a new Cannon no longer
  pulls it (sure).
- **Blind default target** (LOGIC_XPOS_BASED_TOWER_TARGETING, LOGIC_PRINCESS_TOWERS_ALWAYS_AS_DEFAULT_TARGET): the
  enemy crown tower with the smallest |dx| (its lane), respecting the lane split; the King is compared by distance
  when princess towers are down (likely, details of the lane rules not fully read).
- **USE_TARGET_QUEUEING is not about pending damage.** With it on, *forced* target requests (abilities, taunts;
  callers `0x71fff50428`, `0x71fff4a488`) are queued with a priority. Once per tick the highest-priority request
  is applied (`0x71fff55e80`) instead of the last one winning (sure).

**Engine differences** (CombatEntity::findTarget, effectiveSightWith, Tower):
- **sight:**
  - the engine is centre to centre plus a fitted TROOP_SIGHT_BONUS of 1.75 between troops;
  - the client is SightRange + both radii (Knight vs Knight: 5.5 + 0.5 + 0.5 = 6.5 tiles centre to centre;
    engine 7.25);
  - crown towers +2 in both;
  - the engine also floors sight at attack reach, the client doesn't;
- **ranking:** the engine ranks by centre distance minus the target's targeting radius (towers 1.5 / 2.0), the
  client by plain centre distance after the priority class;
- **missing in the engine:**
  - building-targeters preferring buildings via priority;
  - DeprioritizeTargetsWithBuff;
  - TargetLowestHP;
  - SightClip / SightClipSide;
  - the 0.025 keep extension;
- **queueing:** the engine's `Fidelity::targetQueueing` (USE_TARGET_QUEUEING) implements a pending-damage skip
  for every attacker with no time window. In the client that rule belongs to the pending-damage logic: projectile
  attackers only, within 600 ms. USE_TARGET_QUEUEING is the forced-target request queue;
- lock-on while in reach and per-tick rescans while chasing: same in both.

## 4. King activation, towers, DeployTime / DeployTimerDelay

- **King activation (sure on the tags, likely on the polling):**
  - king_tower.toml starts the King with the INACTIVE tag and a wait action whose condition is "king damaged or a
    tower destroyed";
  - when the condition holds, the King gets ACTIVATING for ActionDuration 3300 ms, then the tag clears;
  - while INACTIVE or ACTIVATING, the character's per-tick target logic is skipped and COMBAT_DISABLED is set
    (`0x71fff11980`), so it neither targets nor shoots;
  - after waking it acquires like any unit. With LoadTime 500 / HitSpeed 1000 and spawned loaded, **its first shot
    lands 0.5 s after it acquires**, then every 1.0 s.
- **Princess Tower:** its BUILDING entry has HitSpeed 800 and **no LoadTime**. By the attack-cycle rule, a tower
  acquiring from idle shoots **0.8 s after acquiring** (timer from 0 to HitSpeed). One acquiring a new target in
  reach while still in its cycle keeps the cycle.
  - Likely: towers use the same combat component (they share the CharacterData schema and loader), but the tower's
    object construction was not traced.
  - This contradicts Cole's "fires right away" and the 10-29 fix (towers with no load fire at once). **Lab L18**
    (Hog alone to the tower) settles it: the time from the Hog entering 7.5 + radii to the first arrow leaving.
- **DeployTime (sure):**
  - a spawned unit goes through a placing state into a deploying state with a deploy timer = DeployTime (or the
    spawner's override);
  - the timer counts down 50 ms a tick, scaled by the unit's speed buffs only if DeployTimeAffectedByCharacterSpeed;
  - the unit is inert until it reaches 0, then it starts walking / looking for targets;
  - DeployTime 0 skips the deploying state.
- **DeployTimerDelay (likely):** read only by the spawner, which passes it with DeployTime to the unit's
  presentation component (the one that plays animations and effects), not to logic. No card in the client tables
  sets it (all 0), so it has no gameplay effect today.
- **DeployDelay** (400 on most troops, 0x5ac): not traced. **Open**.

**Engine differences:**
- King: wakes 3.3 s after its trigger, as in the client, but then fires at once; the client fires 0.5 s after
  acquiring;
- Princess: fires at once; the client's first arrow comes 0.8 s after acquiring from idle (pending L18);
- deploy: same model at 0.1 s resolution.

## Not answered from code, and the lab test that settles it

| Question | Lab test |
|---|---|
| Battle loop is one update per 50 ms | any lab replay: a Knight's hit times modulo 50 ms vs 100 ms (frame-accurate HUD damage numbers) |
| Starting elixir 5 or 6 | first frames of any lab replay (HUD) |
| Tower first arrow 0.8 s after acquiring | L18 Hog alone to the tower; L04 Giant alone |
| Stun pauses rather than resets a swing | a new test: Mini P.E.K.K.A on a tower, Zap mid-swing; next hit should come HitSpeed - (time already wound) after the stun ends |
| SightClip direction | a Hog passing a Cannon placed behind vs ahead of it (L09 variant) |
| HitSpeedOffset use | L13 (Fireball / Arrows on a tower) has no pulses; a Poison or Earthquake on a tower: first damage tick time |

## 5. Unit collision (body pushing), avoidance, river and jump (2026-10-29)

Asked because of two play-page reports from Cole: a Mini P.E.K.K.A in front of a Hog slows the Hog (not in the real
game), and a Mini P.E.K.K.A and a Battle Ram reaching a bridge together get stuck. Units: client positions and radii
are in 1/1000 tile (the engine's `bodyRadius = CollisionRadius / 1000`); one logic tick is 50 ms.

**Functions:** move component per-tick `0x71fff5e134` (vtable slot at `0x72008f0a30`); walking tick `0x71fff5ea20`;
knockback tick `0x71fff5e49c`; **separation (collision) `0x71fff6058c`**; avoidance `0x71fff600b8`; apply one move
step `0x71fff60bd4`; terrain move with per-axis cell clamp `0x720017de58`; cell has river flag `0x71ffe0ceac`; ability
push into the same accumulator `0x71fff61d5c` (caller `0x71fff0bd18`); knockback start gates `0x71fff61750`,
`0x71fff61c7c`; jump start `0x71fff61220` (from the walking tick). Character virtuals used: radius (slot 0xa0,
`0x71fff146c0`, CollisionRadius 0x4e4), Mass (0xb0, `0x71fff146e8`, CharacterData 0x4ec), step length (0xb8,
`0x71fff146f4`), avoidance on (0xd0, `0x71fff149f0`), collisions on (0xd8, `0x71fff14a88`), collidable (0xe0,
`0x71fff14b44`), avoidable (0xe8, `0x71fff14cb4`), flying (0x108, `0x71fff14224`), hovering (0x118, `0x71fff142fc`,
data Hovering 0x3a1). Height above ground `0x71fff26100`. Tag masks (initialiser `0x71fff04350`): NO_CHECKCOLLISIONS
`0x7200a26178`, NO_CHECKAVOIDANCE `…180`, NO_PUSHED_BY_ENEMY `…190`, NO_PUSHBACK `…228`,
DISABLE_PHYSICAL_INTERACTIONS_WITH_OBJECTS `…248`, NO_PUSHED_BY_ALLY `…268`, AVOIDANCE_AS_OBSTACLE `…270`; code reads
them through GOT slots `0x7200936640…66f0` (gstore.py on the GOT address finds the readers).
Move component fields: push sum x/y +0x1d4/+0x1d8, contact count +0x1dc, charge progress +0x1e0 (>= 10000 = charging),
avoidance steer +0x1f4, ability-push flags +0x216 (clamp to terrain) / +0x217 (no length cap), path +0x38/+0x3c.

**Separation (sure, read line by line in `0x71fff6058c` and `0x71fff60bd4`):**
- Each unit moves **only itself**, inside its own move tick, before its own step. Units are updated one after another,
  so a later unit sees the positions earlier units already moved to this tick (the order over units is likely the
  object list order; the battle loop was not traced).
- Neighbours: everything within own radius + 0.02 tiles, same layer (both on the ground or both above it, by height,
  so a jumping unit is in the air layer), collidable, not the same object.
- Per neighbour: overlap = own radius + its radius - centre distance, clamped to [0, 0.3 tile]; multiplied by
  **its Mass / own Mass** (not a share of the total), capped at 0.299; the push is (that + 0.001) along the line from
  it to self. Against something with no move component (a building), own radius counts at most 0.5.
- The pushes are **averaged** over all contacts (sum / count), the average is capped at **0.15 tile per tick**, and it
  is added to the unit's walking step for the tick. One pass, no iterations.
- Same rule for friend and enemy. The only team difference is the tags: NO_PUSHED_BY_ALLY / NO_PUSHED_BY_ENEMY on self
  = ignore that team's bodies; on the other unit = it counts as immovable (full overlap, no mass scaling). No card in
  our decks has them (evo Valkyrie, hero Wizard do).
- Coincident centres: pushed 0.001 along y, the sign by the unit's side.
- Who does it: walking, idle and attacking units all run it (the push is applied even with a zero step), charging
  units too. Not run / not counted: NO_MOVE units, NO_CHECKCOLLISIONS, units attached to another, **jump (state 5)**,
  dash (state 3, unless a dash data value is set), state 12, DISABLE_PHYSICAL (also set during the deploy states 6/7:
  likely), units winding up a special attack (Fisherman).
- Mass 0 on self: the push is just the 0.001 (likely only matters for buildings; how a building reports Mass through
  slot 0xb0 is **open**).
- Consequence (my integer re-implementation, 1-D, both update orders): a Hog (120) running into a walking Mini
  P.E.K.K.A (90), equal Mass: the Mini P.E.K.K.A is shoved forward at the Hog's speed and the Hog keeps its own
  speed. Behind a Giant (Mass 18) the Hog drops to the Giant's pace. Behind a Battle Ram (6) the Hog still keeps 120.

**Avoidance (sure on the code, likely on the geometry, `0x71fff600b8`):** only while walking (states 1, 4, 10, 11).
It looks at a circle of the unit's radius (max 0.5) 0.256 tile ahead. A blocker is any static object, or a unit that
is not moving the same way (facing dot product <= 0, or idle / attacking / deploying) and is at least as heavy, the
weight test applying only while self is charging. It sets a steering value of +-200 / 256 (side: by which side the
blocker is on, or the blocker's own steering side); the step is turned toward the side by that weight and renormalised.
The value decays by 10 a tick. Units walking the same way are never avoided: they are pushed.

**River and bridge (sure on the code, `0x71fff60bd4`, `0x720017de58`; flags from the tilemap):**
- Map: 36 x 64 cells of 0.5 tile (`assets/tilemaps/tilemap.csv` in the asset pack, LZMA). River = flag 32, rows 30-33
  (2 tiles). Bridges are cells 5-8 and 27-30, all four river rows: 2 tiles wide, x 2.5-4.5 and 13.5-15.5 client tiles,
  centred on the princess towers' columns.
- Walking follows a cell path and does not test terrain. A collision push from a dry cell is **not clamped**: a unit
  can be shoved with its centre into a river cell.
- When the centre is already in a river cell and the unit is being pushed: the move is clamped per axis at the cell
  border (it cannot enter another river cell, the other axis keeps moving: a slide), and it is nudged 0.256 tile a
  tick in y toward the nearer bank (by which half of the map it is in). Flying and Hovering units skip all of this.
- Ability pushes (`0x71fff61d5c`, e.g. the evo Ram's PushToSide) always clamp and skip the 0.15 cap. A knockback that
  ends in a river cell relocates the unit to the nearest free cell (`0x71fff5e49c`).

**Jump (sure):** JumpEnabled units (Hog, Royal Hogs, Ram Rider, Prince, Dark Prince, ...) path straight across the
river. When the next path cell is a river cell, the unit jumps (state 5) to the first dry cell beyond it, at its
JumpSpeed (Hog 160 vs walking 120). While jumping it is neither pushed nor pushes (state 5 is not collidable, and it is
off the ground). The Hog's `JumpHack_*` actions (actions.toml) give it a 50 ms IgnorePushBack buff every tick it is
jumping, which blocks knockback, not body pushes. On a bridge cell the Hog just walks.

**The fields asked about:** IgnorePushback (CharacterData 0x58e), the IgnorePushBack buff and NO_PUSHBACK gate
knockback only (`0x71fff61750`, `0x71fff61c7c`), not body collision. CheckCollisions is a projectile field (Hunter).
PushToSide and FullPushBackCollisionCheck are fields of the ActionDamagingPushBack ability (evo Battle Ram, hero Dark
Prince), not of unit collision. MoveSpeedByFriendlyMass / MoveRadiusByFriendlyMass exist in the schema, unused by our
cards.

**Engine differences** (Board.h resolveCollisions / clampToBoard, Troop::moveTowards):
- **share:** the engine splits each pair's overlap by m_other / (m1 + m2) (halves at equal Mass), so the pair always
  separates by exactly the overlap; the client moves each unit by its own m_other / m_self share, sequentially. This is
  the Hog case: measured on the engine, a Hog injected 1.2 tiles behind a Mini P.E.K.K.A covers 9.75 tiles in 3.9 s
  vs 10.34 alone (-6% over the run, about -12% while in contact), and the Mini P.E.K.K.A is dragged 8.1 vs 7.7;
- **when:** the engine resolves after every unit has moved, once per 100 ms; the client resolves per unit, before
  its step, every 50 ms;
- **limits:** the engine resolves the full overlap with no cap; the client caps the overlap at 0.3, the share at
  0.3, averages contacts and caps the result at 0.15 a tick;
- **noise:** the engine adds a 0.01 sideways nudge per pair (0.05 against buildings); the client has none;
- **avoidance:** missing in the engine;
- **river:** the engine snaps any troop in the river band off a bridge to the nearer bank in one step; the client lets
  a push carry the centre into the first river cell, then clamps per axis and nudges 0.256 a tick;
- **bridge position (likely):** the tilemap puts the bridges at client tiles 2.5-4.5 / 13.5-15.5 (engine x 2.0-4.0 /
  13.0-15.0, by geometry.py's 0.5 offset); the engine has 1.5-3.5 / 13.5-15.5. The replay-measured crossings
  (geometry.py: 3.7 / 14.2 in our frame) sit nearer the tilemap's centres (3.5 / 14.5) than the engine's (3.0 / 15.0);
- **jump:** the engine's Hog has riverIgnores and walks straight with collisions on; the client's Hog is
  non-colliding and at JumpSpeed over river cells;
- **deploying units** collide in the engine; in the client they don't (likely).
- The Mini P.E.K.K.A + Battle Ram stall was **not reproduced** on the engine (600 injected same-team placements around
  both bridges, enemy head-on cases too): the cause is open. Likely candidates are the bank snap and the bridge offset.

**Lab test (proposed L22, Hog behind a walking Mini P.E.K.K.A):** lane right, Cole's deck as in L18. Cole plays the
Mini P.E.K.K.A at [14, 21] and the Hog Rider at [14, 23] together at t = 25 (both 4 elixir; Hog 2 tiles behind, same
column). Nothing else. Code prediction: the Hog reaches the BlueStacks right tower as early as a lone Hog from
[14, 23] would (sim with the Mini P.E.K.K.A removed). The Mini P.E.K.K.A crosses the bridge pushed at Hog speed:
about 1 tile further than walking alone at play + 4 s, about 2 at play + 6 s. Current engine (measured, deploy
included): the Hog is 0.5 tile behind its lone run at play + 4 s and 1.0 at play + 6 s. Compare the positions at
play + 1 / 2 / 4 / 6 s (lab_compare) and the Hog's first-hit time.
