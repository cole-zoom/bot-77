# Decisions log

What Cole decided, contributed, and reviewed while building bot-77, in order. Numbers come from
the scoring runs at the time (`uv run bot77 benchmark`, review exports in `data/ground_truth/`).
Technical detail lives in `docs/play-detector-spec.md` and `docs/opponent-plays-spec.md`.

## 2026-09-24 — The idea and the plan

- **Goal:** train a model to get good at Clash Royale from the large amount of pro gameplay on
  YouTube, and use LanceDB (open source) to mine and label moments as scenarios (e.g. being
  outcycled).
- **Build order:** Cole wanted the analysis engine (C: turn videos into a play-by-play log in
  LanceDB) and wanted "the foundation right before we move on." He first proposed A → B → C;
  since the predictor (A) and the autonomous player (B) both need C's data, the order became
  C → A → B.
- **Compute:** local first (M5 MacBook, 16 GB), rent cloud GPU if needed (budget: a few hundred dollars).
- **Scope: one deck.** Hog 2.8 EQ — Hog, Earthquake, Tesla or Cannon (evo), Firecracker (evo),
  Skeletons, Electro Spirit, Mighty Miner (champion), and Log or Barbarian Barrel. Cole plays
  this deck himself (peaked 1800 Ultimate Champion, ~10k wins), so he can judge the output.
- **Chose the source creators:** Alina (@AlinaCR1) and Ian77, both strong Hog EQ players, and
  chose to stay with YouTube rather than in-game replays because there's far more of it.
- **Asked to spec before building** ("better to get it right the first time").

## 2026-09-24 — Domain knowledge Cole supplied (and corrections he made)

- **Corrected a wrong assumption:** card-name labels ("Hog Rider lvl.16") only appear for your own
  plays, not the opponent's. So opponent plays need sprite recognition.
- **Deck structure:** 1 evo slot, 1 hero/champion slot, 1 flex slot. The first champion/hero
  ability button sits on the right; a left button only appears with a second ability.
  Tesla and Cannon are never in the same deck.
- **Card API quirks:** `maxEvolutionLevel` 1 = evo, 2 = hero, 3 = both (confirmed on all 123
  cards); `maxLevel` counts upgrades, so starting levels are 1/3/6/9/11 by rarity; Heal Spirit's
  id is in the spell range because it used to be the Heal spell; Mirror costs the mirrored card + 1;
  tower troops are `supportItems`.
- **Supplied the evo, hero and champion tables** (cycles, ability costs, descriptions; Boss Bandit
  has 2 ability uses with a 3 s cooldown) — kept verbatim in `data/cards/sources/`.
- **Confirmed** evo charge pips, champion frames, and the "slot empty after a play" signal.
- **Suggested a Princess Gambit tournament video** (random 40-card pool) as a stress test for card
  recognition. It found the hero-form detection worked, and caught a next-card bug at small sizes.
- **Verified uncertain reads** in that video (hero Knight, evo Goblin Cage, evo Bomber).

## 2026-09-24 — Review round 1: Alina, matches 1–2

- Cole reviewed all 124 detected events (118 real plays).
- **Before fixes: 87% precision, 93% recall.** High-confidence events were already 83/83.
- **Findings from Cole's notes** that drove the fixes:
  - After a play, the elixir bar's fractional sliver vanishes and spent segments drain pale; this
    caused fake Mighty Miner abilities and cost mismatches.
  - Emotes cover the hand and elixir bar and were read as plays.
  - When a card disappears from the hand, it was played, even if the cost looks off.
  - A Skeletons + Mighty Miner ability merge, a Log played on a bush, cards dragged mid-slide.
- **Decision:** recalibrate before reviewing more, so the rest of the reviews are on better output.
- **After fixes: 100% precision, 99.2% recall** on those matches.

## 2026-09-25 — Review rounds 2–3: held-out matches

- **Alina matches 3–4** (not tuned on): **99.1% precision, 97.3% recall** on first scoring.
  Cole spotted the pattern: "Mighty Miner ability then Hog" read as a 5-elixir Hog. → Built an
  ability-button reader; then 100% / 100%.
- **Alina match 5** (fully held out): **72/72**. Cole flagged one Tesla + Mighty Miner pair with
  the right cards but swapped order and elixir split.
- **Ian77, matches 1–3** (second creator, split layout, ~3k opponents, jump-cut edits, fully held
  out): **188/188**. Cole's findings: a card selected when the game ended looked like a play; a
  Barbarian Barrel released with a Hog was missed; low-confidence drops at edit cuts.
- **Cole's conclusion:** "any high conf is correct" — backed by 384/384 high-confidence events.
- **Total POV ground truth: 490 events, 100% precision and recall** after the fixes.

## 2026-09-25 — Moving to opponent plays

- **Decided the POV detector was solid** and moved on to detecting the opponent's plays.
- **Confirmed** the "−N elixir" popup is POV-only, and pointed out the **purple diamond** that pops
  up when the opponent plays an evo card.
- **Added the KataCR paper** (Wu et al. 2025, arXiv:2504.04783) and asked that it always be
  cited. Its code and dataset are MIT; the PDF stays local (arXiv's license doesn't cover
  redistribution, and the repo is public).
- Approved the opponent plan: spawn events → card identity with a deck constraint → opponent elixir.

## 2026-09-28 — Opponent review round 1: Alina, match 1

- Asked for review-page changes: a guessed opponent deck that updates as he corrects cards,
  corrected cards first in the picker, and a one-click "Not a play" button.
- Reviewed all 64 detected opponent deploys and added 6 missed plays (52 real plays).
- **Result: 29 right (45%), 18 wrong card, 17 not a play.** It found 89% of real plays but named
  only 55% correctly end to end.
- **Cole's findings:** many "wrong player" errors (Alina's own units tagged as enemy); Goblinstein
  is two entities (Doctor and Monster) but one card; **the 2024 dataset has neither Goblinstein nor
  Minion Giant**; units split behind the king tower or in crowds were missed.
- **Cole's call:** the 2024 detector is missing a lot and is outdated — proposed dropping it.
  (Decision on what replaces it: see below.)

## 2026-09-28 — Replace the 2024 detector (decided)

- **Decision (Cole):** stop relying on KataCR's 2024 detector weights. Build our own detector
  the paper's way [Wu et al. 2025, §3]: current classes (all 123 cards, sub-units, evo and hero
  forms), trained on KataCR's MIT sprite slices for the cards they cover, plus new slices cut
  from our footage for the missing ones, plus generated arenas.
- **Team** comes from our own signals (badge colour and a cross-check against the POV play log),
  not the detector.
- **Labels** come from reviews: Cole's corrections say which card was where, and the POV log says
  exactly when and where our own cards land. No manual box drawing planned.
- **Quick wins first**, before the new model: drop "opponent" deploys that are really POV plays,
  and recognise composite cards (Skeleton Army, Minion Horde, Goblin Gang, Three Musketeers)
  by unit count.
- Benchmark: `data/ground_truth/opponent_truth_uz4VVzGlOjE_m01.json` (52 plays).
  Baseline: 29/52 fully right.

## 2026-09-28 — Building training data for cards the old detector doesn't know

### Quick wins (done)
- **POV cross-check:** an "enemy" deploy of the same card within −1.5…+4 s of one of our own
  logged plays is our unit. It dropped 5 fake opponent deploys (3 Mighty Miner, mostly his ability
  resurfacing; an Earthquake; an Electro Spirit). Fake deploys 12 → 9; deploy precision 81% → 85%;
  no real plays lost.
- **Composite cards by unit count:** Skeleton Army (≥6 skeletons), Minion Horde (≥5 minions),
  Three Musketeers (3), Goblin Gang (goblins + spear goblins).
- **Opponent scorer** (`bot77.review.score_opponent`) against Cole's 52 labelled opponent plays.

### Sprite slices, round 1 — detector boxes + SAM (done)
- Downloaded KataCR's MIT detection dataset: 4,627 RGBA sprite slices over 154 classes (median 22
  per class), 28 empty-arena backgrounds, 7,380 hand-labelled real frames (validation).
- Installed SAM 2.1 tiny. For every opponent play Cole corrected to a card the 2024 detector
  lacks, the tool follows the units born with that deploy and has SAM cut them out once a second.
- **Cole asked whether the level-16 badge belongs in a slice.** No: badges and HP bars are their
  own classes in the paper's generator [Wu et al. 2025, §3.1], and a red badge would bake the
  team into the sprite. The detected badge/bar boxes are now erased from every slice.
- **Cole's gallery review:** Minion Giant 5 of 7 kept (rejected: one missing its collar, one mostly
  background). Goblinstein 0 of 7 kept: the old detector boxed spell effects, tiles, or half the
  unit, so SAM had nothing clean to cut (one Doctor was just a face; one Monster had a hole where
  the deploy clock sat). Cole's rule: keep only if the whole unit is there.

### Sprite slices, round 2 — click-to-cut (done)
- For cards the old detector can't box, Cole clicks the unit and SAM cuts from the click point.
  Page: 6 frames after each corrected Goblinstein play, entity picker (Doctor / Monster).
- **Cole clicked 34 units on 17 frames (17 Monster, 17 Doctor)** →
  `data/ground_truth/reviews/clicks_goblinstein_uz4VVzGlOjE_m01.json`. SAM produced 32 slices;
  deploy clocks, badges and HP bars the detector saw are erased, and only the largest mask piece
  is kept. Awaiting Cole's keep/reject in the gallery.

### Planned next — mining training data with LanceDB vector search (Cole: "make sure to record we did this")
This is the project's original idea — *use LanceDB to mine moments from pro footage* — turned on
the training data itself.

**The problem it solves.** A card the 2024 detector has never seen (Goblinstein, Minion Giant, the
other 11 new cards, 27 evo forms, 17 hero forms) shows up across hours of footage, but the old
detector either misses it or calls it something else (Goblinstein was labelled electro-giant,
lightning, royal-guardian; Minion Giant was labelled electro-dragon or earthquake). Reviewing every
match by hand to find them doesn't scale.

**How it works.**
1. **Candidate crops.** The unit detector runs over every match (4 frames/s; running now on
   Alina matches 2–5 and all 8 Ian matches), and its boxes are stored in the LanceDB table
   `unit_detections`. Every box, whatever class the old detector gave it, is a candidate unit crop.
2. **Embed every crop.** Each crop is turned into an image embedding (a vector that captures what
   the unit looks like), and the vectors are stored in LanceDB next to the crop's match, time,
   position, team and old-detector class.
3. **Queries = Cole-approved slices.** The approved slices (5 Minion Giant, plus whichever
   Goblinstein Doctor/Monster slices Cole keeps) are embedded the same way.
4. **Vector search.** For each card, LanceDB returns the crops nearest to its approved slices,
   across all footage — including crops the old detector mislabelled — filtered with SQL (enemy or
   friendly, match, time range, crop size).
5. **Human in the loop.** The top candidates go to the gallery; Cole keeps or rejects each. Kept
   ones become new slices *and* new queries, so each round finds more varied poses and angles.
6. **Stop** once each new class has enough slices (KataCR's median is 22; target 20–50 per
   entity), then generate synthetic arenas [Wu et al. 2025, §3.1, Alg. 1] and train the new detector.

**Why LanceDB fits.** Vectors, metadata (match, time, team, box) and the crop images live in one
table on disk; search is local and fast on the M5; SQL filters and vector search combine in one
query; and every round's approvals are just rows, so the dataset's history is kept.

**What Cole does:** approve or reject candidates in the gallery — no box drawing.

### LanceDB vector search — built and first results (2026-09-28)
- Cole approved 3 Goblinstein Monster + 5 Doctor slices from his clicks (a leftover "16" badge the
  detector hadn't boxed was stripped automatically, plus crown remnants on three Doctors).
  Approved set: Minion Giant 5, Doctor 5, Monster 3.
- Built `bot77.mine`: every unit box in `unit_detections` is cropped once a second and embedded
  with DINOv2 ViT-S/14; **15,728 crops from Alina's 5 matches** are in the LanceDB table
  `unit_crops` (3 minutes on the M5).
- Search with the 13 approved slices as queries (max cosine similarity, one hit per match-second,
  crops next to the query slices excluded).
- **First result: the top ~20 Minion Giant hits are all Minion Giants, including a batch from
  Alina's match 3 — a match nobody had reviewed — every one mislabelled "electro-dragon" by the
  2024 detector.** The idea works: LanceDB found a card the old model doesn't know, in footage no
  one looked at.
- 90 candidates (30 per class) went to Cole's gallery for approval.
- Ian's 8 matches detected and embedded: **22,270 more crops (37,998 total in `unit_crops`)**.
  Searching his footage found **Goblinstein Monsters and Doctors in Ian's games** (the old detector
  called them baby-dragon, royal-recruit, mighty-miner) — no Minion Giant there. 80 Goblinstein
  candidates from Ian went to a second gallery.
- Cole asked whether recording his own games to cover every card is worth it. Answer: yes, aimed
  at the gaps (the 13 new cards, evo and hero forms) — mining can only find cards that are in the
  footage, and his own plays come with exact labels from the POV detector.

### Mining round 1 results, and Cole's own footage (2026-09-28)
- Cole reviewed the mined candidates: **35 of 170 kept (21%)** — Alina 16/90, Ian 19/80. Cole's
  view: "not the best." After SAM cut the kept ones, a final pass rejected 7 more (SAM spilling
  into neighbours or effects, and gold level badges on Ian's footage).
- **Approved slices now: Minion Giant 14, Goblinstein Monster 15, Goblinstein Doctor 12**
  (from 5 / 3 / 5 before mining).
- **Cole recorded ~40 min of his own ladder games** (iPhone Mirroring on the Mac, 3024×1964 at
  120 fps; the phone screen is a fixed 876×1900 region). He faced **Goblinstein, Minion Giant,
  Rune Giant, Spirit Empress and Ronin** — three of those have no slices at all yet. He plans to
  cherry-pick other cards later.
- The phone region is cropped and scaled to 1080×2340 so Alina's calibrated layout applies
  unchanged (aspect ratios differ by 0.1%).
- Cole gave the times of the new cards in his recording: **Ronin** (game starts 3:50),
  **Spirit Empress** 8:13, **hero Tombstone + Tomb Queen ability** 15:53, **Rune Giant** 18:52,
  **Goblinstein** 35:20. A click-to-cut page with 50 frames around those times was built straight
  from the original recording (cropped exactly like the converted video, so clicks line up).
  First hero-form entities: `tombstone-hero`, `tomb-queen`.

## 2026-09-28/29 overnight — Cole asleep; everything that didn't need him

- **Cole's recording processed.** iPhone Mirroring puts the game HUD at different offsets than
  Alina's native capture (top HUD +80 px, hand bar −52 px after scaling), found by template-matching
  Alina's HUD patches into his frame; new layout `cole_iphone_mirroring`. Result: 11 matches, his
  deck (Hog EQ with Barbarian Barrel) read every match, 18,654 unit crops embedded, 300 opponent
  deploys from the old detector (its inferred decks are noisy — effects read as Earthquake/Poison/Tornado).
- Cole corrected me: the times he gave were **game starts**, not when the cards appeared. The click
  page was rebuilt from each named game's opponent deploys (80 frames, 2 s after each deploy).
- **Mining round 2** (41 approved slices as queries, all four videos, already-reviewed crops skipped):
  120 candidates. **The top Minion Giant hits are from Cole's own games 1 and 6**, where he faced it.
- **Synthetic dataset v1:** KataCR's generator [Wu et al. 2025, §3.1, Alg. 1] run unchanged except for
  3 added classes (minion-giant, goblinstein-doctor, goblinstein-monster, slices rescaled to the
  generator's 568×896 arena). 12,000 images (~74 boxes each), validation = 800 of KataCR's real
  hand-labelled frames.
- **Detector v1 training:** YOLO11s at 896 px on the M5's GPU. Batch 16 needed 16 GB and thrashed
  (914 s/step), so batch 4 (1.8 steps/s, ~28 min/epoch); 12 epochs fit overnight. No faction output —
  team comes from the colour of the level badge the model detects above each unit, then the POV
  cross-check. Evaluated automatically on the 52-play opponent benchmark when done.
- Morning review queue written to `REVIEW_TODO.md`.

### Detector v1 results (2026-09-29, morning)
- Trained 12 epochs in 5.2 h. On 800 of KataCR's real hand-labelled frames: **mAP50 0.70–0.73,
  mAP50-95 0.46** (the paper's larger, size-split models reach AP50 0.83–0.85 [Wu et al. 2025, Tab. 2]).
- On Cole's 52-play opponent benchmark (Alina match 1):

  | detector | plays found | fully right | wrong card | not a play | missed |
  |---|---|---|---|---|---|
  | KataCR 2024 | 96% | 31 (60%) | 19 | 9 | 2 |
  | bot77 v1 alone | 62% | 22 (42%) | 10 | 12 | 20 |
  | **2024 + v1 for the new classes** | 88% | **32 (62%)** | **14** | 10 | 6 |

- Per card, **v1 learned Goblinstein (2/3 right vs 0/3)** but is worse on small units (Skeletons,
  E-Spirit, Barbarian Barrel), on spells, and still 0/3 on Minion Giant. It runs 4.7× faster.
- Why v1 alone is weaker: trained only on synthetic arenas (domain gap to real footage), a small model,
  12 epochs, and one model for all sizes where the paper splits by size because small units suffer.
- Current best: the ensemble (old detector + v1 for classes the old one lacks). Small gain, and the
  inferred opponent deck now includes Goblinstein.
- Cole: **Spirit Empress has two forms (ground and air)** → two entities `spirit-empress-ground` and
  `spirit-empress-air`, both mapped to the card (like Goblinstein's Doctor/Monster); the form is also
  worth logging for gameplay. Hero Tombstone likewise → `tombstone-hero` + `tomb-queen`.
- Cole: the first click page's Ronin frames were better — Ronin was only played early in that game —
  so Ronin got its own page with 26 frames every 3 s from the start of the game.
- **Cole agreed to the "be safe before paying" plan:** a go/no-go checklist before any rented-GPU run
  (≥40 approved slices per new entity, ≥2,000 auto-labelled real frames with a 100-frame spot check,
  a second opponent benchmark held out of training, a local 1-epoch test and a ~$1 cloud dry run).
- Cole gave **exact card times** in his recording (the earlier pages still missed the cards):
  Ronin 5:09, Spirit Empress air 8:42 and ground 9:55, Tomb Queen 17:13, Rune Giant 20:43 ("until he
  crosses the bridge, for a good view"), Goblinstein 36:05 and 2:15. Little Prince + Guardian at 20:06
  skipped: the 2024 dataset already has `little-prince` and `royal-guardian`.
  → `data/reviews/click_cole_v3.html`, 106 frames, 1 per second (22 for Rune Giant). This page replaces
  the two earlier click pages for his recording.
- Cole clicked **104 units** on the exact-times page (Ronin 13, Spirit Empress air 11 / ground 11,
  Tomb Queen 11, Rune Giant 20, Goblinstein Doctor 20 / Monster 18) → SAM cut 102 slices, now in
  Cole's gallery. Spirit Empress cuts are clean; several Ronin cuts are only his hat.
- Cole showed **Goblinstein's Lightning Link ability** (the beam between Doctor and Monster). Planned as
  its own class `goblinstein-ability`, like KataCR's `skeleton-king-skill`, so opponent ability uses
  can be logged. A thin beam won't SAM-cut cleanly; it'll be labelled as a box on real frames instead.

## 2026-09-29 — Mining hits diminishing returns; coverage check → footage list
- Mining round 3 (all 8 new entities, 41+ queries): Cole kept **36 of 156**. Strong for Minion Giant
  and Goblinstein; for Ronin, Spirit Empress, Tomb Queen and Rune Giant only the top few hits were
  real. **Cole: "a lot of them don't actually have more footage, so mining is kind of useless"** —
  correct: those cards barely appear in our footage, and similarity search can only find what's there.
- Approved slices now: Goblinstein Doctor 41 / Monster 34, Minion Giant 22, Rune Giant 13, Spirit
  Empress air 11 / ground 10, Tomb Queen 7, Ronin 6.
- **Coverage check against all 123 cards** (KataCR's 2024 slices + ours): 31 cards at ≥40 slices,
  41 at 20–39, 38 under 20, 13 with none (8 genuinely new: Berserker, Boss Bandit, Goblin Curse, Goblin
  Demolisher, Goblin Machine, Suspicious Bush, Vines, Void; 5 composites). 27 evo forms and all 17 hero
  forms have no slices. Written up as a recording list: `docs/footage-needed.md`.

## 2026-09-29 — Cole + friend dual-POV recordings; farming evo/hero footage from YouTube
- **Cole's idea: he and a friend both play the new cards in friendly battles, both recording.**
  Each POV log labels its own plays; syncing the two recordings on the in-game clock turns each side's
  plays into exact **opponent** labels for the other — opponent ground truth with no review.
  Built `bot77.pairing` (match pairing by constant clock offset, then opponent truth files
  `data/ground_truth/opponent_truth_auto_<match>.json`).
- Recordings: Cole 3:29 PM (14 min) and friend 3:12 / 3:23 PM (10 / 19 min), friend's Mac at
  2940×1912 (phone region 852×1848). Cole's 3:22:59 recording is black after the first seconds
  (mirroring window) — skipped. Friendly battles on a new teal arena skin; Void, Vines, Goblin
  Demolisher, Rune Giant etc. visible.
- **Layout calibration automated** (`bot77.autolayout`): the hand-made method for Cole's iPhone as
  code. HUD templates for unique elements (timer, chat, "Next:", elixir drop) give the UI scale and the
  top / hand offsets; the tower HP bars are found by *colour* (red opponent bar top-left, blue ours
  lower-left) because templates failed on a different arena skin; only in-game frames are used (menus
  fooled it). Agrees with the hand-made layout to within 0–22 px; friend's and Oyassuu's layouts
  verified on calibration sheets.
- **Cole picked 11 YouTube videos** for evo and hero forms, mostly raw footage without facecam:
  OYASSUU (2.6 Hog: evo Cannon, hero Ice Golem, evo Skeletons; 2.1 Log Bait: evo Dart Goblin, evo
  Goblin Barrel; evo Lumberjack; Queen + evo Ice Spirit), Ryley (evo Barrel, hero Knight, evo Princess;
  evo Royal Giant, hero E-Wiz, evo Royal Ghost, and opponent evo Zap, hero Mega Minion; evo Bats, evo
  Tesla, hero Ice Wizard; Little Prince, evo Ice Spirit; hero Goblins, evo Dart Goblin, evo Skeleton
  Barrel; a 5-deck video with evo Furnace, hero Valkyrie, evo Skeletons, evo Baby Dragon, evo Knight,
  evo Witch, evo Snowball; evo Royal Hogs, hero Berserker).
- **Hero Electro Wizard isn't released yet** (a few days out), so it's not in the card API snapshot.
  Decision: add it as its own class now; the card data catches up with the next API snapshot.
- **Bug found:** LanceDB's `table_names()` returns only the first 10 tables by default; with 12 tables
  the pipeline thought `videos` didn't exist and crashed at the save step, after the full frame pass,
  losing that run for all three recordings. Fixed everywhere (`table_names(limit=10_000)`); rerun two at
  a time. The decks read correctly: Cole (Rune Giant, Minion Giant, Goblin Machine, Suspicious Bush,
  Goblinstein/Berserker, Spirit Empress, Boss Bandit, Ronin) and his friend (Vines, Void, Goblin Curse,
  Rune Giant, Berserker, Goblin Demolisher, Goblinstein, Mirror) — **every Tier 1 new card is covered.**
- **Pairing, first attempt was wrong:** each of Cole's games paired with several of his friend's at
  different offsets. The match clock runs at real speed in every game, so any two overlapping matches
  line up at *some* constant offset. **Fix:** the recordings' wall-clock start times (from the file
  names) give the expected offset (Cole 3:29:18 − friend 3:23:01 = 377 s); pairs must sit within 20 s of
  it. Result: 3 pairs at 375–376 s. Independent check: in all three, Cole's opponent reads as his friend
  ("Nathan'S gas") and the friend's opponent as Cole every time. Friend's 3:12 recording and 3 of his
  3:23 games have no partner (Cole's 3:22:59 recording was black).
- **123 exact opponent labels with no review**, covering every Tier 1 card: Berserker 20, Rune Giant 11,
  Goblinstein 11, Goblin Demolisher 11, Ronin 10, Boss Bandit 8, Goblin Machine 8, Vines 7, Goblin Curse
  6, Void 6, Minion Giant 6, Spirit Empress 5, Suspicious Bush 5, Mirror 3, abilities 6.
- Paused here at Cole's request (going home); the YouTube batch (4.6 h of footage, two at a time)
  waits for his go.

### Between sessions: which cards to record next (2026-09-29, afternoon)
- Cole asked whether more recordings would help. Counted POV plays per new card across his, his friend's
  and his ladder recordings against a ~15-play target: done — Berserker 32, Goblin Demolisher 25,
  Goblinstein 24, Rune Giant 22; nearly — Void 14, Vines 13; short — Goblin Curse 11, Ronin 10, Boss Bandit 8,
  Goblin Machine 8, Minion Giant 6, **Suspicious Bush 5, Spirit Empress 5** (both forms). Hero forms: only
  Berserker (9); evo forms: only Firecracker and Tesla. Suggested one deck built from the short cards,
  2–3 plays of each per game, both forms of Spirit Empress, abilities used, ~5–6 games, both recording.

## 2026-09-29 evening — 75 more minutes of dual-POV footage; everything processed in parallel
- Cole and his friend recorded ~38 min each (6:10 / 6:11 PM), aimed at the short cards (Suspicious Bush,
  Spirit Empress both forms, Boss Bandit, Goblin Machine, Ronin, Minion Giant, Goblin Curse, Vines/Void)
  plus **hero Tombstone + Tomb Queen, hero Dark Prince with its ability, and hero Barbarian Barrel**.
  No black frames this time. Expected pairing offset 67 s.
- **Parallel batch** (`scripts/run_all.py`): 13 videos (both new recordings + 11 YouTube), 2 CPU workers
  (convert + clock/hand/elixir pass) and 2 GPU workers (unit detection + DINOv2 crops), ~10 GB peak on the
  16 GB M5. Ryley's layout: auto-calibration picked his pink name text as the red tower bar on his red
  arena, so its arena rows were set by hand from measured bars (opp 445 / own 1477 px).
- **Auto-cut candidates were tried and dropped:** taking "the unit born with the play" from the old
  detector's tracks mislabels units when plays land close together; locating the drop with the on-screen
  name label ("Goblin Machine lvl.11") helps, but yield was 5 of 12 and unverifiable. Review time is the
  scarce resource, so **Cole gets click pages instead** (the method that gave 102 of 104 usable slices):
  `data/reviews/click_new.html`, `click_evo.html`, `click_hero.html`, frames 1.5 s and 3 s after each
  labelled POV play (8 plays per new card, 6 per evo/hero form), built by `scripts/build_review_queue.py`.
- **Cole asked to start everything at once, sized for the M5 / 16 GB**: 2 CPU + 2 GPU workers
  (above that, memory thrashes — the batch-16 training run showed it).
- **Batch results (19:59–23:07, 0 errors):** Cole 6:11 PM recording 10 games / 237 POV plays; friend 6:10 PM
  16 games / 334 plays; OYASSUU 2.6 Hog 7 games / 342; Ryley Log Bait 6 / 308; Ryley hero E-Wiz 4 / 106; plus
  Ryley's other five videos. Cole's recording was 240 fps, so its conversion alone took ~20 min.
- **Three OYASSUU videos found 0 games.** His HUD sits ~40 px lower in those than in the video his layout
  was calibrated on (a different recording setup), so the clock was never read. Each got its own
  auto-derived layout (`oyassuu_<video>`; clock read on 5/5 test frames), and `scripts/rerun_zero_match.py`
  now reruns *any* video that produces 0 games with a per-video layout — lesson: calibrate per video,
  not per creator. The reruns started 23:07; pairing (Cole 6:11:20 − friend 6:10:13 = 67 s) and the
  click pages build automatically when they finish.
- **Reruns done (00:30, 0 errors)** — all three OYASSUU videos found their games with per-video layouts.
- **Pairing, second session:** 9 of Cole's 10 games paired with his friend's at 65–66 s (expected 67), →
  **532 exact opponent labels in total** across both sessions, no review: Vines 38, Mirror 37, Goblinstein 37,
  Tombstone 37, Ronin 35, Goblin Curse 31, Boss Bandit 30, Void 30, Goblin Machine 28, Spirit Empress 27,
  Minion Giant 27, Suspicious Bush 25, Berserker 20, Dark Prince 18, abilities 32, and others.
- **POV plays now available for the new cards** (high confidence, dual-POV videos): every one ≥12 —
  Goblinstein 40, Vines 35, Void 32, Ronin 26, Goblin Curse 21, Boss Bandit 17, Goblin Demolisher 16,
  Berserker 15, Rune Giant 15, Minion Giant 14, Suspicious Bush 14, Goblin Machine 13, Spirit Empress 12.
  Cole's targeted session fixed every short card.
- **Hero forms found:** Berserker 36, Tombstone 33, Knight 30, Ice Golem 29, Goblins 28, Dark Prince 14,
  Barbarian Barrel 6, Valkyrie 5, Ice Wizard 4, Mega Minion 2. **Evo forms:** Cannon 41, Firecracker 33,
  Dart Goblin 24, Goblin Barrel 22, Ice Spirit 16, Tesla 15, Skeleton Barrel 12, Bats 10, Royal Hogs 9,
  Skeletons 8, Goblin Drill 5; only 1–2 plays of Baby Dragon, Battle Ram, Giant Snowball, Knight, Royal
  Ghost, Royal Giant, Witch.
- **Hero E-Wiz wasn't found**: the hand reader identifies cards by their API art, and it isn't in the API yet.
  It'll need its art (or a click-labelled example) once released.
- Review pages built: `click_new.html` (208 frames), `click_evo.html` (146), `click_hero.html` (106).

## 2026-09-30 — Click round on new / hero / evo pages
- Cole clicked **439 units**: new cards 208 (on 195 frames), hero forms 102, evo forms 129. His note: "I think
  you missed a few" — the pages had no usable frames for **Spirit Empress ground, Tomb Queen and dismounted
  Dark Prince** (the 1.5 s / 3 s offsets after the play were too early for ability summons and the form change).
- Cole asked whether to click the POV unit or the opponent's: the POV one (the frame follows its play); the
  opponent's copy of the same card is fine too, since slices are team-agnostic. For hero cards: click the
  hero unit, plus separate summoned units (Tomb Queen, dismounted Dark Prince); not transparent effects.
- SAM cut **401 slices** (new 200, hero 87, evo 114) → three approval galleries. Bug fixed: slices from
  multi-card pages were labelled with the page name ("hero") instead of each click's card.
- Cole's gallery pass on the new-card slices: **74 of 200 kept (37%)**. Weak ones: Rune Giant 2/16,
  Berserker 3/12, Goblin Curse 3/13, Void 3/16, Vines 0/12. Cole asked whether spells with troops inside
  count: no — a troop baked into a spell slice teaches the wrong thing. **Vines always wraps a troop**, so it
  has no clean slice at all. Decision: attached/transparent spell effects (Vines, Void, Goblin Curse,
  Earthquake, Rage, Freeze…) are learned from **boxes on real frames** (overlapping a troop's own box where
  they cover one), using the paired and POV labels, spot-checked by Cole — not from slices.
- Approved slices now: Goblinstein Doctor 47 / Monster 40, Minion Giant 34, Spirit Empress air 17 / ground 10,
  Rune Giant 15, Ronin 12, Suspicious Bush 10, Tomb Queen 7, Goblin Machine 7, Boss Bandit 5, Goblin
  Demolisher 5, Berserker 3, Goblin Curse 3, Void 3. Hero and evo galleries not reviewed yet.
- Approved slices are now committed (`data/slices/approved/`, 9 MB); candidates stay out of git.
- Cole's hero and evo gallery passes ("some were kinda brutal"): **hero 31 / 87 kept, evo 47 / 114**.
  Near-zero for Mega Minion hero 0/4, Valkyrie hero 1/9, Knight hero 2/9, Goblins hero 2/7, Berserker hero
  2/9, Cannon evo 2/9, Firecracker evo 2/10, Ice Spirit evo 1/8. Across all click rounds, SAM from a single
  click keeps ~35–40% — it spills into neighbours and effects on busy frames.
- **Conclusion:** slices alone won't reach 40 per entity. The main source for the paid run becomes
  **auto-labelled real frames** (boxes, from POV plays + paired opponent labels), with slices supplementing
  the generator; Cole spot-checks a sample instead of approving each one.

## 2026-09-30 — Decision: auto-labelled real frames become the main training data

**The problem.** Detector v2 needs ≥40 good examples of every new card, evo form and hero form. Our only
source so far was *slices* (SAM cut-outs pasted into generated arenas). After five click/gallery rounds,
SAM keeps only ~35–40% of cuts on real footage (spill-over into neighbours, effects, badges; transparent
spells like Vines can't be cut at all), and most entities are still at 0–15 approved slices. Separately,
v1 showed that training only on generated arenas leaves a gap to real footage (v1 alone: 22/52 right on the
opponent benchmark vs 31 for the old detector).

**The solution.** Train mostly on **real frames with auto-drawn boxes**, generated from labels we already
trust, with slices as a supplement:
- *What card and when* is already known exactly: POV plays (100% on 490 labelled plays) and 532 paired
  opponent labels from the dual-POV recordings (no review needed).
- *Where*: our own plays show the "<Card> lvl.N" name label at the drop spot; for paired games the partner's
  label gives the opponent's drop spot too (the arena is point-mirrored between the two views).
- *The box*: the unit tracked from that spot — old detector boxes plus SAM 2 tracking, anchored by Cole's
  clicks and approved slices. Spells get an effect box (overlapping a troop's box where they cover one).
- Cole spot-checks a random ~100 frames (the plan's existing check) instead of approving every example.

**Why this is the best option we have (and its limits).**
- It scales without Cole: every recorded or downloaded game adds labelled frames. More clicking doesn't scale.
- It trains on what the detector will actually see (real backgrounds, overlaps, effects) — the gap that hurt v1.
- Its labels come from the most reliable signals in the project (POV log, paired recordings), not from the
  detector we're trying to replace.
- It handles transparent / attached spells, which slices can't.
- Limits: boxes can be off when units overlap or the tracker drifts; that's why the spot-check exists, and
  why examples with low tracking confidence are dropped rather than kept. Opponent units in ladder/YouTube
  games (no partner recording) only have the old detector's position, so those are used more cautiously.
- Alternatives considered: more click rounds (accurate but ~35–40% yield and all Cole's time); hand-drawn
  boxes (most accurate, slowest); generated arenas only (already shown to fall short). Combining auto-labelled
  real frames + slices + a spot-check is the most accurate option per hour of Cole's time.

**What changes in the plan.** The go/no-go bar "≥40 approved slices per new entity" becomes **"≥40 labelled
instances per entity (real-frame boxes + slices combined), with the spot-check passing"**. Everything else
stays: two opponent benchmarks held out of training, a local 1-epoch test and a ~$1 cloud dry run before the
real run.
- **Auto-labeler built and run** (`bot77.autolabel`, `scripts/autolabel_all.py`): 1,145 trusted plays →
  **1,281 real training frames** in 35 min (0 errors) — 833 from POV plays, 448 from paired opponent plays;
  783 tracked unit boxes, 498 spell-radius boxes (Vines 146, Void 120, Goblin Curse 102). Testing found the
  opponent spots off by up to ~3 tiles: each layout's mirror line differs, now computed per layout from its
  tower HP bars (paired yield 1/12 → 11/20). Each frame also carries the old detector's boxes for every
  other unit (provisional).
- Spot-check page for Cole: `data/reviews/spotcheck_real_v1.html`, 100 frames round-robin across entities.
  Its accuracy decides whether these frames go into training as-is.

## 2026-09-30 — Spot-check of the auto-labels: 42 right / 36 box off / 22 wrong
- Cole judged 100 random auto-labelled frames. Cole: "some were pretty wrong"; the grey (old-detector)
  boxes "are labelled super wrong, but they do a good job at finding troops and items on the board."
- By method: **paired spells 0/9 right** (mirrored spell spots fail); POV spells 2 right / 4 off / 1 wrong;
  **POV units 34 / 29 / 10** (box fit is the main problem); paired units 6 / 3 / 2. With DINOv2 similarity to
  Cole's slices ≥ 0.6, 13 of 18 were right (≥ 0.5: wrong rate falls to ~10%).
- **Not good enough to train on as-is.** The failures are specific steps, not the approach (paired units held up).
- **Fixes (then rerun, then a 60-frame re-check by Cole before any training):**
  1. drop paired spell labels;
  2. tighten every unit box with SAM (box prompt from the tracked unit) — the old detector sizes boxes for
     the card it *thought* it saw;
  3. require similarity ≥ 0.5 to Cole's approved slices where the entity has any;
  4. re-classify the grey boxes by DINOv2 nearest neighbour against KataCR's 4,627 labelled slices + Cole's
     approved slices (a LanceDB table), keeping the old name only where the old detector was very sure.
- Fix 4, first attempt failed: DINOv2 nearest neighbour against KataCR's *transparent slices* agreed with even
  the old detector's confident labels only 5/14 (slices on a flat background look nothing like units on an
  arena). Rebuilt the index from **crops of KataCR's hand-labelled real frames** (5,051 crops, 142 classes,
  LanceDB table `real_crop_index`): **78% right on held-out labelled crops.** Not trusted alone — a grey box
  is renamed only when ≥5 of its 7 nearest crops agree; otherwise the old name stays.
- Test on 25 plays: SAM tightened 15 of 19 unit boxes (4 kept the original — mask implausible), kNN renamed
  3 of 40 grey units. Full rerun + a 60-frame re-check page (`spotcheck_real_v2.html`) running.
- Rerun done (0 errors): **837 frames** (from 1,281 — paired spells dropped, similarity filter). SAM tightened 366 unit boxes, 226 kept the original; kNN renamed 164 of 1,965 grey units.
  60-frame re-check page ready for Cole.

## 2026-09-30 — Spot-check round 2: wrong units 22% → 6%
- Of the 54 frames Cole judged (6 skipped): **32 right (59%), 19 box off (35%), 3 wrong (6%)**; grey boxes
  flagged on 1 frame (round 1: 8). The wrong-unit rate dropped from 22% to 6%; box fit is the remaining issue.
- Cole: "technically you're right here but you're missing a few hogs" — only one of evo Royal Hogs' four units
  was boxed. Found three causes and fixed them:
  1. **Multi-unit cards** (Royal Hogs, Bats, Skeletons, Goblins, Barbarians, …) now get a box on every unit
     born with the play, on shared frames.
  2. **Spells that drop units** (Goblin Barrel, Spirit Empress — which sits in the API's spell id range —
     Royal Delivery, Barbarian Barrel, Graveyard, Goblin Drill) were boxed as a spell radius; now as units.
  3. The old detector doesn't see flying evo Royal Hogs as separate units. A frame with only some of a group
     boxed would teach the rest as background, so **a group frame needs all but one of its units boxed, or
     it's dropped**. Evo Royal Hogs will mostly need clicked examples instead.
- Rerun with the group fixes: **732 frames**, 0 errors; 71 group frames dropped as incomplete. Side effect:
  groups the old detector can't split were dropped almost entirely — evo Goblin Barrel 44 → 0, hero Goblins
  31 → 1, evo Bats 11 → 4 (their earlier frames had only some units boxed, so they were wrong anyway).
- **Go/no-go check against the ≥40 bar (real frames + approved slices): only 9 of 40 entities pass**
  (Goblinstein Doctor/Monster, Minion Giant, Vines, Void, evo Cannon, Rune Giant, Suspicious Bush, Goblin Curse).
  Close: evo Firecracker 38, Goblin Machine 30, Spirit Empress air 29, evo Tesla 27, Ronin 26. Far: most hero/evo
  forms, Boss Bandit 6, Tomb Queen 7, evo Royal Hogs 6. **By the rule agreed on 2026-09-29, not ready for the paid run.**
- **Cole: "we need to pass the bar before we do a paid run. No point — I'm fine with getting more videos!"**
  The ≥40 bar stays a hard gate. Step 1: 6 frames per play instead of 3 (0.8 s apart); then a click page for
  the cards the auto-labeler can't handle; then more videos for whatever is still short.
- 6 frames per play: 732 → 929 frames, **11 of 40** pass (units rarely live 6 × 0.8 s near their spot).
- **Cole's clicks reused as real-frame labels** (`bot77.autolabel.label_clicks`): a slice needs a clean outline
  (35–40% passed), a training box only needs the unit's extent — so all 577 clicks were re-run through SAM as
  boxes: **484 click frames** (clicks on the same frame → one frame with every clicked unit boxed, so Goblinstein's
  Doctor and Monster don't each leave the other unlabelled). **18 of 40 now pass.**
- Step 2 click page `data/reviews/click_short.html` (322 frames) from not-yet-clicked plays of the 14 short
  entities that still have plays; group cards (Goblin Barrel, Goblins, Skeletons, Bats, Royal Hogs) are marked
  "click every unit". The other 8 (hero Valkyrie, hero Ice Wizard, evo Knight / Battle Ram / Goblin Drill /
  Royal Giant / Baby Dragon, hero Barbarian Barrel) have no plays left → YouTube candidates proposed to Cole.
- Cole on creators: Orange Juice and SirTag zoom and cut — skip; Mohamed Light, Issei_CR and Nideksito are good.
  Corey's (CorPlugg) layout is a split with an enlarged hand bar and no "Next" card → dropped.
- Cole's step-2 clicks: **346 clicks on 212 frames** → 209 frames; group frames complete. **28 of 40 pass.**
  Short: Boss Bandit (needs 2), hero Barbarian Barrel 15, Spirit Empress ground 16, Tomb Queen 17, evo Ice
  Spirit 20, evo Baby Dragon 24, hero Valkyrie 29, hero Ice Wizard 29, evo Goblin Drill 32, evo Battle Ram 32,
  evo Knight 33, evo Royal Giant 37. Spirit Empress ground and Tomb Queen have no video lined up — best source
  is a short recording session by Cole.
- **Batch 3:** 24 videos approved by Cole (Ryley, Ian77, Rolex, Adrian P, Mohamed Light, Nideksito, Issei_CR).
  Chain `scripts/after_batch3.sh`: per-video layout choice (creator layout vs. per-video auto, whichever reads the
  clock), process 2 CPU + 2 GPU, rerun zero-match, re-run the auto-labeler and click labels, recount.
- **Batch 3, first run processed nothing:** YouTube started answering HTTP 403 partway through the downloads
  (rate limiting after ~30 videos), leaving 8 of the 10 batch-3 files as partial `.part` downloads; the layout
  step then crashed on one and stopped the whole batch. Fixes: only finished files are used, an unreadable
  video is reported and skipped instead of stopping the batch, and downloads retry slowly (resume, 20 s
  between videos, up to 3 rounds 5 min apart).
- **Cole recorded a POV session (9:36 PM, 25.6 min) with both remaining gaps — Tomb Queen and Spirit Empress
  on the ground** — added to batch 3 (`cole_2026-09-30`).
- Batch 3 restarted 22:27 with **19 videos** (18 YouTube + Cole's recording); all downloads finished on retry.
  **6 videos skipped** — no layout reads their clock: Ryley `0m7iCgw8sYc`, `H35F0vVMTHg` (evo Royal Hogs), Rolex
  `Pn15Nh1xWoM`, `XuM18Vfd-xc` (evo Goblin Drill), Adrian P `ZXPr1gwF9ys` (evo Royal Giant), Mohamed Light
  `xONsnIlyu1E` (hero Barb Barrel). To be calibrated by hand after the batch.
- **Batch 3 done (02:13, 0 errors; auto-labelling + click labels ~03:00):** 19 videos processed, none needed a
  rerun. **2,206 labelled instances; 29 of 41 entities pass** (from 28 of 40). Big gains: Minion Giant 193,
  Goblinstein 191/175, evo Goblin Barrel 81, hero Ice Golem 79, evo Skeletons 77, evo Bats 72, hero Tombstone 72,
  evo Royal Hogs 56, hero Barbarian Barrel 54.
- **Still short (12):** Boss Bandit 38, Spirit Empress ground 28, evo Knight 25, Tomb Queen 23, evo Royal Giant 21,
  evo Ice Spirit 20, evo Battle Ram 16, evo Baby Dragon 16, hero Valkyrie 12, hero Ice Wizard 11, evo Goblin Drill 8,
  and a new entity, evo Valkyrie 1 (seen once in a Ryley video).
- Findings: hero Valkyrie has 35 detected plays but only 12 labels (auto-labelling rejects most against the single
  approved slice); hero Ice Wizard is often read as normal Ice Wizard by the hand reader, so its click frames come
  from all Ice Wizard plays in the three hero-Ice-Wizard videos.
- **Click page `data/reviews/click_short2.html` (304 frames)** for all 12, Tomb Queen and Spirit Empress ground taken
  from Cole's 2026-09-30 recording first. Not enough plays for evo Ice Spirit (4), evo Baby Dragon (5) or evo
  Valkyrie (1) even if every frame is clicked — those need more footage, as do the 6 videos skipped at calibration.

## 2026-10-01 — Spirit Empress ground fix, hand-calibrating the 6 skipped videos

- **Cole's clicks on click_short2** (`data/ground_truth/reviews/clicks_short2_2026-10-01.json`): 159 frames.
  Most clicks: evo Royal Giant 25, hero Ice Wizard 24, evo Battle Ram 23, evo Knight 20. Tomb Queen got only 2;
  Cole clicked hero Tombstone (13) on those frames instead.
- **Problem (Cole):** "you always tell me to do spirit empress ground yet you only give me images of spirit empress
  air. If u gave me images of placing spirit empress with < 6 elixir, you would get the ground version."
  **Cause:** Spirit Empress costs 3 on the ground and 6 in the air. The play detector expects 6, so every ground
  play came out as "medium" confidence. Click pages and the auto-labeler only take "high" plays, which left
  only air plays.
  **Fix:** the form now comes from the measured elixir cost: Spirit Empress under 4.5 elixir is ground, and medium
  confidence is accepted for those plays (`autolabel.spirit_empress_form`). The two forms separate cleanly:
  ground plays measure 1.7–3.6, air plays 5.2–6.4.
  **Why this fix:** the cost is already measured for every play and needs no new model. Changing the play
  detector's confidence rules would put its 490/490 at risk for one card. This gave 13 ground plays that had
  never been used, all from Cole's recordings (2 + 3 from 2026-09-29, 8 from 2026-09-30).
- **The 6 skipped videos, calibrated by hand.** Each failed for a different reason:
  - **Ryley ×2** (1080×2304): the red opponent tower bar merges with the red overlay into one blob. The other
    anchors were found, so `ryley_portrait` was shifted by the measured offsets (top −7, arena −9, hand −18 px).
  - **Rolex ×2** (1342×1920) and **Mohamed Light** (1080p stream overlay): on wide screens the HUD bands don't
    share one x transform. The top HUD sticks to the screen edges, the arena shrinks (0.88 of the hand scale) and
    the hand bar is centred. Rolex's timer is also cut off at the top edge, which stopped auto-layout before it
    looked at anything else.
  - **Adrian P** (1080p edit): the arena is on the left and the hand bar is re-scaled and pasted bottom-right
    under the face cam.
  - **Method** (`scripts/manual_layouts.py`): each band (hand, top-left, top-right, arena) is a separate linear
    map of Alina's layout, fitted from template-matched chat / Next / elixir drop, the timer, and colour-blob tower
    HP bars. Adrian's ability buttons were boxed by hand.
  - **Check:** the clock reads on 6/6 sample frames for all six, and region overlays line up on hand slots, Next,
    elixir, tower HP and the ability button.
  - Queued as batch 4 at 08:50.
- **Cole's review of click_short3 (`clicks_short3_2026-10-01.json`):** only 18 usable frames (hero Valkyrie 5,
  evo Goblin Drill 5, Spirit Empress ground 4, evo Battle Ram 2, Tomb Queen 2). Problems Cole reported:
  - Spirit Empress ground frames were real ground plays this time, but she was often crowded by other units.
  - Every evo Ice Spirit frame was noisy (it's small, fast, and dies on contact).
  - **Goblin Drill timing (Cole):** it travels underground for 2–3 s after placement before surfacing, and the evo
    then dives every ~2 s and pops back up 2 more times. Frames at +1.5 s showed nothing. New offsets: +3, +4.5,
    +6.5, +8.5 s.
  - **Decision:** pro footage is mostly used up for the remaining entities. Cole will record footage aimed at them,
    with each card deployed alone in open space, so every frame has one clean target.
- **Cole's evo Valkyrie and evo Ice Spirit recording** (`data/raw/cole_2026-10-01.mov`, 29 min, about 6–7 games,
  2+ of each evo per game). Same iPhone Mirroring crop as 2026-09-30. Cole: the plays near the end walk alone for
  about 4 s. Click pages now put Cole's recordings first for every card, latest plays first
  (`DEDICATED` in `build_click_short.py`). Next from Cole: evo Goblin Drill, hero Valkyrie and evo Baby Dragon.
- **Coverage audit (Cole: "are you sure these are the only troops we need?").** We checked every form of all
  123 deck cards in the API snapshot (2026-09-24: 42 evos, 17 heroes) against the detector's classes.
  - **Normal forms:** all covered. The only mismatches were naming: KataCR uses singular names for multi-unit cards
    (archer, skeleton, minion), and splits Elixir Golem and Phoenix into stages.
  - **Evo and hero forms:** 27 have no class at all. The "41" counted only forms we had seen in footage or slices.
    - Evo (19): P.E.K.K.A, Witch, Skeleton Army, Musketeer, Wizard, Minion Horde, Princess, Lumberjack, Inferno
      Dragon, Elite Barbarians, Hunter, Executioner, Royal Ghost, Mega Knight, Goblin Giant, Electro Dragon,
      Furnace, Goblin Cage, Giant Snowball.
    - Hero (8): Giant, Balloon, Musketeer, Wizard, Mini P.E.K.K.A, Bowler, Mega Minion, Magic Archer.
  - **How often they appear:** across about 65 processed videos most have 0 plays. The exceptions are evo Giant
    Snowball (44, a spell), evo Royal Ghost (8), evo Hunter (7), evo Musketeer (5), evo Furnace (3), evo P.E.K.K.A
    (2), hero Mega Minion (2) and evo Witch (1). This is because the footage is chosen around current top-ladder
    decks.
  - Open question for Cole: which of these are released and visually distinct from the base card, and whether
    rare forms can fall back to the base class for v2.
- **YouTube batch 5 (`data/raw/youtube/queue4.txt`, 23 videos, approved creators only)**, aimed at the short
  entities:
  - evo Valkyrie: Ryley ×2
  - evo Goblin Drill: Rolex ×4, Mohamed Light Extra ×1
  - evo Ice Spirit: Ian77, Ryley, Issei_CR
  - evo Baby Dragon: Ryley ×2
  - hero Valkyrie: Mohamed Light, Ryley, Rolex
  - hero Ice Wizard: Ryley, Mohamed Light, Ian77, Rolex
  - Tomb Queen: Ryley, Ian77, Rolex
  - evo Princess: Ryley
  - The evo Valkyrie and evo Ice Spirit videos are about 2 years old, since current pros rarely play them.
  - `scripts/prepare_batch.py` (generalized from prepare_batch3) now also tries each creator's hand-fitted layout.
  - `scripts/after_batch5.sh` chains download → layouts → processing → auto-label → coverage → `click_short5.html`.
- **Decision (Cole): all 27 missing evo and hero forms get their own class.** Cole: all of them are released, and
  "isn't it better to know for sure than to guess? we can use both to be 100% certain". The detector names the
  form, and evo cycle tracking from the play log confirms it. Two independent signals beat one. The footage exists
  online even if it's a few years old.
- **Mining plan (Cole asked whether we should mine now or wait):** wait until the new footage is processed,
  then run one mining pass over all `unit_crops`.
  - **Why wait:** mining searches every embedded crop for neighbours of labelled examples. Each new class first
    needs seed labels (POV auto-labels and Cole's clicks), and the more videos are embedded, the more one pass
    finds.
  - **Why mine at all:** the auto-labeler only covers the POV player's own plays and dual-POV pairs, so units the
    *opponent* plays in pro footage go unlabelled. Mining is the only route to those, and they're about half of
    all units.
  - Last full mining run was before batch 3; footage has roughly doubled since.
  - **Order:** download → process → seed click pages → one mining pass → Cole approves candidates in the gallery
    (fast keep/drop) → recount.
- **Batch 6 (`data/raw/youtube/queue5.txt`, 59 videos)** for the 27 new forms: 2–3 videos per form, approved
  creators only. A helper agent searched the forms that weren't already covered.
  - Downloads start after queue4. Processing follows batch 5 (`scripts/after_batch6.sh`), then auto-label →
    coverage → `click_short6.html` (seed clicks). Mining waits for those seeds.
  - About 20–25 h of machine time for batches 5 and 6 together.
- **Class list made append-only (bug found while adding the 27 forms).** `class_list()` re-sorted the non-KataCR
  classes on every run, and label files store class *indices*. Adding any class would have shifted the indices
  of the ones after it and silently mislabelled existing frames. It hadn't happened yet, because nothing had been
  added since the first auto-label run.
  - Now the order in `config/classes_v2.json` is kept and new classes are appended. A check confirms the first
    190 are unchanged.
  - Also added: every evo and hero form in the card table, which is the 27 new classes (217 in total), and
    `FORM_CLASS` to map card slugs to KataCR's singular names (e.g. elite-barbarians-evolution →
    elite-barbarian-evolution).
  - Coverage now lists zero-count classes: **32 of 69 pass**.
- **Hero Dark Prince dismounted (Cole: "in my own footage we should have a lot of it, should be something that we
  mine for").** The ability-button reader logged 19 ability uses in `cole_2026-09-29_c` but attributed none of them
  to a card, so no frames after an ability were ever looked at. That's why dismounted had 0 labels while mounted
  hero Dark Prince passed with 47.
  - **Fix for now:** `scripts/build_click_abilities.py` matches each unattributed ability to the last hero played
    in the same match within 60 s. In that video that gives 13 Dark Prince abilities (the other 6 follow Barbarian
    Barrel plays). It builds `data/reviews/click_abilities.html` with 52 frames, at +0.5/1.5/3/5 s after each
    ability.
  - **Next:** Cole's clicks seed `dark-prince-hero-dismounted`, and the mining pass looks for more across all
    footage, including opponents in pro videos.
  - **To do:** move the ability → hero attribution into the event pipeline itself, so the event log (goal C)
    records which hero used each ability.
- **Viiper 1 (Cole's tip):** he plays only Hog Exe Nado with evo Valkyrie, evo Ice Spirit and hero Goblins, and
  has hours of it. That makes him the best source for evo Valkyrie, which barely appears elsewhere. We queued
  12 of his videos (`queue_viiper.txt`, about 5 h, the last 6 months) and download them now, in parallel.
  - Any that finish before batch 5's prep runs join batch 5; the rest go into batch 6.
  - Cole's ability clicks (`clicks_abilities_2026-10-01.json`): 12 dismounted and 6 mounted hero Dark Prince.
- **Prioritized (Cole): Viiper and Tier-1 videos first.** `data/raw/youtube/priority.txt` lists 37 videos: Viiper
  ×12, then the queued videos for evo Goblin Drill, Tomb Queen, evo Valkyrie, evo Ice Spirit, evo Royal Ghost, evo
  Giant Snowball, evo/hero Wizard, evo Skeleton Army and evo Minion Horde.
  - queue5 was reordered so the Tier-1 forms download first.
  - Batch 5 now waits for all 37 priority videos and processes only those, Viiper first; batch 6 takes the rest.
  - The point is to learn by tomorrow morning which Tier-1 cards still need Cole's recordings, instead of after
    every video is done.
- **Click page 4 ready (12:27): `data/reviews/click_short4.html`, 195 frames.** Coverage after Cole's
  2026-10-01 recording plus all clicks so far (888 click frames): **2,687 instances; 34 of 69 pass.**
  - **evo Valkyrie now passes (61)**, almost entirely from Cole's recording (59 auto-labelled frames, 15 evo
    plays). evo Battle Ram passes (41).
  - **evo Ice Spirit** stayed at 20: the recording has 18 evo plays, but the auto-labeler kept no frames from
    them. They're on page 4 instead (30 frames, Cole's recording first).
  - Gaps carried over from footage already processed: evo Giant Snowball 30, evo Royal Ghost 6, hero Mega Minion 3,
    evo Witch 2.
  - Hero Dark Prince dismounted is 11 from Cole's ability clicks.
  - Page 4 covers 12 entities: Spirit Empress ground, hero Ice Wizard, hero Valkyrie, evo Giant Snowball, Tomb Queen,
    evo Ice Spirit, evo Goblin Drill (new timing), evo Royal Ghost, evo Hunter, evo Musketeer, evo P.E.K.K.A,
    evo Furnace.
  - Downloads: queue4 and Viiper complete; queue5 6/59 (Tier-1 forms first). Batch 5 waits for those.
- **Cole's clicks on page 4 (`clicks_short4_2026-10-01.json`): 97 frames.** evo Ice Spirit 20, evo Royal Ghost 16,
  evo Hunter 14, evo Musketeer 12, evo Giant Snowball 9, Spirit Empress ground 7, hero Valkyrie 7, evo P.E.K.K.A 4,
  evo Furnace 4, hero Ice Wizard 2, Tomb Queen 1, evo Goblin Drill 1.
  - Cole: great shots of evo Musketeer, P.E.K.K.A, Hunter, Ice Spirit and Royal Ghost.
  - **Goblin Drill timing, corrected again (Cole):** at +3 s every drill was only halfway to the tower; it takes
    about 4 s to arrive. New offsets: +4, +5, +7, +9 s.
  - Clicks converted to labels now, rather than waiting for batch 5's chain, so the recount is ready sooner.
- **Batch 5 started at 13:15** (the priority downloads finished early). 32 of 37 videos, Viiper first.
  - **Skipped at layout:** 2 Viiper videos (GAR755yAL60, s6y6y3ZIYL4). The red opponent bar wasn't found, the same
    problem as Ryley's. His layout from KjQYOTotuYE reads both 6/6, so it's now his known layout and batch 6
    processes them first.
  - **Also skipped, and not fixed yet:** Rolex pVzgSx84S0s (evo Skeleton Army; an older capture where his fitted
    layout reads 0/6), Adrian P bGt-YP6VMM8 (evo Giant Snowball, already at 38) and Mohamed Light Extra
    V-yMHtXPHOc (evo Goblin Drill; a different overlay from his main channel). Hand-calibrate later if those
    entities stay short.
- **Recount after page-4 clicks: 2,781 instances; 36 of 69 pass.** Newly passing: evo Ice Spirit (40, from
  Cole's clicks) and Spirit Empress ground (43).
  - Close: hero Valkyrie 39, evo Giant Snowball 38, hero Ice Wizard 36.
  - Still well short: evo Royal Ghost 20, evo Goblin Drill 19, evo Hunter 14, evo Musketeer 12, Dark Prince
    dismounted 11, evo P.E.K.K.A 4, evo Furnace 4, plus the forms whose videos are still in batch 5 and 6.
- **Cole's hero Valkyrie / evo Baby Dragon / evo Goblin Drill recordings:** `cole_2026-10-01_b` (3 min),
  `_c` (20 min) and `_d` (20 min).
  - **Crop:** `_c` and `_d` are 2940×1912, not 3024×1964 (a different display scale). The phone window was found
    by brightness bounds at x 1044–1895, the same window scaled by 0.972, so the crop is 852:1848:1044:62. A frame
    check against the old crop confirmed it.
  - **Run beside batch 5, one job at a time** (`scripts/run_serial.py`). Memory had about 35% free: room for one
    more job, not a second 2+2 pool. This gets his recordings processed in about 2 h instead of after batch 5
    (~13 h).
  - Then auto-label, recount and `click_short5a.html`. All of Cole's recordings now sort first on click pages.
- **Cole's Spirit Empress ground / evo Wizard / hero Dark Prince dismounted recording** (`cole_2026-10-01_e`,
  22 min, about 5 games). Same 2940×1912 window as `_c`/`_d` (bounds x 1044–1895), so the same crop.
  - Chained after the b/c/d recordings and run one job at a time (`scripts/after_cole_1001e.sh`).
  - After processing, `build_click_abilities.py` (now with a video filter) builds a Dark Prince ability page for
    this recording only, `click_abilities2.html`, alongside `click_short5b.html`.
- **Click page 5a ready (16:02): `data/reviews/click_short5a.html`, 136 frames.** Recount: **37 of 69 pass.** hero
  Valkyrie passes (48). Still short: evo Baby Dragon 33, evo Goblin Drill 19, Tomb Queen 28, hero Ice Wizard 36,
  evo Giant Snowball 38. Page 5a also has 56 evo Executioner frames from Viiper; those are optional, because
  batch 5's auto-label will reach them anyway.
- **Why Cole's recordings yielded so little:** only 12 frames from 3 recordings with 32 hero Valkyrie plays.
  - **Diagnosis:** we added drop counters to the auto-labeler. Of 41 plays, **37 were dropped for low DINOv2
    similarity** to the entity's approved slices (hero Valkyrie and evo Goblin Drill have one slice each), and 14
    had no track near the drop spot.
  - **Tried:** also anchoring on Cole's clicked real-frame crops (up to 40 per entity). Yield went from 12 to 66
    frames, but a 12-frame sample showed about half were wrong (boxes on empty arena or the king tower). Crops from
    the same arena share a background, which lifts every crop's similarity.
  - **Decision:** click anchors are opt-in. Their output goes to Cole as **candidates** (`scripts/candidates.py`:
    build → `data/reviews/candidates.html` ✓/✗ page → import only the ✓ frames as `manifest_reviewed.json`). The
    batch chains keep the stricter slice-only filter.
  - **Why this fix:** it keeps the higher yield while every accepted frame stays human-verified. Reviewing a
    frame takes about a second, where clicking one takes several.
  - Coverage now de-duplicates frames that appear in two manifests.
- **Candidates ready (17:19): `data/reviews/candidates.html`, 234 frames, ✓/✗.** 267 plays of short entities across
  all processed footage. evo Executioner 117 (Viiper), evo Goblin Drill 32, evo Giant Snowball 32, evo Baby
  Dragon 25, evo Hunter 10, evo Furnace 6, evo Witch 4, hero Ice Wizard 3, evo Musketeer 3, hero Mega Minion 2.
- **Recording `_e` done (17:45).** Spirit Empress ground passes (43). evo Wizard has 7 from auto-label, with 7 evo
  plays found.
  - The ability reader found only 3 hero Dark Prince abilities in 21 hero DP plays, so `click_abilities2.html`
    has just 12 frames. Either it missed some, or the ability wasn't used every time; ask Cole.
  - `click_short5b.html` (156 frames) is built from the same unclicked plays as 5a plus evo Royal Ghost and evo
    Wizard. It replaces 5a, which Cole hadn't started.
- **Cole reviewing candidates:**
  - **Drag preview** (a card "being hovered, not yet placed"): ✗ wrong, because it isn't on the board.
  - **Cause:** tracks could start up to 0.3 s before the play time, so the first picked frame could be the
    preview. **Fix:** frames earlier than play + 0.3 s are skipped (`MIN_AFTER_PLAY_S`). This applies to every
    auto-label run from now on, including the batch 5/6 chains.
  - **Evo Giant Snowball:** the candidates boxed the rolled trail and area. The right box is tight on the white
    ball, so a box with the ball plus the trail is "~ box off". To do: box it as a unit, not a spell area.
  - **Deploy marker** with no unit visible: ✗.
- **Cole's candidate review (`candidates_2026-10-01.json`): 42 right / 98 box off / 94 wrong out of 234 (18%
  right).** The 42 are imported (`manifest_reviewed.json`), and the recount is 2,834 instances, still 36 of 69.
  - **By entity:** evo Executioner 35/39/43 (right/off/wrong); evo Giant Snowball 0/30/2 (all trail-sized boxes,
    as Cole flagged); evo Goblin Drill 1/8/23; evo Baby Dragon 1/11/13; evo Musketeer 3/0/0.
  - **Conclusion:** click-anchored auto-labels are low precision. Keep them review-only, and don't extend them.
  - **Salvage:** "box off" means the right unit is in the frame, so those 98 become a click page
    (`scripts/build_click_off.py` → `data/reviews/click_boxoff.html`). Cole clicks the unit and SAM makes the box.
- **evo Giant Snowball (Cole):** "a second or two" from placement until it lands and rolls, depending on the
  distance from the king tower. If the ball isn't visible yet, skip the frame.
  - Click-page offsets are now dense across that window: +1.5, 2.25, 3, 3.75 s.
  - **Why the boxes were all off:** the auto-labeler treated it as a plain spell and drew the effect radius. It's
    now in `ROLLING`, so it takes the unit/track path and boxes the ball, or drops the play if no track is near
    the spot.
- **Cole's clicks on pages 5b, abilities2 and box-off:**
  - `clicks_short5b`: 93 clicks (evo Executioner 53, evo Baby Dragon 13, evo Royal Ghost 8, evo Wizard 8, evo
    Goblin Drill 7, other 4).
  - `clicks_abilities2`: 8 dismounted Dark Prince.
  - `clicks_boxoff`: 48 clicks (evo Executioner 24, evo Baby Dragon 8, evo Goblin Drill 6, other 10). None of the
    30 Snowball "box off" frames got a click; the ball wasn't visible in them, which matches the area-box finding.
- **Download resolution (Cole: "isn't 1440p like overkill?"): yes, so the cap is now 1080p** (`scripts/download.sh`).
  - The detector runs at imgsz 896, so it only ever sees about 900 px of arena. Layouts are built for 1080-wide
    captures, and 1440p made decoding about 1.8× slower for nothing.
  - 720p would shrink the arena (about 690 px) below the detector's input and blur tiny units (evo Ice Spirit,
    Skeletons) and the small text we read (card name popup, elixir digits, clock).
  - Compression matters more than pixel count.
  - Everything already downloaded stays as it is: re-encoding would cost about as much time as it saves.
