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
- **Batch 5 error (18:47):** Ian77 1t--8C08vFQ (evo Skeleton Army) crashed in play detection. A "match" with no
  readable frames (a menu or an edit cut) gave an empty elixir series, and `np.pad` failed on it. `detect_plays`
  now returns no plays for an empty match. The video was never added to the `videos` table, so batch 6's prep
  picks it up again. Tests pass.
- **Recount after clicks 5b / abilities2 / box-off: 2,975 instances; 39 of 69 pass.** Newly passing: evo
  Executioner, evo Baby Dragon and evo Giant Snowball.
  - Close: evo Battle Ram 39, hero Ice Wizard 39, evo Royal Ghost 32, evo Goblin Drill 29, Tomb Queen 27.
  - Dark Prince dismounted is at 19.
  - 17 forms are still at 0. Their videos are mostly in batch 6, and batch 5's auto-label hasn't run yet.
- **Quality audit before the paid run (Cole: "I want to make this good. If anything is not gonna perform, tell
  me").**
  - **Finding:** the real frames hold 3,001 main boxes (the play's unit: auto-labelled and spot-checked, or clicked)
    and **68,955 grey boxes** (every other unit, from the v1 detector plus kNN relabel). Grey boxes are about 96%
    of real-frame labels and their accuracy was never measured. Spot-check 2 only flagged whole frames with
    clearly wrong grey boxes (1 of 60). Training on them as-is makes v2 copy v1's mistakes, worst on the new cards
    v1 never knew.
  - **Measurement plan, spot-check 3** (`scripts/spotcheck3.py`, built after batch 5 by
    `after_batch5_review.sh`):
    - A: 100 auto-labels since spot-check 2
    - B: 50 click labels
    - C: 150 grey boxes, one per item, stratified over classes, cropped with context
  - **Decision rules once measured:**
    - grey boxes mostly right → keep them
    - wrong on certain classes → drop those classes' grey boxes and mask the regions
    - wrong broadly → keep only boxes the detector was confident about and the kNN relabel agreed with, and mask
      the rest
    - auto "box off" ≥ 30% → SAM-tighten the boxes from the box centre
    - borderline entities count only ✓-quality instances toward the 40 bar
- **Batch 5 done (22:08; 31/32, and the failed Skeleton Army video re-ran fine in batch 6). Its chain finished
  at 23:00.**
  - Auto-label added about 800 frames: **3,860 instances, 39 of 72 pass**. Three entities joined the list: evo
    Archers 11, evo Mortar 3 and evo Zap 2. These are KataCR classes that first got labels of ours in this batch.
    KataCR's own real frames also cover them, but the bar counts only our labels.
  - **Close:** hero Ice Wizard 39, evo Battle Ram 39, evo Royal Ghost 32, evo Goblin Drill 30.
  - **Further off:** Tomb Queen 27, hero Wizard 19, Dark Prince dismounted 19, evo Hunter 17, evo Musketeer 15,
    evo Wizard 15.
  - **Click page 5** (`data/reviews/click_short5.html`, 210 frames): Tomb Queen 22, evo Goblin Drill 18, hero
    Wizard 36, evo Archers 24, evo Zap 60, evo Minion Horde 28, and others. evo Zap and evo Archers are lower
    priority, because KataCR's data already covers them.
  - **Spot-check 3 ready (23:01):** `spotcheck3_auto.html` (100), `spotcheck3_clicks.html` (50),
    `spotcheck3_grey.html` (150).
  - **Batch 6 started 23:01:** 52 videos, including both Viiper retries and the Skeleton Army video.
- **Batch 6 error (05:25):** embedding for 1t--8C08vFQ failed with HTTP 504. `torch.hub.load` checks GitHub every
  time, even with DINOv2 already cached. `Embedder` now loads from the local hub cache. Detection had finished, so
  only `build_crops` was re-run for that video.
- **Crop classifier on our own labels (Cole: "are we currently not using our own labels for the embedding? I
  thought that was the point").**
  - **Before this:** our labels were only *references* for raw DINOv2 cosine similarity. Approved slices anchored
    the auto-label filter, and grey boxes were relabelled by a kNN vote over **KataCR's 5,051 crops only**, which
    contain none of the new cards. That second point is likely part of the grey-box problem.
  - **Built** `scripts/crop_classifier.py`: a 2-layer head on DINOv2 embeddings, trained on KataCR's crops plus
    our 1,324 verified crops (clicks and approved candidates) with mirrored copies. Our crops are split by
    video so near-duplicate frames don't leak.
  - **Results:** on held-out videos the head scored 50–61% versus 37–59% for kNN; the split is small, so the
    numbers are noisy. On KataCR it scored 90% versus 83%. When it's ≥95% confident it's right 83% of the time,
    but only on 44% of crops. Main confusions: Goblinstein doctor vs small humanoids, evo Musketeer vs evo
    Executioner, Spirit Empress air vs ground.
  - **Fair test against Cole's 234 candidate verdicts**, with all candidate frames held out of training. Keeping
    p(entity) ≥ 0.3 leaves 42 frames: right 18% → 60%, wrong 40% → 14%. It drops 17 of the 42 approved frames.
  - **Decision:** use it to **filter and rank review pages** (`bot77/cropclf.py`, threshold 0.3; entities it
    can't score are kept). It never accepts labels on its own. It goes into the mining pass and future candidate
    pages. Whether it replaces the grey-box kNN relabel waits for spot-check 3C's baseline.
  - **To improve it:** add spot-check-3A-verified auto-labels as training data (about 3× more crops), and add
    clicks for the classes it confuses.
- **Batch 6 chain done (09:23): 4,858 instances, 46 of 74 pass.** Click page 6 first crashed on a frame past the end
  of a video (late Tomb Queen offsets). Click pages now skip unreadable frames. `click_short6.html` was rebuilt with
  429 frames; it supersedes page 5.
- **Bug found: hero plays logged as the normal form.** hero Giant, Musketeer and Mini P.E.K.K.A stayed at 0 despite
  dedicated videos.
  - In those videos each hero card's real play is logged as the **normal** form (high confidence, elixir drop
    matched), plus a separate "hero" event with no elixir drop ("seen in the hand only: replaced by the previous
    next card, no elixir drop matched").
  - Counts: hero Giant 26 hero events vs 47 normal Giant plays; hero Musketeer 29 vs 19; hero Mini P.E.K.K.A 42
    vs 22. hero Magic Archer is mostly read as normal outright.
  - **Effect:** the auto-labeler (high confidence only) labels real hero plays as normal-form units and drops
    the hero events. This also corrupts the event log for goal C. It is probably the long-standing hero Ice
    Wizard shortfall too.
  - **Proposed fix, pending Cole's confirmation that a hero card is always in hero form once a player has it:**
    within a match, if any hero event of card X appears for a player, all of that player's plays of X in the match
    are hero.
- **Correction, with a bigger finding behind it.** The "hero logged as normal" diagnosis above was wrong. In hero
  Giant / Musketeer / Mini P.E.K.K.A videos the hero form *is* read; what fails is confirming the plays. **20 of 129
  processed videos have almost no high-confidence plays**, so the auto-labeler (high only) got nothing from them,
  and they are exactly the videos for hero Giant, Musketeer, Mini P.E.K.K.A, Mega Minion and more.
  - **Ryley ×3 (FRQBRMzwWXE, 9LTuX9QS2gk, Nr04--Oq2Rk) and Mohamed Light uXfDYoNhu5g:** elixir reads 0 all
    game. They got the hand-fitted layout of a *different* video of the same creator (`auto_0m7iCgw8sYc`,
    `auto_xONsnIlyu1E`) through the per-creator KNOWN list. Batch prep only checked that the clock reads.
  - **Ian77 ×10 (`ian_split`):** elixir reads fine, but every play is low confidence ("slot not refilled with
    the previous next card", "next card did not advance"), so the hand/Next region is misaligned. His overlay
    differs between older and newer videos; the other 8 `ian_split` videos are fine.
  - **Plus 6 marginal videos** (16–29% high), with the same causes suspected.
  - **Lesson:** a layout must be validated on clock, elixir *and* hand, not clock alone. It's the same mistake as
    Oyassuu's shifting HUD (decisions: calibrate per video).
  - **The hero rule** (Cole confirmed: once owned, a card is always played as the hero) is kept in
    `collect_items`. It's correct and harmless, but it wasn't the cause here.
- **Layout fix (`scripts/fix_layouts.py`, `bot77/layoutcheck.py`).** The check scores clock, elixir (a real
  value, not a consistent 0), hand slots and Next card, and the total is their product. Working layouts score
  0.58–0.96, broken ones 0.00–0.18.
  - **Fixed, 13 videos.**
    - **Ian77 ×11** with two refit layouts `ian_split_v2a/b`. In his older overlay the hand panel is 16 px left,
      27–32 px up and 2–4% larger, so the Next card never read. The layouts are anchored on the "Next:" label and
      score 0.58–0.96 (Next 0% → 75–100%).
    - **Ryley FRQBRMzwWXE** (its own auto layout, 0.79) and **Nr04--Oq2Rk** (`ryley_portrait`, 0.44).
  - **Not fixed yet:**
    - Ryley 9LTuX9QS2gk (elixir 27%) and Mohamed uXfDYoNhu5g (elixir 0%): the elixir region needs refitting.
    - 4 two-year-old videos (cKuf3-EGV0M, ElIl0ij-Z6o, SBBy2m-flPU, ucNA_I_hBc0) where the hand reads 2–40%.
      Likely the old card-frame art; they'd need old-art templates and are probably not worth it.
  - The re-run of the 13 started at 11:44 (`scripts/after_fix.sh`). None of them had auto-labels to drop, because
    nothing was ever confirmed in them.
  - **Batch prep now picks layouts with the full check** (minimum 0.4) and skips videos where nothing reads
    everything, instead of using clock reads alone.
- **Cole on 3C:** a hero Balloon grey-boxed as "balloon" is ✗. v1 never learned hero forms, so grey boxes on them
  default to the base card. That's expected and is exactly what 3C should quantify.
- **Cole on 3A: evo Archers frame with a second yellow box on a Cannon → ✗** (any wrong main box makes the frame ✗).
  - For multi-unit cards the labeler boxes the N tracks nearest the drop that started around the play time. A
    Cannon standing beside the drop was picked as "archer #2", while both real archers sat in one box.
  - **Fix:** a track the detector calls a building is never taken as a unit of a non-building card. This applies
    to all auto-label runs from now on, including the fix re-run.
  - **To do:** a group whose units overlap (two archers in one box) should be split or dropped. Check how often
    that happens in 3A.
- **Spot-check 3 results (Cole, 2026-10-02):**

  | Page | Right | Box off | Wrong | Notes |
  |---|---|---|---|---|
  | 3A auto-labels (100) | 48% | 25% | 27% | worse than spot-check 2 (59/35/6): mostly the new forms (evo Archers groups, evo Royal Ghost, evo Executioner, evo Snowball) |
  | 3B click labels (50) | 66% | 34% | 0% | never the wrong unit; the SAM box from the click is often loose |
  | 3C grey boxes (150) | 59% | 2% | 39% | localization is good, classification is bad; all grey labels are KataCR classes, so new cards/forms in the background get old names |

  - **Test: classifier filter on Cole's verdicts** (`bot77/cropclf.py`; no leakage, because none of these frames
    are in its training data):
    - grey, keep only when label = classifier top-1: **59% → 88% right**, wrong 39% → 10%, keeps 32%
    - auto, keep only when p(label) ≥ 0.3: **48% → 86% right**, wrong 27% → 0%, keeps 36%
  - **Decisions:**
    1. Grey boxes stay only where the classifier agrees with the label. All other grey boxes are **masked**
       (filled) in the image, so the model isn't taught that units are background.
    2. Auto-labels go through the same filter. Frames that fail are dropped.
    3. Click boxes: investigate the SAM step (likely too big) and tighten it, then re-measure on a small sample.
    4. Add the 3A ✓ crops to the classifier's training data and retrain.
    5. Recount coverage with filtered labels only. Some entities will fall back under 40, and that's the honest
       number.
  - **Page 6 clicks:** 480, including evo Minion Horde 140, evo Archers 43, evo Electro Dragon 36 and evo Mega
    Knight 32.
- **Click boxes:** when a detector box contains Cole's click, label_clicks now uses that box (tightest one,
  non-HUD classes). It takes only the box from the detector, never the class. SAM from the click is the
  fallback, and a SAM box that is mostly the deploy stopwatch drops the click.
  - Test on 25 clicks: 9 detector boxes, 13 SAM, 2 dropped on the stopwatch. Some SAM boxes still sit on the
    stopwatch, so the clicks need re-measuring after the rebuild.
- **Classifier retrained** with spot-check ✓ crops (1,460 of ours across 112 videos): held-out ours 54% (kNN 49%),
  KataCR 91%; when ≥95% confident, 88% right on 50% of crops.
- **Cleaned dataset** `scripts/clean_dataset.py` → `data/external/real_v2`, with `real_v1` untouched:
  - verified labels kept
  - auto frames dropped if any main box has p(label) < 0.3
  - grey boxes kept only when they match the classifier's top-1; otherwise masked (median fill), or just dropped
    if they overlap a kept box
  - **Concern to check visually:** early frames mask about 2/3 of grey boxes, so frames get many filled patches.
  - **Cleaner, first run stopped:** it masked every tower and HP bar. The classifier never learned HUD classes,
    because the crop index skips them, so it could never agree with them. Now `KEEP_OLD` classes and any class the
    classifier doesn't know are kept unjudged.
  - **Second run:** wrong grey boxes larger than 3 tiles are dropped instead of masked. They're v1 misdetections,
    and masking left big square patches.
  - **Known gap, not caused by cleaning:** units v1 never detected (e.g. a deploying Mighty Miner) have no box, so
    they read as background. v2's own detections on real frames are the fix later (review its new boxes and add
    them).
- **Honest recount, first cleaned set: 2,556 instances, 28 of 73 pass (was 46).** Two flaws in the filter showed
  up in the per-entity drops:
  1. **Spells** (Vines 176 → 22 frames, Void 75 → 21, Goblin Curse 101 → 26): their labels are effect areas,
     which a unit classifier can't judge. Spell-radius labels are now exempt, except evo Giant Snowball area boxes,
     which are dropped (known wrong).
  2. **Rare forms** (hero Knight 180 → 12, hero Wizard 40 → 9): the classifier calls them their base card. The form
     comes from the play log, which is trusted, so the filter now scores the **card family**: the sum over the
     card's normal, evo and hero forms. On the 3A verdicts, with a model that never saw those frames: 80% right,
     5% wrong, keeps 35/48 of Cole's ✓ (per-label at 0.3: 69/9, 38/48).
  - Cleaner re-run started (CLEAN2).
- **Second cleaned set (CLEAN2, 14:09): 2,606 instances, 30 of 73 pass.** Spells are restored (Vines 176, Goblin
  Curse 104, Void 78) and hero Knight passes (40). This is the honest coverage number from now on; uncleaned counts
  are no longer used for the bar.
- **Layout-fix re-run:** 13 videos processed by 14:31 with 0 errors; auto-label is running.
  `after_fix_clean.sh` then rebuilds real_v2, recounts cleaned, and builds `click_short7.html` from the cleaned
  shortfall (CLEAN3).
- **Layout fix done (FIX_DONE 15:21, CLEAN3_DONE 15:39).** 12 of 13 re-run videos now confirm 62–81% of plays (was
  0–5%). Nr04--Oq2Rk is still 10%. **Cleaned coverage: 3,008 instances, 31 of 74 pass.**
  - `click_short7.html`: 860 frames across 28 entities, built from the cleaned shortfall.
- **hero Giant / Mini P.E.K.K.A / Musketeer still at 0–1 after the fix.** Auto-label did produce them (49 / 68 / 35
  frames in real_v1), but the cleaner dropped them all. The classifier has no crops of those forms, and they look
  different enough from the base card that even the card-family score is low. The classifier was judging forms it
  had no basis for.
  - **477 auto frames** are for forms the classifier never saw: hero Ice Golem 90, hero Mini P.E.K.K.A 68, evo
    Elite Barbarians 62, hero Giant 49, evo Furnace 40, hero Balloon 36, hero Musketeer 35, evo Goblin Cage 30,
    hero Bowler 23, and smaller ones.
  - **Decision:** they go to Cole on a ✓/✗ page (`data/reviews/formreview.html`). ✓ frames count as verified in
    the cleaner (via `formreview_*.json`). The ✓ crops then train the classifier, so next time it can judge these
    forms itself.
- **Cole's form review (`formreview_2026-10-02.json`): 477 frames, 111 right / 197 box off / 169 wrong (23% right).**
  This confirms that forms the classifier never saw were also the auto-labeler's weakest. The 111 ✓ frames now
  count as verified in the cleaner.
  - Best: evo Musketeer 11/15 right, hero Balloon 16/36, evo Goblin Cage 14/30, hero Giant 14/49.
  - Worst: hero Mini P.E.K.K.A 5/68, evo Elite Barbarians 3/62.
  - The 197 box-off frames became `click_boxoff2.html` (Cole clicks the unit, the detector or SAM boxes it).
- **Cole's clicks on page 7 (`clicks_short7_2026-10-02.json`): 677 clicks on 586 frames.** evo Elite Barbarians 85,
  evo Wall Breakers 61, evo Witch 53, hero Giant 51, hero Musketeer 39, evo Skeletons 35, evo Mortar 35, hero Ice
  Golem 33, ...
- **Box-off 2 (`clicks_boxoff2_2026-10-02.json`): 173 clicks.** evo Elite Barbarians 51, evo Furnace 28, hero Ice
  Golem 23, hero Mini P.E.K.K.A 17, hero Giant 12, ...
- All clicks are re-labelled with the new box rule, then re-cleaned and recounted (CLEAN5, then CLEAN6 with
  box-off 2). Spot-check 4 then re-measures 50 click boxes.
- **Cleaned recount after all clicks (CLEAN6, 2026-10-02 22:58): 3,718 instances, 47 of 74 pass** (was 32).
  - Within 5: evo Electro Dragon, evo Archers, Berserker, hero Musketeer (39 each), evo Furnace 38, evo Mortar 38,
    hero Berserker 37, evo Mega Knight 36, Boss Bandit 35.
  - Further: evo Zap 33, hero Wizard 32, evo Goblin Cage 32, hero Mega Minion 30, evo Royal Ghost 30, evo Bomber
    29, evo Goblin Drill 28, evo Wizard 27, Tomb Queen 27, evo Hunter 24, evo Princess 20, Dark Prince dismounted
    19, evo P.E.K.K.A 17, evo Goblin Giant 8, evo Lumberjack 5, evo Inferno Dragon 4, hero Magic Archer 0, evo
    Skeleton Army 0.
  - Spot-check 4 (50 click boxes after the box fix) is ready. `click_short8.html` was built from the cleaned
    shortfall.
- **Spot-check 4 (50 click boxes after the box fix): 72% right / 26% box off / 2% wrong** (before: 66/34/0). Only a
  modest improvement. Click labels almost never carry the wrong card, but about a quarter are loose or partial
  boxes. Telling detector boxes from SAM boxes after the fact didn't work, so the effect of the detector-box rule
  itself is unmeasured.
  - **Recommendation:** train v2 with them, then replace click boxes with v2's own box where it overlaps the click,
    and retrain. The alternatives were point+box SAM prompts or Cole redrawing boxes.
- **Cole's page 8 clicks:** 31 (hero Mega Minion 9, hero Wizard 6, evo Furnace 5, ...).
- **Classifier retrained** with every verified label (2,656 of ours across 186 videos plus 5,051 KataCR crops, 208
  classes): held-out ours **70%** (was 54%), KataCR 88%. At ≥0.9 confidence it's 92.5% right on 67% of crops.
- **Mining pass on every card** (Cole: "take a mining pass on every single card"): `scripts/mine_all.py` scores all
  1.92M embedded unit crops with the classifier and keeps up to 40 per non-KataCR entity at p ≥ 0.9.
  - It prefers the opponent side, at most one crop per match-second and 6 per video, and skips anything within
    3 s of an existing label for that entity.
  - Output: a ✓/✗ crop page (`data/reviews/mining.html`).
- **Mining pass results (run 2):** 2,170 crops, **99% opponent side** (2,156 enemy, 14 friendly), which is exactly
  what our labels lacked. Most entities reached the 40 cap. evo Lumberjack (1) and evo Inferno Dragon (9) are
  genuinely rare. hero Magic Archer and evo Skeleton Army are absent because the classifier has no examples of
  them; Cole is recording both.
  - Crashed once at the end of scoring: duplicate crop rows broke the heap sort. The cause: `build_crops`
    *appended* on re-run, so the video re-embedded after the 504 (and the 13 layout-fix re-runs) had doubled
    crops. 1t--8C08vFQ had 28,896 rows for 14,448 crops. `build_crops` now deletes a video's crops first, and
    mining skips duplicates. The doubled videos still need their crops rebuilt.
  - Run 3 adds every entity still under the bar that carries a KataCR name (evo Archers, Zap, Bomber, Mortar, Wall
    Breakers...). The page is sorted shortest entity first.
  - `scripts/mine_import.py` turns ✓ crops into full training frames with the auto-labeler's frame writer, and the
    cleaner treats them as verified.
- **Mining review (Cole, `mining_2026-10-05.json`, took him two days): 2,330 crops, 776 right / 475 off / 1,036
  wrong (33% right),** even though the classifier's held-out ≥0.9 confidence measured 92.5% right.
  - **Why:** its calibration didn't carry over to opponent-side crops. Nearly all its training crops were the
    POV player's own units.
  - **Lessons:** measure precision on a small sample before building a big page, and cap review pages at ~150
    items (Cole: "please try not to give me such massive lists").
  - Best: evo Cannon 39/40, evo Mortar 38/40, evo Witch 33, evo Archers 32, evo Goblin Cage 32, hero Balloon 31.
  - Zero right: Dark Prince dismounted, evo Zap, Goblin Curse, evo Goblin Giant, hero Musketeer, evo Royal Hogs.
  - The 776 ✓ are imported as training frames (`after_mining.sh`).
- **Failure modes Cole found while reviewing:** evo Goblin Giant ↔ Phoenix, and Vines ↔ Suspicious Bush. ("Lets make
  sure the actual model doesn't do this.")
  - **Cause, from looking at our own labels.** They're label contamination, not look-alikes.
    - evo Goblin Giant: in that deck Phoenix flies over the Giant, so of 17 labels several boxes sit on or
      include the Phoenix.
    - Vines: the 204 spell-radius labels are taken 0.3–2 s after the drop, while the "Vines" name popup is
      showing, so they teach "popup text over whatever's underneath" (towers, troops, often a Bush). Void and
      Goblin Curse probably have the same problem.
  - **Plan:**
    1. Fix the labels: drop Goblin Giant boxes overlapping a Phoenix; re-time spell labels to after the popup
       (asked Cole for the delays).
    2. Synthetic data: make sure these four have approved slices, since independent pasting breaks the pairing.
    3. Targeted gate before any paid run: hand-checked frames where the pairs co-occur, with a per-pair
       confusion report.
    4. Recordings: evo Goblin Giant without Phoenix, Vines on open ground.
- **Spell timing (Cole):** Vines hits ~0.1 s after placement (it misses a moving unit if placed on its edge). Void
  fully forms ~0.1 s after placement and its beam comes ~0.1 s later. Goblin Curse shows ~0.1 s after placement,
  and both behave like Poison.
  - So timing wasn't the problem. Every Vines frame shows Vines, wrapped around the unit it hit.
  - **The real cause:** that unit (often a Suspicious Bush, which v1 doesn't know) had a wrong grey label, so the
    cleaner masked or dropped it, leaving an unlabelled unit inside the Vines box.
  - **Rule:** in a spell-area frame, if any unit inside the spell box would lose its own label, the frame is
    dropped. Applies to Vines, Void, Goblin Curse and Poison alike.
- **Recount after importing the 776 mined frames (clean 7, 2026-10-05 16:22): 4,704 instances, 64 of 74 pass**
  (was 47).
  - The cleaner dropped 112 spell frames with an unlabelled unit inside the spell and 3 evo Goblin Giant/Phoenix
    overlaps. Vines, Void and Goblin Curse still pass.
  - **Remaining 10:** evo Tesla 34, evo Zap 29, evo Princess 26, evo Hunter 25, Dark Prince dismounted 19, evo Goblin
    Giant 9, evo Inferno Dragon 7, evo Lumberjack 5, evo Skeleton Army 0, hero Magic Archer 0. Mostly recordings;
    Tesla and Zap may close with a small click page.
- **Deck rule (Cole):** a deck holds 1 evo, 1 hero and 1 wildcard (a second evo or a second hero).
- **Session 1 recording** (`cole_2026-10-06`, 20.6 min): evo Skeleton Army, evo Lumberjack, hero Magic Archer. Cole
  isn't sure he reached 10 deploys each; we'll count from the play log. Standard crop. Chain:
  `after_cole_1006.sh`.
- **Session 1 result: 12 labels** (hero Magic Archer 8, evo Lumberjack 4). The hand reader logged nearly every play
  as the normal form, even though hero Magic Archer's golden card is visibly in hand.
  - **Fix: declared decks** (`config/recording_decks.json`, `declared_forms()` in autolabel). For Cole's dedicated
    recordings he declares the evo/hero cards. Every play of a declared hero is the hero. For evos with N cycles:
    N normal plays, then the evo, repeating, anchored on any evo play the reader did catch. Cole: since an update
    the first evo sometimes needs one extra play. Session 1's one caught evo Lumberjack was the standard 3rd play,
    so the extra play isn't consistent.
  - **The real constraint is evo math:** each card was played 2–3 times per match, and a 2-cycle evo charges every
    3rd play, so 6 matches gave only 2 evo Skeleton Army and 3 evo Lumberjack plays.
  - **New recording guidance:** one evo per session in a cheap cycle deck, played every time it's in hand, about 10
    short matches (~20 evo plays). Heroes are easy because every play counts.
  - `click_session1.html` (~60 frames): Cole clicks the inferred evo/hero plays.
  - **Correction (Cole):** he played each evo "at least once a game, sometimes twice", so the 3rd-play rule
    (from our card data's 2 cycles) undercounts. The card data's cycle counts may be outdated. The session 1 click
    page now covers **every** Skeleton Army / Lumberjack / Magic Archer play (2 frames each), and Cole picks evo/hero
    or normal per frame. His picks also give the real cycle pattern, which will fix `declared_forms`.
- **Hand reader bug found from Cole's timestamps (2026-10-07).** At 2:52 in session 1, evo Lumberjack and hero
  Magic Archer were plainly in his hand, but the reader returned nothing for both slots (score 0.22–0.27). Their
  plays never reached the event log, which is why session 1 looked "all normal".
  - **Cause:** evo and hero cards are drawn larger than the reader's calibration. The template scale sweep peaked at
    **0.88** (score 0.69–0.70) against the calibrated 0.80; normal cards are at 0.73.
  - **Fix:** evo/hero templates are now tried at 0.80, 0.84, 0.88 and 0.92, and the best score per card wins.
    Session 1 now reads evo Lumberjack 0.69 and hero Magic Archer 0.70 with clear margins. 70 tests pass.
  - **Likely impact everywhere:** evo/hero plays under-detected in all footage. That plausibly explains hero Ice
    Wizard stuck at 39, the hero Giant/Musketeer zeros, and the evo yields.
  - **Plan:** re-run the hand-reading (process) stage. Session 1 first, then Cole's recordings, then dedicated
    videos for short entities, then the rest. About 25 min per video.
  - Cole asked how to help: timestamps plus a screenshot of a miss are the most useful input. They pinpointed a
    bug the metrics never showed.
- **Second reader problem: outdated card art.** Charged evo Skeleton Army shows the **Skeleton General** art in hand,
  but the API's evo icon is still the old skulls picture, so it scored 0.23. Fix: **in-game art overrides**
  (`data/cards/art_overrides/<Card>__<form>.png`). These are art patches cut from a real frame at slot scale, at the
  box where evo art sits; the reader loads them as extra templates.
  - evo Skeleton Army now reads 0.93–0.98 on frames other than the source, and normal Skeleton Army still reads
    normal (0.91).
  - After the scale fix alone, session 1 had hero Magic Archer 20 plays (was 0) → 35 auto-labels, evo Lumberjack 4
    plays (was 1) → 8, evo Skeleton Army still 0 until this override. Session 1 is being re-run again.
  - **Other cards may have outdated art too.** A quick check is to compare each evo/hero icon with an in-game
    crop from Cole's recordings.
- **Session 1 after both reader fixes (16:37):** play log hero Magic Archer 20, evo Lumberjack 4, evo Skeleton Army
  6, matching Cole's "about one evo per game". Cleaned counts: hero Magic Archer 25, evo Lumberjack 7, evo Skeleton
  Army 2 (clicks only). Overall 64 of 74.
  - evo Skeleton Army got 0 auto-labels: Skeleton Army is a 15-unit group, and frames without every skeleton
    boxed are dropped.
  - **Decision (proposed to Cole):** the Skeleton General is labelled `skeleton-army-evolution`, and the plain
    skeletons are `skeleton`.
  - `click_session1_evo.html`: the evo plays, 3 frames each.
- **Session 1 evo clicks:** 22 (evo Lumberjack 12, evo Skeleton Army 10).
- **Re-read round 1** (`scripts/reprocess_hand_list.sh`, 2 videos at a time): 12 Cole/friend recordings plus 18
  dedicated hero/evo videos for short entities (evo Princess, Hunter, Goblin Giant, Inferno Dragon, Lumberjack,
  Skeleton Army; hero Magic Archer, Giant, Musketeer, Mini P.E.K.K.A).
  - Old auto-labels for those videos are dropped first; verified labels (clicks, reviewed, mined) are kept.
  - Then auto-label, re-label clicks, re-clean and recount once.
- **Incident during the round 1 launch:** the job limiter counted `jobs -r` inside `$(...)`, a subshell that sees no
  jobs, so all 30 videos started at once.
  - Killed within ~30 s. The events table's version history shows no writes after 15:15 (session 1's re-read), so
    nothing was lost. Each job writes only at the end of its video.
  - **Why it mattered:** each video rewrites the shared events table, so concurrent finishes can overwrite each
    other, besides exhausting memory.
  - Now `xargs -P 2` with `scripts/reprocess_one.sh`; restart verified at exactly 2 processes.
- **Round 1 narrowed to videos that play a still-short card** (Cole: "make sure this is only running for cards we
  need"): 18 of 30, by POV plays in the pre-fix event log (version 317).
  - Dropped: most of Cole's recent recordings, the friend recordings, and three hero Giant/Musketeer videos (those
    entities already pass).
  - **Caveat:** the old log was blind to evo/hero plays, so a video whose only short card is an unlogged evo
    could be missed.
  - The chain waits for the 2 in-progress re-reads, re-reads the remaining 16 (2 at a time), then labels all 18.
- **Early before/after check of the hand-reader fix (Cole: "check the diff after a few vids"),** events v317 vs now:
  - Ryley evo Princess video tGONo2rzFHw: **evo Princess 4 → 16 plays**, high-confidence plays 214 → 228.
  - Cole 2026-10-01_e: evo Royal Ghost 2 → 5, high-confidence 72 → 77.
  - Cole's Sept 28/29 recordings: identical. Their evo/hero cards (hero Dark Prince, evo Tesla, evo Firecracker)
    were already read, so the bug depends on the card/capture.
  - Only regression: evo Goblin Barrel 13 → 12 (noise, already passes). **Round 1 continues.**
- **Session A recording** (`cole_2026-10-07`, 28 min): evo Goblin Giant, evo Inferno Dragon, hero Dark Prince.
  Deck declared in `config/recording_decks.json`. Standard crop.
  - It converts now and is processed after re-read round 1 (one writer to the events table at a time).
  - Then auto-label, a Dark Prince ability page (`click_abilities3.html`), re-clean and recount (`after_cole_1007.sh`).
- **Re-read round 1 done; recount at 23:00.** The before/after across the 18 videos recovered: evo Princess 9 → 40
  plays, hero Magic Archer 1 → 46, evo Mega Knight 0 → 14, evo Inferno Dragon 1 → 13, evo Skeleton Army 1 → 11, evo
  Lumberjack 1 → 9, evo Hunter 1 → 6. High-confidence plays rose in nearly every video; nothing regressed beyond
  noise.
  - **Cleaned coverage: 64 of 75.** evo Princess 34, evo Tesla 34, evo Zap 29, evo Hunter 25, hero Magic Archer 24,
    Dark Prince dismounted 19, evo Lumberjack 19, evo Skeleton Army 11, evo Goblin Giant 9, evo Inferno Dragon 7.
    A stray `king-tower` main box also showed up (1); it needs a look.
  - **Labels grew much less than plays:** the cleaner rejected nearly all new auto-labels for these forms (hero
    Magic Archer 70 auto → 0 kept). The classifier knows hero Magic Archer only from Cole's recording, so pro-video
    crops look unfamiliar. That's caution about rare forms, not proof the labels are bad.
  - **Decision:** these go to Cole on `formreview2.html` (capped at 40 per entity). ✓ count as verified (cleaner
    `formreview_*.json`) and teach the classifier.
- **Form review 2 (Cole, `formreview_2026-10-08.json`): 126 frames, 29 right / 39 off / 58 wrong.** The cleaner's
  rejections were mostly correct: hero Magic Archer 2/40 right, evo Princess 2/16, evo Hunter 0/6, evo Inferno Dragon
  0/4. Exceptions: **evo Tesla 17/30 right**, evo Lumberjack 7/21.
  - **Conclusion:** for rare evo/hero forms, auto-labels from pro footage are mostly wrong even once the plays are
    found. Cole's clicks are the reliable source. The 29 ✓ count as verified (cleaner re-run), and the 39 box-off
    frames went to `click_boxoff3.html`.
- **Session A** (`cole_2026-10-07`, done 00:23): 8 matches, 185 plays, evo Goblin Giant 8 plays, evo Inferno Dragon 5,
  hero Dark Prince 24. Only 3 ability uses read, so the ability reader likely misses most. Coverage after: evo Goblin
  Giant 12, evo Inferno Dragon 7 (its auto-labels were all filtered), Dark Prince dismounted 19.
  - `click_sessionA.html`, 111 frames: evo plays at +1.5/3/4.5 s; every hero Dark Prince play at +4/7/10 s, so Cole
    clicks the dismounted Prince when he's off.
- **Cole on session A:** some +1.5 s frames show the card's "falling down" deploy animation and he couldn't tell
  what to click. Rule: click only when the unit is clearly recognisable; skip blurry drop-ins, since later frames of
  the same play show it standing. Default click-page offsets moved from +1.5/3/4.5/6 s to +2.5/4/5.5/7 s
  (per-card offsets like Goblin Drill and Snowball unchanged).
- **evo Zap merged into Zap for the detector (Cole, 2026-10-08).** Neither Cole nor I could tell evo Zap from Zap in a
  board frame, but "u can always tell the evo by the card in the hand".
  - The detector trains evo Zap boxes as `zap`. The form comes from the hand reader (POV) or opponent cycle counting
    (the detector sees each opponent Zap). This is the same "use both" design as for evo P.E.K.K.A.
  - `MERGE_FORMS` in the cleaner; only evo Zap so far, and other board-indistinguishable forms can join.
  - This removes evo Zap from the 40 bar.
  - **Reverted the same day (Cole):** evo Zap does look different. It's a truer blue and zaps twice; opponent
    cycles can be counted as a cross-check. evo Zap stays its own class (`MERGE_FORMS` is empty). Click frames for
    it should catch the blue colour and the second zap.
- **evo Zap timing (Cole):** the second zap hits about a full second after the first. Click offsets for evo Zap
  are +0.3, +1.0 and +1.3 s. `click_evozap.html`: up to 16 unclicked high-confidence evo Zap plays, at most 3 per
  video; Cole picks evo Zap or Zap per frame.
  - Only 1 unclicked high-confidence evo Zap play remained (55 evo Zap plays exist: 33 high, 21 medium). The page now
    includes medium-confidence plays (usually an elixir mismatch, and Cole verifies each): 20 plays, 3 frames each.
  - The 6 Zap videos not in re-read round 1 (9HqAjiFqAlk, ysEKLO8sMp8, 8Y5yJQVqZpU, dIT3p1kmf3M, -gRfP38ohaE,
    kjO4VIQpRmM) are queued for the fixed reader after the session A chain (`after_zap_reread.sh`).
- **Sprite quality (Cole: "some of them could use some better cutting out").** His example had a see-through hole
  where a badge/HP bar was removed, baked-in deploy dust, and stray specks. The synthetic generator pastes these
  sprites thousands of times, so one bad sprite repeats in every image that uses it.
  - **Audit of all 296 approved sprites** (`data/logs/slice_audit*.json`): 24 with a hole enclosed by the sprite,
    9 mostly dust/blur (low edge density), 1 with stray pieces. 33 flagged.
  - **Auto-fix:** holes inpainted (Telea) and made opaque; stray pieces removed. Fixed copies are in
    `data/slices/fixed/`, and originals are untouched until Cole decides.
  - **Review:** `slice_review_fix.html` (33 items, before | after on a checkerboard). ✓ = replace with the fixed
    version, ✗ = drop the sprite.
- **Cole's sprite fix review:** 22 replaced with the fixed version, 2 dropped, 9 queued for recut.
- **Group boxes (Cole: evo Bats / hero Goblins boxed as the whole group, "I thought we fixed that").** The group fix
  covered the auto-labeler only.
  - **Size check:** KataCR single-unit 90th percentiles are bat-evolution 2.18 tiles, goblin 1.40, skeleton 1.15.
    Our boxes: evo Bats median about 2 tiles, with 33% of click boxes over 2.5; hero Goblins 22% of click boxes over
    2.5. Clicks are the main leak: the detector box or SAM from one click grabs the flock. All 5 evo Bats / hero
    Goblins sprites were 3–4-tile blobs.
  - **Bug found:** `bats-evolution` and `skeletons-evolution` duplicated KataCR's `bat-evolution` /
    `skeleton-evolution` (same units, two names). Merged into KataCR's names (`FORM_CLASS`, cleaner `MERGE_FORMS`,
    sprite manifest).
  - **Fixes:** the cleaner drops main boxes for group cards above a per-card limit (`MAX_UNIT_TILES`); the group
    sprites are pulled and queued for recut (14 total).
- **Sprite recut (`clicks_recut_2026-10-08.json`, 28 clicks; Cole skipped the bad Goblinstein monster shots).**
  The cutter now fills enclosed holes on every sprite it writes (`fill_holes` in slices.py), so badge/bar erasure
  can't leave see-through holes again.
  - **Result:** most of the 28 new cuts are unusable. Level badges were left on; hero Goblins / evo Bats cuts still
    caught 2–3 units (SAM from one click outlines the bunch); the deploy stopwatch was included; Goblin Curse cuts
    included the floor; evo Bats cuts were motion-blurred.
  - **Decision:** none of them are added. For small, fast or grouped units, point-click cutting from gameplay frames
    doesn't produce clean sprites. Those classes learn from real frames (clicks, cleaned labels), and synthetic
    pastes use KataCR's clean base sprites. The approved library stays at 289 (hole-free).
- **Group-box filter check:** KataCR single-unit sizes: royal hog 98th percentile 1.94 tiles, goblin 1.49,
  bat-evolution 2.26. Ours: evo Royal Hogs median 1.45 but 75th percentile 2.86; hero Goblins median 1.76, 75th
  2.58. The limits (2.4 / 2.2) only remove real group boxes. After the filter (midday recount, 62 of 77): hero Goblins
  24 and evo Royal Hogs 29 are honestly short of single-unit labels; evo Bats (merged) 39.
  - Coverage no longer lists merged duplicate names or HUD classes (the stray `king-tower` main box).
- **Recount after the Zap re-reads and evo Zap clicks: 4,752 instances, 63 of 74 pass** (evo Zap passes).
  - Left: evo Bats 39, evo Princess 38, hero Magic Archer 32, evo Goblin Giant 32, evo Lumberjack 30, evo Royal Hogs
    29, Dark Prince dismounted 27, evo Hunter 26, hero Goblins 24, evo Inferno Dragon 18, evo Skeleton Army 11.
  - `click_short9.html` (76 frames) covers the six closable from existing footage. `build_click_short` now caps
    pages at 150 frames. For group cards the page says "click EACH unit separately", since every unit needs its
    own box or the unboxed ones teach "background".
  - The rest needs sessions B and C.
- **Page 9 (`clicks_short9_2026-10-08.json`): 214 clicks on 68 frames**, groups clicked unit by unit: hero Goblins 95,
  evo Royal Hogs 80, evo Bats 21, hero Magic Archer 12, evo Princess 6.
  - **Recount: 4,851 instances, 67 of 74 pass.** evo Bats, evo Princess, hero Magic Archer and evo Royal Hogs passed.
    The group filter dropped 192 group-box frames.
  - **Left:** hero Goblins 37, evo Goblin Giant 32, evo Lumberjack 30, Dark Prince dismounted 27, evo Hunter 26, evo
    Inferno Dragon 18, evo Skeleton Army 11. Mostly sessions B and C.
- **Click labelling made incremental** (`scripts/label_clicks.py`): results are cached per click file by content
  hash plus `CODE_VERSION`, so a review round processes only the new file instead of re-labelling every click
  (~1 h). `--force` redoes all; bump `CODE_VERSION` when the click-to-box logic changes.
- **Session B recording** (`cole_2026-10-09`, 36 min): evo Skeleton Army, evo Lumberjack, hero Dark Prince. Deck
  declared; standard crop. Chain: `scripts/after_session.sh` (process → auto-label → Dark Prince ability page →
  re-clean → recount).
- **Session B processed (12:36):** 11 matches; evo Lumberjack 11 plays, evo Skeleton Army 8, hero Dark Prince 37, with 10
  ability uses read. Auto-labels moved evo Lumberjack 30 → 32 and evo Skeleton Army stayed at 11, so clicks are
  needed. Pages: `click_sessionB.html` (evo plays at +2.5/4/5.5 s; evo Skeleton Army = click the General) and
  `click_abilities_cole_2026-10-09.html` (40 frames after the 10 ability uses).
- **Session B clicks:** 45 (evo Lumberjack 27, evo Skeleton Army 18) plus 29 ability clicks (28 dismounted Dark
  Prince; using the ability ASAP worked).
  - **Recount: 4,935 instances, 69 of 74 pass.** evo Lumberjack 58 and Dark Prince dismounted 54 pass.
  - **Left:** hero Goblins 37, evo Goblin Giant 32, evo Skeleton Army 26, evo Hunter 26, evo Inferno Dragon 18.
- **Session C recording** (`cole_2026-10-09_c`, 18.5 min; Cole: "not nearly enough"): evo Inferno Dragon, evo
  Hunter, hero Goblins. Deck declared.
  - `after_session.sh` now also builds the session's click page automatically (`scripts/session_click_page.py`):
    every evo/hero play of the declared cards, 3 frames each, capped at 150, with group cards asking for each
    unit.
- **Session C processed (14:33):** 6 matches; hero Goblins 20 plays, evo Inferno Dragon 5, evo Hunter 4. Auto-labels
  added nothing (still 69 of 74). Click page `click_session_cole_2026-10-09_c.html` (29 plays, 87 frames).
  - Expected after clicks: hero Goblins passes; evo Hunter about 34, evo Inferno Dragon about 28.
  - Still needed: session C2 (about 10 matches, evo Inferno Dragon + evo Hunter) and a top-up (evo Skeleton Army + evo
    Goblin Giant, no Phoenix).
- **Session C clicks:** 170 (hero Goblins 143, each goblin separately; evo Inferno Dragon 15; evo Hunter 12).
  - **Recount: 4,983 instances, 70 of 74 pass.** hero Goblins 62 passes.
  - **Left:** evo Hunter 36 (need 4), evo Goblin Giant 32 (8), evo Inferno Dragon 31 (9), evo Skeleton Army 26 (14).
  - Plan: session D (evo Skeleton Army + evo Goblin Giant, no Phoenix) and session E (evo Inferno Dragon + evo Hunter).
- **New card forms (Cole, 2026-10-09): hero Electro Wizard and evo Electro Giant.** Cole fetched a fresh API snapshot
  (`data/cards/api_snapshots/2026-10-09.json`).
  - The ingest aborted on a 404: Supercell's CDN hadn't uploaded the new heroMedium / evolutionMedium images yet,
    although the API lists them. `fetch_art` now skips missing images with a warning.
  - The card table has hero Electro Wizard and evo Electro Giant, and the class list appended
    `electro-giant-evolution` and `electro-wizard-hero` (217 → 219; existing indices unchanged).
  - hero Electro Wizard's hand art is cut from Cole's session D at 4:42 (art override). It reads 0.85–0.91 on other
    frames, including greyed cards.
- **Session D recording** (`cole_2026-10-09_d`, 37 min): evo Skeleton Army, evo Goblin Giant, hero Electro Wizard. Deck
  declared; processing via `after_session.sh` (auto click page included).
- **Session E recording** (`cole_2026-10-09_e`; Cole: "kinda short"): evo Inferno Dragon, evo Hunter, hero Electro
  Wizard. Deck declared; queued after session D's chain.
- **Supercell-Extractor (Cole found it; github.com/baraklevy20/Supercell-Extractor).** A Node tool that unpacks
  Supercell `.sc` asset files (textures, shapes, animated "movie clips") for Clash Royale and other games.
  - **Why:** the game's own unit sprites, cleanly cut, for every card, evo/hero form and animation frame. That fixes
    our weakest area: cutting sprites from gameplay fails for small/fast/grouped units, and new forms have no
    footage yet. It also enriches synthetic data and breaks co-occurrence pairings.
  - **Cole decided to use it.** Conditions:
    - extracted assets stay local and gitignored (the repo is public)
    - follow Supercell's Fan Content Policy
    - don't vendor the extractor's code (no licence in its README; package.json says ISC)
    - pilot first: compare extracted sprites with real frames before building on them
  - **Setup:** cloned to `~/tools/Supercell-Extractor` (outside the repo). Last commit 2022-04, so it may not read
    current asset formats. One-line patch for Node 26 (`require('yargs')`); native deps (sharp, lzma-native) load.
  - **Next:** Cole downloads the current Clash Royale APK (APKMirror, Supercell-signed); unzip `assets/sc/*.sc`;
    pilot extraction for hero Electro Wizard, evo Bats, hero Goblins, Skeleton Army, Goblin Giant.
  - **APK check (Cole's APKMirror bundle, version 160402020):** no `.sc` / `.sctx` files anywhere in the 30 parts
    (base plus splits, ~115 MB, mostly code). Current Clash Royale downloads its graphics from Supercell's servers on
    first launch, so the extractor has nothing to read from the APK.
  - **Options:**
    1. Android emulator: install, launch once, copy the downloaded assets out (hours of setup, a game login on an
       emulator, and the extractor may not read 2026 formats).
    2. Fetch from Supercell's content servers by imitating the game client. Rejected: beyond fan content.
    3. Drop it for now.
  - **Recommendation: 3.** Real-frame labels are the better signal and we're near the bar. Revisit the emulator route
    if v2's benchmark shows rare forms need clean-sprite synthetic data. The extractor stays set up in `~/tools`,
    and the APK bundle is in `~/tools/cr_apk`, outside the repo.
- **Session E processed (19:01):** 6 matches; hero Electro Wizard 23, evo Hunter 5, evo Inferno Dragon 4. Click page
  96 frames. Cole's session D clicks (101: hero Electro Wizard 64, evo Goblin Giant 25, evo Skeleton Army 12) arrived
  after E's label step, so they're counted in the next recount.
- **Session F** (`cole_2026-10-09_f`, 9 min): evo Electro Giant.
  - Art override cut from the charged card at 2:06 (pink goggles, purple glow). Normal colour cards still read normal
    (0.87), but **greyed normal cards in the evo slot read as evo**: the reader works in greyscale, and evo vs normal
    differ mostly by colour (pink vs blue goggles), plus the override includes the evo-slot frame top.
  - **Decision:** the reader's evo call isn't trusted for Electro Giant. Session F's click page shows every Electro
    Giant play and Cole picks evo or normal. A colour-aware form check is a to-do if it matters elsewhere.
- **Session E clicks:** 76 (hero Electro Wizard 50, evo Hunter 14, evo Inferno Dragon 12). These and the session D clicks
  are counted in a recount queued after session F's chain (`after_recount.sh`).
- **Prior art: Hasty-CR (github.com/hastylmao/Hasty-CR, MIT; Cole found it).**
  - **What it is:** a deterministic Clash Royale simulator (integer millitiles, 50 ms ticks; card stats decoded from the
    game client's own tables), plus PPO self-play (behaviour-cloned from a rule bot first), for one deck (2.6 Hog
    Cycle). Live play uses YOLO vision (0.959 mAP50) on an Android emulator.
  - **Reported:** 93.3% vs its rule bot and 86.7% vs meta decks, measured in its own simulator. Its stated vision gap:
    no unit HP ("a half-dead Musketeer looks healthy").
  - **Our assessment:** neither approach is strictly better.
    - Sim + RL can exceed human play and has perfect labels, but carries a sim-to-real gap (they list deadlocks and
      approximations) and a per-card mechanics burden (one deck).
    - Learning from pro footage covers every card from real play and is what goals C and A need.
  - **Decision:** stay on C → A. Plan a hybrid for B:
    1. initialise the bot from our pro-move predictor (better than a rule bot)
    2. fine-tune with self-play in a simulator (Hasty-CR's or similar, credited)
    3. use our pro event logs to validate the simulator against real play
  - **Ideas to adopt:** read unit HP from the HP-bar classes our detector already has, and decode exact card stats
    from client data instead of curated YAML.
- **Recount with session D and E clicks: 5,153 instances, 74 of 76 pass.** hero Electro Wizard 110, evo Goblin Giant 57,
  evo Hunter 49 and evo Inferno Dragon 43 pass.
  - **Left:** evo Skeleton Army 36 (need 4) and evo Electro Giant 0 (session F had only 7 Electro Giant plays; its page
    is 21 frames).
  - **Plan:** one session with evo Electro Giant (evo) + evo Skeleton Army (wildcard), about 10 matches.
- **Session F clicks:** 24 (evo Electro Giant 9, normal 15).
- **Final session** (`cole_2026-10-09_g`): evo Electro Giant + evo Skeleton Army. Deck declared. Its click page shows
  every Electro Giant play (Cole picks evo or normal) and every evo Skeleton Army play (click the General), capped at 150.
- **Pre-training audit (Cole: "verify that the GPU pass is right").**
  - **OK:** real_v2 has 4,771 frames, 0 label errors (class ids, box bounds, sizes), all 896×576, from 134 videos with
    no video over 6%. synth_v1's 158 classes match classes_v2[:158], so the sets can be mixed. Empty classes are
    harmless; merged duplicate names stay as unused indices so numbering never changes.
  - **Gaps (to fix before any paid run):**
    1. synth_v1 has none of our new classes; 27 approved-sprite classes are absent, so synthetic data must be
       regenerated with our 289 sprites.
    2. No video-level train/val split.
    3. Held-out benchmark not built.
    4. No v2 data.yaml / training config (current script is v1: 158 classes, synth only, batch 4).
  - **Order:** benchmark videos first (to prevent leakage) → split by video → synth_v2 → v2 config → 1-epoch local run →
    ~$1 cloud dry run → full run.
- **Benchmark hold-out chosen (`config/benchmark_holdout.json`):** Alina uz4VVzGlOjE_m01 (Cole's 52-play truth) plus Cole/friend
  2026-09-29 match 5 (cole_c_m05 ↔ friend_c_m07; pairing gives exact opponent plays, 34 + 25). That's 111 opponent plays.
  - Holding these out removes 105 training frames and no entity falls below 40 (the cole_c_m10 alternative would have
    dropped hero Barbarian Barrel to 37).
  - The cleaner now skips held-out matches. The crop classifier and mining must exclude them too before v2
    (to do in the split step).
- **Final session processed (22:15):** 5 matches; evo Electro Giant 6 (reader call, Cole confirms), evo Skeleton Army 5. Page
  63 frames.
  - **Recount with the benchmark held out (125 frames): 73 of 77.** Boss Bandit fell 40 → 36 because the held-out match
    had 4 of its frames.
  - Boss Bandit page (`click_bossbandit.html`) uses its unclicked medium-confidence plays outside the hold-out.
  - Normal Electro Giant (clicked while choosing evo vs normal) is a KataCR class and is excluded from the bar
    (`INCIDENTAL` in coverage).
  - Still short after these pages: evo Electro Giant (8, needs about 32), probably one more session.
- **Recount after the final session's clicks (61) and Boss Bandit clicks (33): 5,154 instances, 75 of 76 pass** (benchmark
  held out). Boss Bandit 68 and evo Skeleton Army 47 pass. **Only evo Electro Giant is short (23, needs 17)**: one more
  short session.
- **Decision (Cole, 2026-10-09): train now; evo Electro Giant goes in at 23 (under the bar).** "lets just try it rn.
  please make sure nothing will go wrong." Run 1 uses real_v2 (cleaned, benchmark held out) plus synth_v1 as is.
  synth_v2 (with our 289 sprites) is deferred to run 2: the new cards are covered by real frames, and regenerating
  costs hours.
- **Training prep for run 1 (all local, verified before paying):**
  - `scripts/build_train_v2.py` → `data/external/train_v2`, 219 classes. Train: real_v2 from most videos + synth_v1
    (12,000). Val: real_v2 from 34 videos + KataCR's 800 hand-labelled real frames (v1's val set, for comparison).
    16,288 train / 1,294 val images.
  - **Split fix:** a random 10% of videos took most of some rare cards' frames (Boss Bandit 23 train / 40 val, evo
    Minion Horde 6 / 29). Val videos are now chosen so every entity keeps ≥75% of its real frames in train.
  - `scripts/train_v2.py`: same script for Mac and cloud (auto device; batch 4 MPS / 32 GPU; v1 augmentation, no
    mosaic). **Smoke test** (`--smoke`: 1 epoch, 3% of data, MPS, ~3 min): nc=219, 0 corrupt labels, train + val ran,
    weights saved. The only warning was benign (auto optimizer).
  - `scripts/eval_v2.py`: scores any weights on the held-out benchmark into separate tables. Tested with the smoke
    weights; it now reports "nothing detected" instead of crashing. v1 is being scored as the baseline.
  - Upload bundle: `data/upload/` (train_v2.tar + script + base weights + sha256), gitignored.

## 2026-10-10 — v1 baseline on the held-out benchmark (the number v2 must beat)

`scripts/eval_v2.py <v1 weights> v1bench` → `data/logs/eval_v1bench.json`. Opponent plays, scored against truth:

| match | true plays | found | fully right | wrong card | false plays | recall | fully right | precision |
|---|---|---|---|---|---|---|---|---|
| Alina uz4VVzGlOjE_m01 | 52 | 44 | 22 | 10 | 12 | 0.62 | 0.42 | 0.73 |
| Cole 2026-09-29_c m05 | 34 | 26 | 2 | 4 | 20 | 0.18 | 0.06 | 0.23 |
| friend 2026-09-29_c m07 | 25 | 30 | 0 | 10 | 20 | 0.40 | 0.00 | 0.33 |

v1 is usable on an older pro match and close to useless on current-meta games with evo/hero forms and new cards
(Goblinstein, Rune Giant, etc. are not v1 classes). Target for v2: well above 0.42 fully right on Alina, and real
numbers (not ~0) on the two 2026-09-29 matches. RunPod plugin installed (runpod@runpod 1.5.0, OAuth, no API key).

## 2026-10-10 — v2 training on RunPod

- Pod `bot77-train-v2` (id 03ouipsy0lqm06): 1× RTX 4090, Secure Cloud, EUR-IS-1, $0.89/hr, image
  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, 40 GB persistent /workspace (network FS) + 20 GB local disk.
  Access over direct SSH with Cole's public key (registered via the RunPod MCP; OAuth, no API key).
- Bundle uploaded with rsync (2.7 GB, ~15 min; `-rt` because the network FS refuses chown), sha256 verified on the
  pod, unpacked to local disk (/root/data) since 16k small images read slowly from the network FS; data.yaml `path`
  repointed. ultralytics pinned to 8.4.160 to match the Mac, so weights load back home without version drift.
  pip on this image needs `--break-system-packages` (PEP 668).
- Dry run (yolo11s, 1 epoch, batch 32): clean exit, 0.045 h/epoch, GPU 98%, val mAP50 0.441 / mAP50-95 0.294
  after one epoch. Pipeline proven.
- Full run: **yolo11m**, 100 epochs, batch 16 (11m at batch 32 won't fit 24 GB at imgsz 896; nbs=64 keeps the
  effective batch), patience 20. ~4 min/epoch → ≤ ~6.5 h, ≤ ~$6. Weights land on /workspace/bot77/runs/v2m.
  Then: pull best.pt, stop the pod, run eval_v2 against the v1 baseline above.

## 2026-10-10 — v2m result: not usable yet; root cause is sparse labels in real frames

- Run stopped by early stopping at epoch 23 (1.2 h). Total pod time ~1.75 h ≈ $1.56. Weights + results.csv/png +
  log pulled to data/models/bot77_detector/v2m/ (sha256 checked), pod stopped then terminated (list-pods empty).
- Curve: train loss falls throughout; val mAP50 rises 0.58 → 0.68, val mAP50-95 peaks at epoch 3 (0.409) and drifts
  down while val box loss rises. Early stopping tracks mAP50-95, so best.pt = epoch 3.
- Split val: last.pt beats best.pt on real frames (mAP50 0.608 vs 0.538, mAP50-95 0.358 vs 0.338); only synth
  (KataCR hand-labelled) mAP50-95 drops (0.574 → 0.460) — our real boxes are looser than KataCR's.
- Play benchmark (fully right / true plays):

| weights | Alina | Cole m05 | friend m07 |
|---|---|---|---|
| v1 | 22/52 | 2/34 | 0/25 |
| v2m best (epoch 3) | **25/52** | 3/34 | 0/25 |
| v2m last (epoch 23) | 17/52 | 2/34 | 1/25 |

- Detector-level check (does any enemy detection of the right card appear 0–3 s after each true play): Alina v1 44/52,
  v2 39/52; Cole m05 v1 8/34, v2 6/34; friend m07 1/25 both. On Cole's footage v2 finds towers and bars at 0.9+
  but almost no troops (frames at 939 s / 960 s: skeletons, Electro Spirit, goblins, Tombstone unboxed).
- **Root cause:** real_v2 frames average 2.6 unit boxes (one verified main box + whatever v1 caught), synth frames 26.
  Every other troop in a real frame is unlabelled, so the model learns "troops in real footage are background"; the
  longer it trains, the worse (best.pt at epoch 3 > last.pt at epoch 23 on Alina). Real frames are where all the new
  cards live, so the new cards suffer most.
- Also fixed: `class_to_card` dropped every `*-hero` class (and `dark-prince-hero-dismounted`), so hero plays could
  never reach the opponent log; `tesla-evolution-shock` is now a non-unit.
- Lesson for the pipeline: pick checkpoints by real-val mAP50 / the play benchmark, not by mAP50-95 on a mixed val.

### Fix plan (run 2)
1. synth_v2: regenerate KataCR-style synthetic arenas with the 289 approved sprites, so every new card/form appears
   in fully labelled frames (the KataCR recipe; no unlabelled troops by construction).
2. Real frames: stop teaching "background" — crop each real frame to a window around its verified box(es) and fill
   the window with pseudo-labels (v1 + v2m-best ensemble, crop-classifier agreement); anything detected but not agreed
   is masked. Frames with unlabelled units left in the window are dropped.
3. Val/selection: real val split reported on its own; checkpoint chosen on real-val mAP50, then the play benchmark.

## 2026-10-10 — Run 2 build: synth_v2 + empty-arena real frames

Cole: "yea do it" (fix plan above), and "I bought $30 of gpu credits. we can and should use all of them."
Budget used so far ~$1.56. Run 2 is sized to it: synth generated on the pod's 64 cores, yolo11l, ~40 epochs, no
early stopping, a checkpoint every 5 epochs, chosen by the play benchmark.

**Sprites (`scripts/auto_sprites.py`).** 52 classes had < 8 approved sprites (all hero forms, most evos: synth_v1
had none of them). Cut up to 10 per class with SAM from verified real_v2 boxes (Cole's clicks / approved / mined /
form review). real_v2 frames are already the generator's 576×896 arena crop, so no rescaling. Badge/HP bar erased,
largest component, leftover badge stripped, holes filled; kept if not clipped by the frame edge, the mask fills
20–97% of the box and the crop classifier gives the card family p ≥ 0.3 — except where < 5 boxes pass (classifier
never saw hero Magic Archer / evo E-Giant: all rejected), then best-ranked anyway. Max 3 per video. → 472 sprites.
My look at a sample: hero Balloon, evo P.E.K.K.A, evo E-Giant clean; hero Goblins includes level badges cut as the
unit; Vines includes a tower; some evo Skeleton Army are blobs. Cole reviews `data/reviews/sprite_review_v2.html`
(one row per class, keep by default, click the bad ones).

**Generator (`scripts/katacr/setup_synth_v2.py`).** KataCR clone kept at ~/tools/KataCR (the run-1 clone lived in
a temp folder and was gone). label_list: classes_v2[155:] appended (64 classes) so generator ids == classes_v2 ids;
evo/hero forms copy their base card's ground/flying/spell lists (KataCR's choices, e.g. Balloon "ground"), new
cards set by hand. Sprites copied as `<class>_1_*.png`. After setup only Mirror, tesla-evolution-shock,
zap-evolution and the two merged names have no sprites. Test: ~1 s/image/core, ~76 boxes/image, new classes
present. `gen_parallel.sh` runs one seed per core on the pod.

**Real frames (`scripts/build_real_v3.py` → data/external/real_v3).** Each real_v2 frame becomes: a window around its
verified box(es) (+1.5 tiles, grown over any kept box it cuts), pasted (feathered) onto an EMPTY arena of the same
match — the per-pixel median of frames across that match (units move, the arena doesn't; same crop geometry, real
theme and lighting). Inside the window: pseudo-labels from v2m-best (epoch 3) where the crop classifier's top-1 is
the same card family (HUD classes at conf ≥ 0.4), unconfirmed detections painted over with the empty arena. Towers
and tower bars kept from the frame; a tower standing in the empty arena but destroyed in the frame gets the frame's
pixels (ruin). Empty arenas where v2m still finds a unit are skipped.
- First empty arenas had the enemy half tinted red (the "can't place" overlay while a card is held). Fixed: 30
  frames sampled, the 15 least red used for the median.
- 40-frame test: every frame kept, 22 pseudo-labels, 14 detections painted over, 5 ruins restored.

**Bundle (`scripts/build_train_v3.py` → data/upload/train_v3).** real_v3 split by the same val videos as train_v2;
val = real_v3 val + KataCR's 800; extra yamls for real_v2 full-frame val (recall on real footage) and each val alone.
The generator kit ships in the bundle; `pod_assemble.py` writes train.txt with real frames ×3 + synth_v2.
`train_v2.py` gained `--patience 0` (off) and `--save-period`.

## 2026-10-10 (evening) — Run 2 launched

- Cole's sprite review (`data/ground_truth/reviews/sprite_review_v2.json`): 472 auto-cut sprites, 112 rejected,
  360 kept. Spells cut worst (Vines 6/7 bad, Void 8/10, Goblin Curse 3/4). setup_synth_v2.py → 649 sprites in 68
  classes on top of KataCR's segments.
- First pod drew a bad machine (no runtime, no logs after 11 min) → deleted, new pod `bot77-train-v3`
  (id nl4bu55935xmbq), RTX 4090 Secure, $0.89/hr, 80 GB disk + 60 GB volume.
- macOS tar adds `._*` AppleDouble files; KataCR's background counter matched them (background51.jpg not found) and
  they would break the sprite-name assert. Deleted on the pod; bundles now built with COPYFILE_DISABLE=1 --no-xattrs.
- synth_v2 generated ON the pod: 24 processes × 2,000 = 48,000 images in ~12 min. Check on 6,000: 0 bad labels,
  72 boxes/image, every new class present (~580 boxes per 6,000 images; rarest Vines 565); only the merged names absent.
- real_v3: 692 empty arenas built (30 frames, 15 least-red, median); 4,357 frames kept (424 dropped: empty arena still
  had a unit), 2,379 pseudo-labels added, 1,334 unconfirmed detections painted over, 735 tower ruins restored. Cards
  losing the most frames: Berserker hero 48→29, evo Inferno Dragon 43→29, Goblinstein Doctor 17→11 — all now also
  in synth_v2.
- Bundle: 3,909 real train ×3 + 48,000 synth; val = 448 real_v3 + 800 KataCR; extra val sets realfull (494 real_v2
  full frames) and each alone.
- Smoke (yolo11l, batch 16, 3%): clean, 20.7 GB of 24. Full run: **yolo11l, 35 epochs, batch 16, no early stopping,
  checkpoint every 5 epochs**, ~14 min/epoch → ~8–9 h, ~$8.
- `scripts/watch_run.sh` (on the Mac): pulls each epochN.pt as it lands and runs the play benchmark on it
  (`data/logs/eval_v3l_eN.json`), then pulls best/last/results at the end.
- Cole asked about extracting assets from his BlueStacks install: on the list for after run 2 (local files of his
  own install only; nothing from Supercell's servers, no memory reading; assets stay local and gitignored).
- **Game asset extraction from Cole's BlueStacks install (2026-10-10, 20:40), no root, no adb.**
  - Route: BlueStacks Air keeps Android's /data as `/Users/Shared/Library/Application Support/BlueStacks/Engine/Tiramisu64/data.qcow2`.
    - BlueStacks' own `qemu-img` converts it (read-only, `-U`) to a sparse raw ext4 image in `~/tools/bluestacks/`.
    - BlueStacks' own `7zz` reads the image. Not encrypted.
    - Nothing from Supercell's servers, no memory or network reading; only files already on his disk.
  - Why the APKMirror check found nothing: the art is in a Play Asset Delivery split,
    `split_install_time_asset_pack.apk` (913 MB), which APKMirror bundles leave out. It holds 2,106 `.sc`, 2,167 `.sctx`
    and 1,425 `.glb` files, unpacked to `~/tools/bluestacks/pack/` (outside the repo, never committed).
  - Decoder: sc-editor (danila-schelkov/sc-editor, MIT, v1.6.24) CLI `inspect` / `render` reads SC2 v5 + `.sctx`.
    - Needs Java (brew openjdk) and Khronos KTX tools for ASTC (unpacked from the official pkg to `~/tools/ktx`, no sudo).
    - Wrapper: `~/tools/sc-editor/sce`.
    - First render (red and blue Giant, run direction 3) is pixel-clean with alpha.
  - Coverage (163 `chr_*.sc` files):
    - 2D sprite animations for nearly every classic card and most evos, in both team colours
      (`<unit>_enemy_<anim>1_<dir 1..9>` or `_red_`). New cards: Goblinstein, Berserker, Goblin Demolisher, Goblin Machine.
    - **3D only** (`sc3d/characters/*.glb`, Supercell "Odin" glTF): all 18 hero forms, Minion Giant, Ronin, Boss Bandit,
      evo Electro Giant, evo Elite Barbarians. Their `.sc` files hold only effects.
  - Run 2 so far (epoch 5), plays fully right: Alina 27/52 (v1 22), Cole 9/34 (v1 2), friend 4/25 (v1 0).
- **Game-art sprites for V3 (2026-10-10, evening).** Cole: "set it up so we got everything and everything is good for V3".
  - **2D** (`scripts/extract_game_sprites.py`): SPEC maps 147 of our classes to their `.sc` file and red/blue export prefixes.
    - Many classes go by Supercell codenames: black_knight = Dark Prince, ghost = Royal Ghost, axe_man = Executioner,
      rage_barbarian = Lumberjack, angry_barbarian = Elite Barbarians, dark_witch = Night Witch, zap_machine = Sparky,
      guard02 = Royal Recruits, skeletonWarrior = Guards, snowman = Ice Golem, buffing_giant = Rune Giant,
      merge_maiden (+ _dragon) = Spirit Empress ground/air, furnace_rework = Furnace (now a troop).
    - Every animation × 9 directions × 2–4 frames, both teams, plus Star Level skins. Skins only hold the overlay
      pieces, so they're composited over the base unit unless they cover ≥70% of it.
    - Render on a fixed canvas centred on the unit's origin so layers line up (Ram Rider, Goblin Giant + bag).
    - Filters: duplicates, mask shapes sc-editor draws as flat dark blocks, clipping.
    - Result: **32,381 sprites, 145 classes.**
    - Removed after the contact-sheet review: evo Minion Horde (dark silhouette), Elixir Collector (UI badge),
      Mother Witch and Cannon Cart skin 2, Archer Queen's ability (cloak).
    - Check: the 2D Giant matches Giants in Cole's 2026 footage, so the 2D art is what battles show.
  - Spells are particle compositions, so they keep the real cut-outs. Projectiles that get labelled are included
    (Goblin Barrel, Royal Delivery, Snowball).
  - **3D** (`scripts/blender/render_3d_units.py`, background Blender 5.2 + gltf-Supercell-IO, used as a tool):
    hero forms, Minion Giant, Ronin, evo Electro Giant, evo Elite Barbarians.
    - Textures: decoded locally (KTX-Software). The extension's texture web service is stubbed out, so no game files
      are sent anywhere.
    - Camera: 47° elevation, calibrated against the official 2D Giant rendered from the same model (silhouette IoU 0.83–0.90).
    - Materials rebuilt from the file's uber_pbr parameters, including the team mask (blue channel of `_spec`).
    - Animation fixes: the plugin only retargets onto the *selected* rig. The top helper bones (toEngineGrp /
      EngineJoints) also parent the meshes, so their keyed 90° rotation was applied twice (units lay down);
      those channels are dropped.
    - Result: Minion Giant and Ronin render clean. Several hero rigs still come out stretched or with detached
      props. Each class is triaged by eye; failures keep the real cut-outs.
  - **Cutouts vs sprites (Cole asked):** sprites are better: clean, every pose/direction/skin, both teams.
    - Real cut-outs are kept at ~10% of a class's files (uniform sampling within a class), so the model still sees
      video-looking units.
    - They remain the only source for spells and failed heroes.
    - 70% of game sprites get a one-off video degradation (down/up-scale blur + JPEG 55–85).
  - Scale (`scripts/katacr/setup_synth_v3.py`): game art is drawn at its own resolution (evo sprites ~2×; median ratio
    to arena-crop size 0.91), so each class is resized to the median height of its real cut-outs, else 0.91.
  - Run 2 so far: e5 27/9/4, e10 25/10/3 plays fully right (Alina/Cole/friend). Detection recall on Cole's match is up
    (found 0.41), but 21–25 false plays per match on Cole's and the friend's games. The detections→plays step is now
    the bottleneck.
- **Goal set by Cole (2026-10-10, late): ~90% of opponent plays fully right by the next version, without overfitting
  to the 3 benchmark games.** Working rules:
  - The benchmark is only ever scored.
  - Play logic is tuned on a **development set**: the 22 other dual-POV paired matches, 473 exact opponent plays
    (`scripts/eval_dev.py`; fit/check split cole_* vs friend_* in `scripts/tune_decoder.py`).
  - Caveat: the detector trained on frames from those videos, so detection there is optimistic.
- **Dev baseline (run-2 epoch 10, v1 play logic): 67/473 fully right (14.2%), 14.8 fake plays per match.**
  `scripts/analyze_dev.py` breaks it down:
  - Missed (334): 151 cards never detected; 56 timing (outside 1.5 s); 53 seen as enemy but no play;
    27 hero/champion abilities (never output); 27 Mirror; 20 seen as the wrong team only.
  - Fake (325): mostly spawns and cards the v1 rules predate. Goblinstein's doctor wandering back,
    Suspicious Bush / Goblin Drill goblins, hero Tombstone's skeletons and Tomb Queen, Spirit Empress form switches,
    and our own units near the river called enemy.
- **Play rules v2** (`src/bot77/playrules.py`, `--rules` in eval_dev / eval_v2): continuation, long-lived spawners,
  Tomb Queen → `__ability`, our-deck cards near the river, per-card timing (`config/play_timing.json` from dev tracks;
  e.g. Void 3.35 s, Ronin 3.1 s, Goblin Drill 2.7 s, Ice Spirit 1.0 s).
  - In `spawns.py`: only Tesla resumes after a long gap (a new Tombstone often goes on the same spot), and births get
    2 s to confirm (1 s lost slow-to-detect new cards).
  - **Dev: 14.2% → 26.8% fully right; fakes 14.8 → 9.7 per match.**
  - No gain: Mirror as a quick repeat (8/15/25 s), dismounted hero Dark Prince → ability. Both left off.
- **Game-rules decoder** (`src/bot77/decode.py`: deck, 4-play cycle as a penalty, Mirror, drop weak candidates):
  +1 point on the fit half, 0 on the check half, best with the cycle penalty off. With ~40% of plays detected, too many
  plays are missing for the cycle to help. Kept for when recall is higher; not used yet.
- **Time convention:**
  - Paired/POV truth marks the moment the card leaves the hand. Detections come 1.2–1.5 s later.
  - Alina's hand-made labels mark unit appearance (median detection − label = 0.0 s).
  - Canonical play time is card-leaves-hand. Events now also store `t_seen` (first unit seen).
  - Alina's truth file is marked `time_convention: appearance` and scored against `t_seen`. This fixes the label
    definition, not a fitted parameter.
- **Benchmark, run-2 epoch 15 (score only):**
  - v1 logic: 29/52, 9/34, 4/25 = 42/111.
  - Rules v2: 27/52, 10/34, 6/25 = 43/111. Fakes fell from 14/29/27 to 8/12/21.
- **Run 2 stopped at epoch 16 (of 35)** to start V3 sooner. e5 → e10 was flat; V3 fine-tunes from epoch 15 anyway.
- **V3 launched** on the same pod (`scripts/katacr/pod_v3.sh`):
  - synth_v3: 48k images from `segment_v3.tgz` (32,207 game sprites + 6,276 accepted 3D renders + real cut-outs,
    166 classes, per-class scale).
  - Fine-tune from v2 epoch 15 for 30 epochs; run name `v4l` on the pod.
  - 3D triage: accepted 20 classes; rejected hero Giant (stretched arms), hero mini P.E.K.K.A (props off the body),
    mounted hero Dark Prince (no rhino), hero Ice Golem (armour floats).
  - Vines added as layered spell sprites (coil back + front, 5 sizes, 80 sprites). Void and Goblin Curse have no art
    in the pack.
- **V4 data engine started** (`scripts/mine_v4_extract.py`): arena crops at 4 fps around the 4,523 trusted plays
  (benchmark excluded), ~185k frames.
  - Next: detect them on the pod with V3, then auto-label (exact class from the hand reader) → real_v4 → V4 training.
- **V3 training (2026-10-11 ~00:10 PDT):** 3,909 real ×3 + 48k synth_v3, from v2 epoch 15 (all 1,015 tensors
  transferred), 30 epochs at ~15 min each, ending ~07:30 PDT, ~$6.70.
  - Watcher: `watch_run.sh … v4l` scores the benchmark per checkpoint. Checkpoint choice will use the dev set.
- **V4 data engine, automated:**
  - Mining pod `bot77-mine-v4` (A40, $0.59/h, id xo7fugj7naawvp) receives the play-window frames (rsync loop) and
    detects them with V3's epoch-10 checkpoint (`scripts/mine_v4_chainA.sh`). Terminated right after.
  - `scripts/mine_v4_chainB.sh`: import → `autolabel_v4.py` (2 workers; 16 GB Mac) → `clean_dataset.py`
    (now takes src/dst) → `build_real_v3.py` (now takes --src/--dst/--pseudo/--bg-dir) → data/external/real_v4.
  - Smoke test on 4 dev plays: 18 labelled frames.
- **Why not upload the videos:** 69 GB at ~2 MB/s would take ~9 h. The arena crops at JPEG quality 80 are ~10 GB, and
  extraction is resumable.
- **Hero/champion abilities (5.7% of dev plays) are not handled yet.** Candidate approach for V5: ability-summon
  classes from the 3D/2D files (Goblins totem, Musketeer turret, Ice Wizard snowman, Magic Archer decoy,
  Berserker bear, Goblinstein's beam), mapped to `__ability` like Tomb Queen. 612 POV ability presses give real labels.
- **Dev set, V3 epoch 5 with rules v2: 25.2% fully right** (benchmark is still score-only).
  - Detection cut-off sweep (`BOT77_MIN_CONF`, dev only): 0.5 → 25.2% (9.8 fakes/match), 0.4 → 26.4% (10.9),
    0.35 → 24.9% (12.0). The new classes often score 0.35–0.5 this early.
  - Kept 0.5: 0.4's gain is small and adds fakes. Re-check on the cole/friend split once training is done.
  - Planned once V4 exists: a learned "detected class → card played" table fitted on one half of dev and checked on
    the other, combined with the inferred deck (e.g. "poison" → Goblin Curse when Curse is in the deck and Poison
    isn't). It has to be fitted on the final detector because the confusions change between models.
- **Mining done (2026-10-11 01:56):** V3 epoch 10 detected all 185,167 frames on the A40 in ~30 min; pod terminated.
  Chain B (import → auto-label → clean → real_v4) started.
- **Dev set, V3 epoch 10: 27.3%** fully right (epoch 5: 25.2%). Plays found 37% → 45%; wrong card 54 → 82.
- **real_v4 built (chain B):**
  - 8,765 auto-labelled → 5,342 after cleaning → 5,241 composites; split by video: 4,488 train / 753 val.
  - Mostly the same plays as real_v3 (3,920 of 5,241 frames share match+entity), so it adds new frames of known
    plays, not new plays.
  - Paired-opponent spells are still never auto-labelled (mirrored spell spots were 0/9 right in round 1).
- **What the detector outputs on missed plays** (scratchpad diagnostic, dev only):
  - Void is detected in 20 of 27 plays, but 16 come out as team "friendly": spells have no badge, and an enemy
    Void lands on our half.
  - Goblin Curse is always read as clone/poison.
  - The "clock" deploy-timer class appears with most Ronin, Bush, Boss Bandit and Minion Giant plays, even when the
    unit itself isn't recognised.
- **Fixes (dev, V3 e10 detections, rules v2):**
  - Spells team-agnostic (`deploys(spell_team="enemy")`; our own spells removed by the POV log): plays found
    44.6% → 46.7%, fully right 27.3% → 26.6% (the extra spells mostly get the wrong card). Kept, because recall is
    what the V4 detector and the confusion table build on.
  - Track card = confidence-weighted vote. Whole track: 25.2% (tracks jump to other units later). First 1 s:
    27.9% with 10.4 fakes/match (2 s: 27.3%). Kept the 1 s vote (`BOT77_VOTE_WINDOW`).
  - Both are A/B switches by env var.
- **Confusion table: no gain, dropped** (`scripts/fit_confusion.py`, V3 e20 dev detections).
  - Fit on cole_* → friend_*: 21.9% → 21.9%. Fit on friend_* → cole_*: 39.8% → 39.8% at best, and much worse with a
    weak prior.
  - The two halves face different opponent decks (each recorder's opponent is the other player), so the confusions
    don't carry over. A table fitted on one deck is overfitting.
  - The halves also differ a lot in difficulty: cole_* 39.8%, friend_* 21.9%. The deck in the friend_* recordings
    (Vines, Ronin, Suspicious Bush, Goblin Curse, Boss Bandit, Void) is where most misses are.
- **Dev re-score with current play code:** V3 e10 27.9%, e15 30.4%, e20 30.7%.
- **Opponent spells labelled from paired truth** (`scripts/autolabel_paired_spells.py`):
  - Paired truth gives the card and time; the detector's new spell boxes 0.3–2.5 s later are relabelled as the
    true spell. Only classes that look alike may be relabelled (Curse: curse/poison/clone/rage; Void: void;
    Vines: vines/poison/clone). Unconstrained, about 1 box in 4 was wrong.
  - 92 frames from 52 of 88 plays → `real_v4s`, added to the V4 bundle (4,580 train / 753 val).
  - These plays are dev plays, so they make the dev set more optimistic. The benchmark game
    friend_2026-09-29_c_m07 (same opponent deck) is the honest check.
- **V3 checkpoints on dev (current play code):** e10 27.9%, e15 30.4%, **e20 30.7%**, e25 29.0%, last 27.9%.
- **V4 launched 2026-10-11 06:25** (`pod_v4.sh`, run `v5l`): from V3 e20, 25 epochs, training list
  3,909 real_v3 ×3 + 4,580 real_v4 ×1 + 48k synth_v3. Same RTX 4090 pod. Watcher `watch_run.sh … v5l`.
- **V4 checkpoints on dev (old play code):** e0 29.8%, e5 32.6%, e10 31.9%, e15 33.0%. Benchmark (score-only):
  e5 41/111, e10 44/111, e15 46/111. More epochs on the same data stopped helping after e5.
- **Why V2 → V4 plateaued (dev error analysis, 2026-10-11):**
  - The big jump was play logic (14.2% → 26.8%), not the detector; each retrain since added ~3 points.
  - Wrong cards are ~60 plays; plays with no event at all are ~257. Recognition is not the bottleneck.
  - The auto-labeller only labels plays the detector half-sees, so retraining on mined data improves the same half.
  - Never-caught groups: spells (Vines/Void/Curse, no deploy clock), hero abilities, Mirror (looks like the copied
    card), units hidden by towers/spells, and Cole's deck cards in red (the detector sees their level badge but
    not the unit: Minion Giant named in 2 of 17 clocked plays, Suspicious Bush 3–5 of 16).
- **Deploy clocks as a play signal** (`src/bot77/clocks.py`, rules `clock` in `playrules.py`, on by default):
  - Measured first: 91 of 239 missed plays had an enemy clock start within 1.5 s; 153 of 206 fakes had none.
    Only 44% of all plays show a clock (spells, abilities and some Mirrors never do).
  - Clock start lags the true play time by a median 0.85 s (cole_*) / 1.10 s (friend_*): offset 1.0 s.
  - A deploy near a clock takes the clock's time; a clock with no deploy becomes one, named by the detections at
    its spot over 3 s (cardless clocks dropped: naming them from the cycle gave +5 plays but +5 fakes/match).
  - Team from the river side (the landing units' badge team: −8 plays). Clocks above y 1.5 tiles are the HUD
    trophy icon. Requiring a clock for every unit play cut fakes 10.4 → 8.6 but lost 8 plays: not used.
  - Mirror by the cycle: the same card again within 25 s with < 2 plays in between → Mirror (+4 / +1 on the two
    halves). The game-rules decoder still drops too many candidates (its keep scores predate clocks).
  - Candidates left with no deck card are dropped (fakes 12 → 9.7/match, no plays lost).
  - The crop classifier cannot name the unit at an unnamed clock (9 of 79 right): not used.
  - Tuned on cole_* vs friend_* halves of dev (V4 e15 detections); both halves gained.
  - **Dev: V4 e15 33.0% → 42.3% fully right, fakes 10.4 → 9.5/match.** All V4 checkpoints gain 4–10 points
    (e0 35.3%, e5 40.4%, e10 40.4%, e15 42.3%).
  - Benchmark (score-only, e15): 49/111 (was 46), fakes 62 → 31. Alina 27/52, cole_m05 11/34, friend_m07 11/25.
- **Clock-spot labels for V5** (`scripts/autolabel_clock_spots.py`): paired truth names the card, the enemy clock
  gives the exact spot, the level badge above the unit anchors a SAM point prompt (prompting at the clock boxed
  the clock icon; flying units are drawn 1–2 tiles above their spot). Single-unit troops/buildings only, DINOv2
  similarity ≥ 0.4, plays with another play within 2 s skipped. Trial: 18 of 21 boxes right; the 3 wrong ones
  were all below 0.4 similarity.
  - Three dev games (both POVs) are kept out of these labels (`config/v5_check_games.json`, 152 plays) to measure
    V5's gain on games it did not learn the new labels from.
- **V4 done 2026-10-11 12:27; pod terminated.** Dev with clock logic: e0 35.3%, e5 40.4%, e10 40.4%,
  **e15 42.3%**, e20 41.0%, last 41.2% (fakes 8.6/match for e20/last). **V4 = epoch 15** (clock settings were tuned
  on its detections, so it has a slight home advantage). Benchmark (score-only, e15): 49/111.
- **Clock-spot labels:** 70 frames from 46 plays (`data/external/real_v5c_raw`), about two thirds right by eye;
  review page `data/reviews/v5_clock_labels.html` for Cole. The dev pool is too small to move the detector: V5
  needs new paired recordings (target ~40 games, ~8 new decks, spell- and ability-heavy) or both-hands broadcasts.
- **V5 recordings: Cole plays both sides himself** (two accounts, friendly battles, both screens recorded). Skill
  doesn't matter to the detector; it needs to see every card in red, in many spots, in fights. One person plays one
  side at a time, so plays come out more spread apart than in real games. That makes labels cleaner, but the games
  must still include forced fights, spells on groups and abilities, or the detector only learns isolated drops.
  These games train the detector only, never the pro move predictor (A).
- **Tournament broadcasts don't show both hands** (Cole, 2026-10-11), so they can't give opponent truth. YouTube
  stays an unlabelled pool: at most a source of pseudo-labels (clock spots + confident detections), never scored.
- **First self-play recording (2026-10-11 2:47 PM):** Cole played both sides; A = 2:47:28 (3024×1898 Mac,
  1064×1896 window, converted at its own 9:16 aspect, auto layout `auto_cole_2026-10-11_a`), B = 2:47:53
  (friend's old Mac setup, `friend_iphone_mirroring`). 2 games, paired at −26 s (expected −25, spread 0.00).
  - Decks read: A Tombstone, Ice Wizard, Skeleton King, Barbarian Barrel, Phoenix, Graveyard, Poison, Tornado;
    B Firecracker, Skeletons, Electro Spirit, Ice Golem, Fire Spirit, Ice Spirit, Goblin Curse, Rage.
  - Opponent truth: 84 + 73 plays from B, 33 + 30 from A. B's count is real, not overcounting: an all-cheap cycle
    deck in triple elixir, and 0 violations of the 4-card cycle in either POV log.
  - Disk filled at the end of this run (25 GB free after Cole cleared space). Converted `.mp4`s are what the
    pipeline reads; the `.mov` originals can go once a recording's layout is verified.
  - Cole confirmed both decks are right. The `.mov` originals are moving to external storage; the pipeline reads only
    the `.mp4`s, so nothing breaks (the `cole_2026-10-11_*.mov` links will just dangle).
- **Second self-play session (2026-10-11, 10:44 PM and 11:01 PM), two games, nothing usable yet:**
  - The Mac recordings (10.44.43, 11.01.34) are black for their full length (5 MB each). They were recorded with
    "Record Selected Window" (1066×1898), which doesn't capture BlueStacks' game picture. The 2:47 PM full-screen
    recording worked. Use Record Entire Screen / Selected Portion and do a 10 s test first.
  - **I destroyed the Mac copies of both iPhone recordings.** I linked `cole_2026-10-11_d.MP4` to the original and
    wrote the converted file to `cole_2026-10-11_d.mp4`. The Mac filesystem ignores case, so ffmpeg overwrote the
    original it was reading (now 0.5 s, about 0.5 MB). Same for `_f`. The originals are still in the iPhone's Photos and need
    exporting again. Rule from now on: a link to a source file always gets a different base name than any output
    (e.g. `<id>_src.MP4`), never one that differs only in case.

## 2026-10-12 — Replays instead of self-play
- Cole hates playing both sides, and friends can't record. Decoding replay files was declined: the game streams
  replays from Supercell's servers in its own encrypted format, so decoding them is the same line as reading memory
  or the network.
- **Cole checked: replays show the opponent's hand.** The pipeline reads plays from hand changes, not the
  "<Card> lvl.N" label (that was only for the spot), so a recorded replay gives exact opponent plays. It may even
  give both sides from one file. Plan: Cole plays normal ladder games, then records the replays. Top players'
  battles work the same way, which gives pro games with full truth for step A as well.
- **`scripts/replay_runner.py`** (teach / run): Cole demonstrates one cycle in BlueStacks (Replay → end screen →
  back + scroll to the next battle). The runner repeats it and records each replay to
  `data/raw/replays/<date>/replay_NN.mp4`. It records the whole display cropped to the window, which avoids the
  black window captures. Replay end = the screen patch at the exit click matches the taught end screen.
  - It only clicks through menus. It never plays a battle and never reads the game. Run on the alt.
  - Still needed: one recorded replay to build the replay layout (both hands' positions) and the hand reader for
    the top side.
- **Replays show both hands** (first test recording, 2026-10-12): the opponent's 4 cards, Next and elixir bar at the
  top, ours at the bottom. One replay recording = play truth for both sides, no pairing. The Battle Log entry shows
  both decks. Replays expire: on Cole's alt, 21 of the log's 25 battles still had Watch, and those from ~6 days back
  said "Replay no longer available".
- **`scripts/replay_runner.py` rewritten to use BlueStacks' ADB** (Cole turned ADB on in BlueStacks settings), so
  Cole can keep using the Mac. Taps are `adb shell input`, screenshots `adb exec-out screencap`, recording is
  scrcpy (`brew install scrcpy`, ADB=BlueStacks' hd-adb), 1080×1920 at 30 fps, no 3-minute limit.
  - Loop: Battle Log → the entry after the last one recorded (found again by a fingerprint of both decks; slow
    250 px swipes, since a 600 px swipe moved the list ~820 px and lost entries under the header) → save the entry
    crop (decks) → record → Watch → end screen (VS + OK templates) → stop → OK, which returns to the Battle Log at
    the top.
  - Walk test: all 21 watchable battles found in ~110 s, stopping at the expired ones.
  - Screen templates are game frames, kept in `data/raw/replays/_ui/` (gitignored).
  - End-to-end test: 2 replays recorded back to back (209 s, 184 s), both end screens seen, entry crops were the right
    battles (battles 2 and 3). The first ADB test recording (battle 1) is kept as `replay_00.mp4`. Each file starts
    at the Battle Log, then the VS intro, the full game with both hands, and the end screen.
  - Not usable by the pipeline yet: it needs a replay layout and the hand reader pointed at the top (opponent) hand.
- **How much replay data (Cole asked what gets us to ~95%, 2026-10-12).** A rough plan, not a fitted curve: we
  have no scaling measurement yet (every retrain so far reused the same ~22 dev games).
  - Where the misses are: of the dev misses, ~257 plays produce no event at all and ~60 get the wrong card. Never
    caught: spells with no deploy clock, hero abilities, Mirror, units hidden by towers, and rare cards in red.
    Data fixes the rare-card and red-unit misses. Spells, abilities and Mirror also need logic (deck + cycle).
  - Rough targets, assuming ~35 opponent plays per game and ladder opponents bring varied decks:
    ~100 replays (~6 h of games) for V5 and a first scaling point; ~300 (~18 h) for most cards to have real red
    examples; ~800–1,000 (~50–60 h) to give long-tail cards a few hundred plays each.
  - Measure before committing: train V5 on ~100, rescore dev and benchmark, and fit the gain per 100 games.
  - 95% "fully right" is probably not reachable from detector data alone; the last points need deck/cycle logic.
- **First full batch (2026-10-12, 14:00–15:18):** `replay_runner.py -n 25 --skip 3` recorded replays 04–19
  unattended, then stopped at the expired entries ("no more battles in the Battle Log"). With 00–03: **20 replays,
  ~72 min of footage, 3.4 GB, every end screen seen**, entry crops all distinct and matching their battles
  (spot-checked 03, 10, 18, 19). Opponents ~7,600 trophies, all different decks.
- **TV Royale mode** (`replay_runner.py --source tv`, Cole's idea, 2026-10-12): TV Royale lists top players' games
  (Ultimate Champion, Ranked: top-100 players) with both decks and a Watch button, the same as the Battle Log, and its
  replays show both hands. So each one is a **top-ladder game with exact plays for both sides**: truth for the
  detector and real data for the move predictor (A). Cole: this is high ladder, not pro (tournament) play; it is the
  best gameplay available outside pro footage. Recordings go to `data/raw/tv_royale/<date>/`. The runner records the
  tab that is open (it never leaves TV Royale; OK on the end screen returns to it).
  - Re-runs skip battles recorded on an earlier day: the deck part of each saved entry crop is compared with the
    screen (≥ 0.98). Different Battle Log battles reach 0.967 (our deck is in all of them); friendly rematches
    with identical decks can't be told apart and get skipped.
  - Test: 2 replays (302 s, 211 s), both end screens seen, the right entries. The list changes while the runner
    works (new games appear at the top); those are picked up by the next run.
  - TV games are in different arenas (themes), so the replay layout must not depend on the arena art.
- **`scripts/tv_loop.sh`** (unattended TV recording while Cole is away): waits for a run in progress, then re-runs the
  TV runner every 30 min for 5 h; already-recorded games are skipped. Started under `caffeinate -i -s`: the Mac's idle
  sleep is 1 min, which would kill a run. ~170 MB per replay.
- **TV leagues (Cole: "if we run out of high ladder, go down the ladder"):** each TV run first goes to Ultimate
  Champion (right arrow until its tab title shows), records it, then taps the left arrow for each league down
  (`--max-tabs`, default 6). The league changed if the tab title band changed. Each log row has `tab`
  (0 = Ultimate Champion); the move predictor (A) keeps to tab 0, the detector uses all of them. Not yet seen
  live: the first switch happens when the running loop's current tab runs out.
- **Fix: TV Royale loads slowly after OK** (a spinner over the end screen for a minute or more). The runner quit on
  the unknown screen after replay 9. In TV mode it now waits out unknown screens (loading, or a replay still playing
  when a run starts) for up to 10 min, taps OK at most every 15 s, and never presses Back (Back offers "Return to the
  main menu?", which leaves TV Royale).
- **Replay reader** (2026-10-12): a replay recording is processed twice, `<id>` with layout `replay_bottom` and
  `<id>_top` with `replay_top` (same file, two links in `data/raw/replay_links/`); the top player's plays become the
  bottom player's opponent truth, `data/ground_truth/opponent_truth_replay_<id>_m01.json`
  (`bot77.pairing.write_replay_truth`, offset 0). Separate prefix so the 22-match paired dev set
  (`opponent_truth_auto_*`) stays fixed and comparable with V4. `scripts/ingest_replays.py` does all new replays
  (IDs `bl_<date>_rNN` Battle Log, `tv_<date>_rNN` TV Royale); bottom copies also get V4 detections
  (v5l/epoch15.pt → `unit_detections_v5l_e15`; `run_all.py` job fields `weights`/`table`, `gpu: false` for the top copy).
  - Layouts, hand-measured on replay_01: both hands at 0.702 of Alina's hand (card centres x 210/352/494/637),
    timer box top right (y 258–346), x2/x3 drop below it. Arena from the four tower HP bars:
    x = 92.9 + 0.829·x_alina, y = 196.9 + 0.8014·y_alina (slightly anisotropic: KataCR tile y ~3% off).
  - Next card is drawn larger relative to the hand: per-layout `next_art_scales` 0.35–0.39 (live: 0.31–0.35).
  - BlueStacks' framebuffer (screencap/scrcpy) is darker than what the Mac shows: whites cap near 178, pink
    elixir is ~(B151, G27, R148). Not Android brightness (102/255; setting 255 changed nothing, put back).
    `_is_pink` also accepts dark saturated pink. Live captures still match the original rule. The live bot (B)
    will see this same framebuffer, so training on these frames matches its runtime.
  - Elixir: the bar starts under the number circle (x 70). The top bar's empty part is dark red and both bars'
    partial segment is grey purple, so on replay layouts (`elixir_empty: "any"`) anything not pink or white is
    empty/partial. Result: smooth 0.36/s fill and clean drops on both bars.
  - **Plays by swap:** in the replay viewer a played card never leaves an empty slot; it is highlighted, then
    replaced directly by the next card. `detect_plays(swap_plays=True)` (layout `plays_by_swap`) adds a slot switch
    near the drop as a candidate, and "replaced by the previous next card" as a hand-only play; Next is taken 0.5 s
    before the swap (it advances in the same frame). POV layouts are unchanged.
  - Test (TV replay_04, top-100 Hog EQ vs Mortar/Goblinstein): bottom 86 events (68 high, 4 unnamed: mostly
    Mighty Miner abilities), top 49 (43 high, 4 unnamed); **0 cycle violations** on either side (77 and 45 named
    plays). Four top plays checked by eye (Cannon Cart, Mortar, Fireball, Goblinstein): all right.
  - To do: V4 detections for tv_2026-10-12_r04 (processed by hand before the script) and the self-play pair
    cole_2026-10-11_a/b (their GPU stage failed on 2026-10-11 with the full disk).
- **End of the 2026-10-12 TV session** (paused at Cole's request after game 39): 39 TV replays (24 Ultimate Champion,
  15 from one league down; the left-arrow league switch worked live) + 20 Battle Log = **59 replays today**.
  Locking the Mac's screen doesn't affect recording (ADB/scrcpy record inside Android): games after the lock have
  normal frames. The ingest run covers the 35 replays that existed when it started; TV replays 16–39 need another
  `scripts/ingest_replays.py` run.
- **TV runner starts from the home screen** (19:10): TV mode now opens TV Royale itself (home menu → third item,
  TV Royale, which reopens on the last tab used; the runner then goes right to Ultimate Champion). First run
  after the break started from home and recorded a new Ultimate Champion game (#157 vs #103, 9 min old).
- **Cole's review of the replay play reader** (bl_2026-10-12_r03, his side): all right except one Barbarian Barrel
  counted twice (an unnamed drop + a hand-only play at 69.4 s). Cause: that slot *did* show empty for 0.3 s and
  the drop ended after the refill, so neither "vacated" nor the direct swap matched. Fixes, swap mode only:
  - `_swapped` accepts card → empty frames → card.
  - `_went_unreadable`: a slot that turns "unknown" at a drop while Next advances gives up its previous card
    (TV r04 top 59.9 s: Mother Witch, then an unreadable Cannon Cart).
  - Effect on the reviewed games: TV r04 bottom 86 → 81 events (5 double counts gone, 0 unnamed); top pages
    7 → 2 unnamed drops; 0 cycle violations (Mighty Miner ability uses aside).
  - All replays redetected with the fixes and their truth rewritten (`data/logs/replay_redo.log`).
- **Recorder glitch** (19:47): scrcpy didn't start for TV replay 47, so the game played unrecorded and the run
  crashed remuxing a missing file; the loop then idled 30 min (Cole: "is it stuck?"). The runner now retries the
  recorder up to 3 times before tapping Watch, and if a recording is missing at the end it skips that game (entry
  crop removed so a later run retries it) instead of crashing.
- **Duplicate check tightened** (Cole: "make sure we're not recording games we already recorded"): a pairwise check of
  all 47 TV entries found TV 44 = TV 30 (same game re-listed, now marked "Watched 1-0"). A watched entry shows its
  result and score, which pushes the decks 8 px down; the thumbnail compare had no slack and scored 0.80. Now:
  quarter-size blurred thumbnails compared with ±24 px vertical slack, both ways round; duplicates 0.97 (both
  known cases), different games ≤ 0.82 (TV and Battle Log), bar 0.90. TV 44 moved out of the dataset (scratchpad).
- **Leagues stop early** (Cole: don't scroll through a league we already have): TV lists are newest first, so once
  `KNOWN_RUN` = 4 already-recorded entries come in a row, the rest of that league is older and recorded: the runner
  moves to the next league at once instead of scrolling to the bottom. 4 rather than 1 because a few re-listed
  games can sit near the top. Takes effect from the loop's next run.
- **Replay top views lost their HUD reads** (found after the re-detect: TV 09 top gave 0 plays). Storing a video's
  HUD rows first deleted `match_id LIKE '<video_id>%'`, which also matched `<video_id>_top_m01`. Whenever the bottom
  view finished processing after the top view, the top's HUD rows were wiped (7 TV replays: 09, 16–19, 33, 42; their
  events survived, except 09's). The delete is now scoped to `<video_id>_m%`; those 7 top views were reprocessed and
  the truth rewritten. The pre-replay videos were unaffected (no ID is a prefix of another).
- **Replay test set (V4 = v5l e15, clock rules, score-only)**: 67 replays, 2,580 opponent plays read from the top
  hand (`scripts/eval_dev.py ... --set replay`; none of these games were in V4's training data). **48.2% fully
  right**, 17.6% wrong card, 34.2% missed, 9.6 fakes/match. Battle Log 55.3% (20 games), TV Royale 45.6% (47 games,
  top-ladder decks with more cards V4 rarely saw). In line with dev (42.3%) and benchmark (49/111), so the dev set
  wasn't flattering V4. Caveat: the truth is the hand reader's, spot-checked on 3 review pages, not fully audited.
  Logs: `data/logs/eval_replay_v4.log`, `data/logs/evaldev_rp_e15.json`.
- **Overnight recording** (Cole: "keep recording till the morning"): loop restarted at 22:40 as `tv_loop.sh 11 1800`
  (until ~09:40), on AC power. The dedupe spans every date folder, so games after midnight go to
  `data/raw/tv_royale/2026-10-13/` (numbering from 1) without re-recording earlier ones. ~180 MB per game; 112 GB free.
- **Overnight result (2026-10-13)**: recorded TV 85–108 (24 games) until 00:38, then stuck 8 h. TV 108 was an
  overtime game whose end screen had units fighting over "VS" (0.74 < 0.80), so the runner never saw the end, timed
  out at 600 s and stopped; every later run found an unknown screen and quit. Fix: `at_end` accepts the OK button
  alone at ≥ 0.97 (it scored 0.99), and a missed end no longer stops the run. TV 108 is kept (complete game plus a
  tail on the end screen). Totals: 107 TV + 20 Battle Log = 127 replays.
- **Processing keeps up with recording** (Cole: "continue to process until everything we've recorded is processed"):
  `scripts/ingest_loop.sh` waits for the running ingest, then re-runs `ingest_replays.py` every 10 min (only new
  recordings), and exits once the TV loop has stopped and a pass finds nothing new. Ingest now skips mp4s younger
  than 2 min (the recorder may still be remuxing). Log: `data/logs/ingest_loop.log`.
- **Replay test set, 127 games (2026-10-13)**: V4 e15 48.4% fully right on 4,942 plays (16.6% wrong card, 34.5%
  missed, 9.0 fakes/game); Battle Log 55.3%, TV 47.3%. The 60 new TV games alone: 48.6%. Stable vs the 67-game
  score (48.2%), so the estimate is settled; the next number to move it is V5.
- **All leagues** (Cole: "we still have Master 1 plus the other 26 arenas"): the 2026-10-13 run stopped at the 6-league
  cap after 123 games (250 replays total). `tv_loop.sh` takes a third argument, max leagues; restarted 19:05 as
  `tv_loop.sh 12 1800 60`, so each run walks down through Master 1 and every arena until the arrow stops changing
  the league. Lower-arena games are for the detector (more cards, arenas, levels); the move predictor (A) keeps to
  tab 0. Each runner_log row records its league as `tab`.
- **Recorder start retries widened** (2026-10-13 20:07): replay 140's recorder failed 3 tries 2.5 s apart and the run
  ended (the loop retries 30 min later); started by hand right after, it worked. Now 5 tries with 0/5/10/20/30 s
  waits, and scrcpy's stderr goes to `data/logs/scrcpy_last.log` so the next failure shows its cause.
- **Replay test set** (`config/replay_test_games.json`): 30 replays held out of all V5+ training, picked by game
  (seed 0) across leagues: Battle Log 4, Ultimate Champion 8, then 4/4/3/3/2/2 down the leagues. V4 and V5 are
  compared on these. `autolabel_clock_spots.py --truth replay` labels the other replays (the test set, benchmark
  and V5 check games are never labelled). Trial: 90 plays -> `data/external/real_v5r_trial`, review page for Cole.

## 2026-10-14 · Replay labels named the wrong play: clock window fixed

Cole's review of the first replay label trial (51 judged): 45 wrong, 3 right, 3 off. The boxes sat on real units but the
names were wrong. Cause: `autolabel_clock_spots.py` looked for the enemy deploy clock 0.2-2.5 s after the truth's play
time, which is right for the dev pairs (clock ~1 s after the play) but not for replays. On replays the top-view hand read
and the clock coincide (1,062 plays from 40 replays: median +0.05 s, almost all within ±0.2 s). So the right clock fell
outside the window, and the next play's clock (e.g. Wizard 2.5 s after Giant) fell inside it and lent its unit the
earlier card's name. It also explains the 2,396 "no clock" skips. The window now depends on the truth set: dev
(0.2, 2.5), replay (-0.3, 0.4). Rerun of the same 90-play trial: a clock found for 75% of plays (was 13%), and 1,103 of
those were already named correctly by V4, which suggests the pairing is now right. Round 2 review:
data/reviews/v5_replay_labels2.html (60 frames).

## 2026-10-14 · Replay labels, round 3 method: reference check, V4-confirmed boxes, tracking

Round 2 (60 frames, clock window fixed): 21 right, 7 off, 32 wrong. Wrong ones were mostly buildings, spirits, hidden
units, and the nearest badge belonging to another unit; 50 of 60 frames had no approved slice to check against.
Changes in `autolabel_clock_spots.py` for replays:
- Reference check: `scripts/replay_refs.py` took 2,890 crops (116 entities, max 40 each) from plays where V4 already
  boxed exactly the truth's entity at the clock (conf >= 0.7), from non-holdout replays. A label is kept only if, among
  those references (own match left out), its nearest entity is the named one and similarity >= 0.6. Same-arena crops
  lift every similarity, so "nearest" matters more than the value. On round 2's verdicts: kept ~10/21 right, 1/32 wrong.
- Buildings and spirits skipped for replays.
- Plays V4 already named are labelled from V4's own box (still checked); every badge near the clock is tried, not just
  the nearest; plays 0.8 s apart count as alone (was 2 s); each labelled unit is followed 5 s, a frame a second,
  re-checked every frame.
Trial (120 plays): 46% of attempted plays labelled, 3.3 frames per play (396 frames: 157 V4-confirmed, 196 tracked,
43 badge+SAM). Nearly all labelled plays were ones V4 already named; the new information is in the tracked frames
where V4 misnames the unit later. Full run estimate: ~5,600 eligible plays in 379 non-holdout replays, so ~2,600
plays and ~8,500 frames. Round 3 review: data/reviews/v5_replay_labels3.html, 20 of each kind.

## 2026-10-14 · Play logic on replays: own-deck rule, replay timing, clock-weighted deck (fixes 1 and 5)

Why plays are missed (V4, 127 scored replays): 1,429 of 4,942 opponent plays had no event. Spells 35% of misses,
single troops 40%, groups 11%, hero abilities 9%, buildings 5%. Of 504 missed spells, 353 were detected (conf >= 0.5)
within 3 s, so most are logic losses, not detector losses; 132 were never seen (Zap 50, Arrows 35).
Stage trace (60 non-test replays, 610 missed of 2,099): 272 lost in apply_rules, 159 dropped as cardless after the deck
filter, the rest never became a candidate. Inside apply_rules: 95 deleted by the own-deck rule (card also in the
bottom player's deck, y > 11; replay players share meta cards), 172 kept but timed 2-4 s off.
Experiments on 140 non-test replays (70 to calibrate timing, 70 to tune; scratch harness), fully right / wrong / missed /
fakes per match:
- current logic 48.7 / 17.5 / 33.7 / 8.0
- own-deck rule only for cards we played in the last 10 s: 50.4
- replay timing (clock offset 0.05 s instead of 1.0, per-card table): +0.9; together 51.3
- deck counts unclocked deploys at 0.6: 52.6 / 14.7 / 32.7 / 6.1 (kept)
- true deck (oracle): 56.7 with the deck filter, 61-63 with confusion-based renaming of cardless plays. Our inferred deck
  is 65-70% precise / 69-74% recall; a known-decks kNN prior (310 decks) gave ~+1 and was not kept, nor was renaming
  (more right, but wrong +7 and fakes +3.5/match).
Code: `playrules.replay_rules` (applied by `bot77.opponent` for replay layouts): config/play_timing_replay.json,
own_deck_recent_s 10; `infer_deck(..., w_unclocked)` with rules `deck_w_unclocked` 0.6 everywhere.
Held-out check (never tuned on): replay test set (30 games) 45.9% -> 48.1% fully right (wrong 212 -> 219, fakes
9.1 -> 9.5/match); dev (POV recordings, own-deck and timing unchanged, deck weighting only) 46.9% -> 47.3%, fakes
10.2 -> 9.8. Smaller than on the tuning set. The ceiling says knowing the deck is worth ~+8-10 points; the inferred deck
is limited by the detector's naming, so V5 is what moves it.

## 2026-10-14 · Replay truth: unknown plays and a cycle guard

Cole's review of tv_2026-10-12_r04 top (49 plays, page built 10-12): 46 right; 3 marked as unknown plays he could name.
Two (Mother Witch 59.9 s, evo Minion Horde 275.7 s) were already fixed by the 10-13 unreadable-refill rule. The third,
Mother Witch at 214.7 s 0.8 s after Guards, was still unknown: in swap mode the slot's earlier switch (Mother Witch
swapping in at 212.0, already attributed) hid the later unreadable refill, and the refill scan only tried the first
unreadable frame (the whole hand flickered unreadable at 214.0). Across 410 replays 1,053 of 28,494 hand plays (3.7%)
were unknown; unknown plays are left out of the truth, so they get no labels and a detector hit on one scores as a fake.
Changes (`bot77.events`): candidates fall through an already-attributed switch to the unreadable-refill check; every
unreadable run in the window is tried; an unexplained drop is named by the card gone from the settled hand (1-3 s
after) when its cost is within 0.8. Checked on stored HUD states with the cycle rule (a card can't be played again
within 4 plays, counted 3 either side): unguarded, unknowns 1,061 -> 905 but impossible repeats 562 -> 636. So names
from the weak cues (unreadable refill, settled hand), including the 10-13 rule, are dropped back to unknown when they
break the cycle: unknowns 1,360, impossible repeats 179. That trades ~220 names for ~380 fewer wrong ones; Cole's three
cases keep their names. Truth rewritten from the redetected events (backup of the previous truth kept locally).

## 2026-10-14 · V5 replay labels: final set and spot check

Full run (`autolabel_clock_spots.py --truth replay`, V4 boxes): 8,563 frames from 2,788 plays (v4-confirmed 3,601,
track+sam 3,165, clock+sam 1,039, track 758). After the truth fix: 159 frames from 51 plays whose truth was dropped or
renamed set aside (`real_v5r/dropped/`), 96 frames from 34 newly named plays added (`--only-new`, `real_v5r_new`).
Bundle (`scripts/build_train_v5.py`): 7,478 train / 1,018 val, split by match (47 val matches by hash); benchmark,
v5 check games and the 30 replay test games asserted absent. Cole's spot check (`data/reviews/v5_spotcheck.html`,
60 random frames across all four methods + 10 from new plays): 58 right, 2 right unit with a loose box, 0 wrong.
- **V5 launched 2026-10-14 ~15:50 PDT** (`scripts/katacr/pod_v5.sh`, run `v6l` on the pod): fresh RTX 4090 Secure pod
  (bot77-train-v5, $0.89/h, US-NC-1), synth_v3 regenerated from the kit (48,004 frames, ~10 min on 11 processes).
  From V4 (v5l epoch 15), 15 epochs: 3,909 real_v3 x3 + 4,580 real_v4 + 7,478 real_v5r + 48,004 synth_v3; val adds
  real5_val. GPU 83-95% busy with 8 loader workers (cache=ram would need ~100 GB), ~17 min/epoch, ~4.5 h, ~$4.50.
  Watcher `watch_run.sh … v6l` scores each 5-epoch checkpoint on the benchmark.
- V5 run finished 19:43 (TRAIN_EXIT 0, 15 epochs). Val mAP50 / mAP50-95 peaked at epoch 11 (0.742 / 0.434, = best.pt),
  epoch 15 0.733 / 0.423. Weights checksummed against the pod, then pod `ygcpxpqac3xccw` terminated (~4 h 20 min, ~$3.90).
  Benchmark fully right during training (old play logic, 111 plays): V4 46; V5 e0 37, e5 40, e10 43 (fakes 60 / 63 / 57
  vs V4 62). Scoring best and last vs V4 on dev, the 30 replay test games and the benchmark with the current play logic.

## 2026-10-14 · Mirror rule: clock + deck room, drop impossible repeats

Why V5 still fails (replay test set, V5 best): 409 missed (spells 167/225 missed vs troops 186/807), 212 wrong card,
239 fakes. Biggest single logic bug: the Mirror rule (any repeat within 2 plays -> Mirror) fired 74 times on the 30 test
games, which have no Mirror plays, and 276 times on 60 tuning replays; on dev, 73 calls with 9 right (34 real Mirrors).
A repeat in OUR play sequence is mostly the same unit seen twice or a real play in between that we missed.
Caveat: replay truth never says Mirror (the hand reader has no Mirror icon); its 33 back-to-back repeats are likely
Mirrors labelled as the copied card.
New rule (Cole: Mirror copies the last card played; check the deck can hold Mirror): a repeat is Mirror only if it has
a deploy clock (a new placement) and the deck has room (fewer than 8 other cards with >= 2 weighted deploys); a repeat
that fails either test is dropped as a duplicate (the cycle forbids it). Sweep, fully right / wrong / fakes per match
(dev with V5 dets; 60 non-test replays with V4 dets):
- old rule: dev 49.8 / 58 / 8.0, replays 51.8 / 360 / 9.0
- rule off: 49.5 / 70 / 8.8, 54.5 / 339 / 10.0
- clock + deck room: 50.4 / 62 / 8.4, 53.7 / 340 / 9.6
- clock + deck room + drop the rest (kept): 49.9 / 45 / 7.0, 53.2 / 293 / 7.2
Held-out after the change: replay test V5 best 47.0 -> 46.5 fully right, wrong 212 -> 191, fakes 8.0 -> 6.2/match;
V4 47.9 -> 49.0, wrong 217 -> 160, fakes 9.8 -> 7.8. Benchmark V4 51 / 14 / 22, V5 49 / 7 / 25. Dev V4 47.3 -> 46.5,
fakes 9.8 -> 8.4. Mostly a precision change: about the same right, far fewer wrong and fake.
Spells (replay test, V5): only lingering ones are named (Poison 14/30, Rage, Earthquake). Zap 0/27 and Arrows 0/16 are
never detected in any frame; Log 21 of 50 never seen, Barbarian Barrel 30 of 45 seen in only 1-2 frames. Detection
runs at 4 fps (DETECT_FPS), so a ~0.3 s Zap usually falls between frames. KataCR (Wu et al. 2025) reports only
per-frame mAP over all classes (no per-spell or per-play results) and its agent never names opponent plays, so it has
no answer here; its one relevant finding is that splitting classes across 2-3 detectors by object size raised
small-object mAP (39.8 -> 48.3).
V4 vs V5 with the new Mirror rule, fully right / wrong / fakes per match:
- dev: V4 46.5 / 66 / 8.4; V5 best (e11) 49.9 / 45 / 7.0; V5 last (e15) 49.2 / 51 / 7.2
- replay test (30): V4 49.0 / 160 / 7.8; V5 best 46.5 / 191 / 6.2; V5 last 43.4 / 205 / 6.4
- benchmark (111 plays): V4 51 / 14 / 22 fakes; V5 best 49 / 7 / 25
V5 best wins dev (+3.4) and cuts wrong/fake cards, but loses 2.5 on the held-out replays; e15 is worse than e11
everywhere. No clear winner: keep V4 as the default for replays, V5 best for POV recordings, pending the spell stage.

## 2026-10-14 · Spell stage (overnight build, Cole's go; RunPod budget $10)

15 fps check (V5 best over [t-0.5, t+3] around every instant-spell play, 30 replay test games), plays seen in any frame
at 4 fps-equivalent -> 15 fps: Log 29 -> 39 of 50, Barbarian Barrel 37 -> 41 of 45, Fireball 12 -> 16 of 19, Goblin Barrel
4 -> 5 of 18, Zap 0 -> 0 of 27, Arrows 0 -> 1 of 16. The Zap flash is visible by eye for ~3 frames at 15 fps; the
detector simply never learned it (6 real Zap labels). So: a separate spell stage, not a denser detector.
Design (Cole: combine the effect, the clip and the play logic): a clip classifier (R(2+1)D-18, Kinetics weights) over
1.6 s of the arena at 10 fps, a window every 0.3 s, acting as its own trigger; peaks above a threshold become spell
plays; our own spells (POV hand log) are removed; detector spell plays nearby are replaced and barrel troops dropped;
deck inference then sees spells and troops together (`bot77.spells`, rules `spell_probs`). Training clips: every spell
play of BOTH players in the non-test replays (top from replay truth, bottom from our hand log) plus "no spell" clips
at troop plays and quiet moments; val = 10% of matches by hash (`scripts/spell_clips.py`). Goal set before building:
replay-test spells right 10% -> 50-60%, fully right 46.5% -> ~53-55%, dev 49.9% -> ~53-56%, fakes +0.5/game at most.
Built and run 2026-10-14/15 (pod `9jsrjfg7eig10p`, RTX 4090, 1 h 18 min, ~$1.20, terminated after pulling weights and
probabilities; data/models/spell_v1/). Clips: 12,597 (4,901 spell, 7,696 no-spell) from 350 non-test replays.
Classifier, held-back clips (10% of matches): 91% of spells named right, 0.5% of no-spell clips called a spell; Zap 94%,
Arrows 98%, Log 95%, Fireball 99%, Barbarian Barrel 93%; Royal Delivery 0/28 and Rage 2/4 weak.
Threshold tuned on the 48 held-back replays only (V4 dets; troops only 53.8% / 6.3 fakes): 0.3 63.1% / 7.9, 0.5 63.2 /
7.3, 0.7 63.4 / 6.5 (kept), 0.85 62.7 / 6.4, 0.95 62.5 / 6.2. Spell-stage calls there: 289 of 342 exactly right.
Held-out results, fully right / wrong / fakes per match (troops only -> troops + spells):
- replay test (30): V5 best 46.5 / 191 / 6.2 -> 59.8 / 141 / 6.5; V4 49.0 / 160 / 7.8 -> 59.8 / 123 / 7.9
- spells named right on replay test (V5): 22 -> 185 of 225 (Log 0 -> 41/50, Barbarian Barrel 0 -> 42/45, Zap 0 -> 24/27,
  Fireball 0 -> 17/19, Goblin Barrel 0 -> 17/18, Arrows 0 -> 9/16, Poison 14 -> 25/30); Spirit Empress 3 -> 0 and
  Earthquake 2 -> 0 (the detector's calls get replaced by a nearby classifier call; to fix)
- dev (POV recordings): V5 49.9 / 45 / 7.0 -> 59.5 / 40 / 9.2; V4 46.5 / 66 / 8.4 -> 56.6 / 57 / 10.3
- benchmark (111): V5 49 / 7 / 25 -> 59 / 7 / 31; V4 51 / 14 / 22 -> 59 / 13 / 27
Beat the goal on replays (59.8% vs 53-55%) and dev (59.5% vs 53-56%), but dev fakes rose +2.2/match (budget +0.5): 32 of
37 pure spell fakes there are Vines / Goblin Curse / Void (31-114 training clips, Cole's newer-card games), and 33 calls
sit on our own plays under another name (our Mirror'd Barbarian Barrel, our Vines as Poison). Tried: subtract any of
our spells at that moment (dev +0.2, held-back replays -1.0; kept off, `spell_any_own`), higher thresholds for the three
new spells (right and fakes trade ~1:1; not kept). Next: POV-recording clips and more Vines/Curse/Void for training.
With spells in, V4 and V5 tie on replays (59.8 both; V5 fewer fakes, V4 fewer wrong); V5 wins dev by 2.9. V5 best is
the default detector from here.

## 2026-10-15 · Back-to-back plays, unread replay cards, abilities (Cole: "focus on fast sequential plays")

Cole's example (playback video, tv_2026-10-12_r46 at 0:08.9): we said Bandit, the answer key said Wall Breakers. Both
were played ~0.3 s apart (opponent elixir 8.2 -> 4.4, Next advanced twice); the detector saw both units in one deploy
and kept one. The answer key had only Wall Breakers: the replay viewer draws Bandit with new art, the hand reader never
read it in any of 400+ replays, so it was missing from every answer key.
Measured before changing anything (V5 + spell stage, fully right): plays <= 1 s from another 58% (test, n=130) / 58%
(dev, n=124) vs plays alone 63% / 65%; champion/hero abilities 0% right (43 test, 29 dev; 81% / 93% missed).
Decision (Cole agreed): no extra "bunch" or ability model. Fix the answer keys, add ability classes to the spell-stage
clip classifier (same model, no added latency), then let one deploy split into two plays only with evidence (150 of
1,035 test events hold 2+ unit classes, mostly lookalike noise: a blind split would add fakes).
Answer-key fixes:
- 8-card cycle rule: a played card shows as Next 3 advances later (91% of 12,047 logged replay plays agree; 0.9% at 4
  advances, 0.4% at 5). `bot77.pairing.next_card_plays` uses it to (a)
  add a play at a Next advance with no logged play near it, (b) name logged plays the hand couldn't name (568 replay top
  plays had no card, mostly names dropped by the cycle guard), only where the neighbouring logged plays agree with the
  rule. `scripts/add_next_card_plays.py` appends these to the existing paired / replay keys (tagged "next-card
  advance"): 167 plays over 438 keys.
- 150 of 410 replay keys had fewer than 8 different cards. Contact sheets of the unread slots: Mirror (all frames),
  Lightning (reads as Rage 0.66 vs Lightning 0.56, under the 0.2 margin), Bandit (new art), plus Witch (new art?), Mega
  Knight in replays, some champions/heroes with glowing frames (less clear; for Cole to check). The replay viewer also
  draws cosmetic card frames (gold, pink); the art patch mostly avoids them.
- In-game art overrides from replay frames (`scripts/hand_art_override.py cut/check`, box from a readable normal card
  in the same frame): Mirror 0.94 vs Rage 0.57-0.66, Lightning 0.83-0.86 vs 0.42-0.44 (gold and grey frames), Bandit
  0.90-0.92 vs 0.32. The r46 top hand re-read: deck now 8 cards, Bandit and Wall Breakers both logged at 11.9 s.
- Re-reading the 142 replays whose key had < 8 cards (`bot77 process`, replay_top, ~4.5 min each, 6 at a time).
Abilities: `scripts/spell_clips.py` adds "<card> ability" positives from both hands' ability events (1,071 in replay
logs: Valkyrie hero 119, Archer Queen 91, Golden Knight 93, Bowler hero 75, ...); `bot77.spells.merge_spells` turns
them into "__ability" plays, kept only if the opponent played that card earlier and we didn't use the same ability
then (rule `spell_ability_thr` 0.7); opponent.py keeps them out of deck inference.

### 2026-10-15 (afternoon) · Answer keys corrected, abilities in the spell stage, storage
**Answer keys.** Re-read the top hand of the 142 replays whose key had < 8 distinct cards (art overrides for
Bandit, Lightning, Mirror; 2 workers, ~4 h), regenerated the replay keys and re-ran `add_next_card_plays.py` (214
plays added from the 8-card cycle rule; Mirror 68 of them, then Skeletons, Ice Spirit). Keys with < 8 cards:
150 -> 81 of 410. Same model (V5 + spell_v1) rescored, fully right / wrong / fakes per match:
- replay test: 59.8 / 141 / 6.5 -> 60.2 / 130 / 5.9 (1,171 -> 1,207 true plays): part of "wrong" and "fake" was key error
- dev: 59.5 / 40 / 9.2 -> 56.1 / 42 / 9.0 (693 -> 738 true plays): the added plays, mostly Mirror, are new misses
**Mirror** is now the largest group of newly exposed misses: the copy is detected as the mirrored card. Next: a rule.

**Spell stage v2 = abilities** (`data/models/spell_v2/`, pod `eqzl39dbhei3y9`, RTX 4090, ~70 min, ~$1.05, terminated).
Clips rebuilt with "<card> ability" positives from both hands: 13,489 clips; 15 ability classes cleared the 25-clip
minimum (Archer Queen, Golden Knight, Boss Bandit, Mighty Miner, Skeleton King, Little Prince, Monk, Goblinstein and
the hero forms Knight, Valkyrie, Bowler, Balloon, Mega Minion, Mini P.E.K.K.A, Wizard). Held-back clips: spells 88%
named right (v1 91%), no-spell clips called a spell 1.2%; abilities 44-92% caught at 78-100% precision.
Tuned on the 48 held-back replays (V4 dets, as for v1): ability cutoff 0.7 (0.5 identical, 0.85/0.95 lose catches).
Two fixes found by looking at the new fakes:
- our own Lightning: the replay hand reader can't name it, so our unnamed play wasn't subtracted -> `spell_unnamed_own`
  (an unnamed play of ours in the own window explains a spell call): held-back fakes -21, right -3;
- Barbarian Barrel over-called (after Royal Delivery, near Fire Spirit / Wall Breakers) -> class cutoff 0.97;
- on dev only, Void fired 10-20x per POV recording (17 -> 154 fakes; replays fine: a POV-vs-replay look gap) -> Void
  cutoff 0.995 (dev 0.9: 56.6 / 258 fakes; 0.97: 56.0 / 223; 0.995: 56.1 / 211).
These are now the defaults in `playrules` (`spell_unnamed_own` True, `spell_class_thr` BB 0.97 / Void 0.995).
Held-out results, spell_v1 -> spell_v2 + fixes (corrected keys), fully right / wrong / fakes:
- replay test (30): 60.2% / 130 / 5.9 -> 60.6% / 140 / 6.6; abilities right 0 -> 22 of 41 (17 ability fakes)
- dev (26): 56.1% / 42 / 9.0 -> 56.1% / 44 / 8.1; abilities right 0 -> 9 of 29
- benchmark (116 plays): 60 / 7 / 30 -> 65 / 5 / 28
- held-back replays (V4 dets): 63.1% / 207 / 6.2 -> 64.2% / 209 / 6.6
Abilities work; the overall gain is small because the new model also adds spell fakes on replays (+0.7/match, over
the +0.5 budget). Next for this stage: POV-recording clips in training (the Void gap), then the Mirror rule.

**Storage.** Disk was 94% full. Deleted regenerable upload copies (mine_v4 17 GB, real_v4 bundle/tgz, real_v5r_bundle,
train_v2.tar, old spell clips, unpacked spell clips/streams), compacted `events` (5.6 -> 3.0 GB) and `hud_states`
(238 -> 81 MB; thousands of versions -> 1). Moved 532 cold videos (161 GB: everything except replay test, V5 check,
benchmark and dev games) to Cole's LaCie drive (`/Volumes/LaCie/bot-77-data/raw/`, same layout) with symlinks at the
old paths, so the `videos` table is unchanged; those videos need the drive plugged in. Free space 56 -> ~230 GB.
`unit_crops` (1.89M crops, 73 GB, 34,667 versions) not compacted yet: compaction needs room to rewrite it.
Cloud (Cole): only when one of these happens — the same data re-uploaded to pods often, part B needs a server,
an off-site backup is wanted, or data outgrows laptop + drive (LanceDB can read from S3).

### 2026-10-15 (evening) · Mirror, back-to-back plays, recording clips (Cole's go on all three)
**Mirror (Cole's correction).** Mirror copies the card just played, so it is ALWAYS the very next play; by the 8-card
cycle a card can't return sooner by itself. Rule: the same card twice in a row = the second is Mirror. Measured:
- the answer keys themselves have the same card twice in a row 46 times without a Mirror label: unread Mirrors in keys;
- a pure "twice in a row -> Mirror" pass (`mirror_repeat`) changed little (held-back -6 right, test 0, dev +2): most
  repeats it found were our own double counts (already fakes) or a misread first card; a confidence guard only
  neutralised it. Kept off.
- Mirror delay from the copied card varies (median 5.2 s, up to 50 s), so no time limit is possible.
- the real loss: an older rule. A repeat of a transforming card nearby counts as the same unit wandering back
  (`continuation_*`), and the old Mirror rule drops unclocked repeats as duplicates. Example (cole 09-29 b m03, 7:50):
  Goblin Demolisher, Mirror 2.6 s later, two Demolishers visible at once in different lanes, the second dropped.
  Added `Deploy.twin` (spawns._twin: another unit of the card alive elsewhere at the birth, both seen several times)
  and `twin_new` (a twin is a new play, never a continuation or duplicate): dev +2 right / +8 fakes, held-back -1 / +4.
  The example is now called Mirror but 3 s late (copy first detected late), so still scored missed. Kept off.
**Back-to-back plays.** Of the 64 missed plays within 1 s of another (held-back replays), 35 were seen by the detector,
but their detections joined the other card's track nearby (tracking allowed a detection to continue a track of a
different card, for class flicker on one unit). `spawns.CROSS_CARD` (env `BOT77_CROSS_CARD`): off = a track only takes
its own card; flicker becomes a short extra track that grouping folds into the same deploy anyway.
Held-back replays (V4 dets, spell_v2): 64.2% / 209 wrong / 318 fakes -> **67.8% / 190 / 329**. An evidence split of
mixed-card groups (`split_min_dets`, spawns._split) adds nothing on top (4, 6, 10, 16 dets all 67.6-67.7%); kept off.
Held out (V5 best, spell_v2), cross-card joins on -> off: replay test 60.6% / 140 / 6.6 -> **63.1% / 126 / 6.8**;
dev 56.1% / 44 / 8.1 -> **57.2% / 48 / 8.6**; benchmark 65 / 5 / 28 -> **72 / 7 / 31**. Now the default (`BOT77_CROSS_CARD=1`
restores the old tracking).
**Recording clips in the spell training (spell_v3).** 16,337 clips incl. 2,848 cut from Cole's own recordings (POV),
14 epochs on a RunPod 4090 (~$1.35; pod terminated), best val macro-F1 0.778 (epoch 11). Alone it trades: on the
replay test +26 right, on Cole's recordings worse (dev -11 right, benchmark -5): Poison fakes 16 -> 1, Barbarian Barrel
and Void up, but Goblin Curse (-7 on dev; the missed Curses then show as fake Goblins), Vines (-6) and abilities (-5)
down. The Void/BB cutoffs no longer help with it (no cutoffs: held-back 67.8 -> 68.2%).
Mix (`scripts/mix_spell_probs.py` -> `data/models/spell_mix/spell_probs`): v2 and v3 probabilities averaged, but Goblin
Curse, Vines and ability classes from v2 alone; no per-card cutoffs. Picked on held-back + dev only (plain average
was close: held-back 1179 / 323 fakes, dev 423 / 204).
| set | v2 (previous default) | mix (new default) |
|---|---|---|
| held-back replays (tuning) | 1178 right / 190 wrong / 329 fakes | 1181 / 187 / 326 |
| dev (tuning) | 422 / 48 / 224 (57.2%) | 430 / 47 / 215 (58.3%) |
| replay test (held out) | 762 / 126 / 203 (63.1%) | **780 / 120 / 196 (64.6%)** |
| benchmark (held out) | 72 / 7 / 31 | 72 / 7 / 31 |
Now the playrules default (`spell_probs` = the mix dir, `spell_class_thr` = {}). New videos need both v2 and v3
inference (scripts/katacr/infer_spell.py), then the mix script. Open: why v3 under-calls Goblin Curse / Vines on POV
(likely some of Cole's recording clips labelled "none" contain an unlabelled opponent spell).

## 2026-10-16 · Failure mining with LanceDB (Cole: "use LanceDB to mine for our failure modes")
**The table.** `scripts/build_failures.py` scores the default pipeline against every answer key outside the test set
and benchmark (380 replays: 332 train split + 48 held-back, V4 dets; 25 dev recordings, V5 dets) and writes one row
per true play (right / wrong / missed) and per fake to the LanceDB table `play_outcomes` (16,909 rows), with context:
was the true card ever detected near the play (`seen_n`, how late), neighbour gaps, repeat of the previous card,
spell-stage coverage. Spell plays in games without spell-stage probabilities are left out of the analysis.
**What it showed** (11,691 true plays: 7,634 right / 1,336 wrong / 2,721 missed; 2,096 fakes):
- most misses are not the detector: 1,508 non-spell misses were seen. A per-stage snapshot (`BOT77_STAGES=1`,
  `bot77.opponent.STAGES`) says where they were lost: tracking 704 (298 with only 1-2 dets), timing 336 (found but
  > 1.5 s off), deck re-pick 228, play rules 200, own-unit filter 40;
- all 228 deck re-pick losses: the true card was pushed out of the inferred deck, mostly by a fake **Rage** (127) —
  the detector's "rage" class fires on any spell circle (Poison, Goblin Curse, Tornado, Earthquake; frames checked) and
  was in the inferred deck of 141 of ~400 replays (true Rage: 13 games). Also Clone (53), Mirror (52), Poison (44);
- wrong cards are mostly swarms read as their unit: Skeleton Army -> Skeletons 89, Furnace -> Fire Spirit 62, Goblin Gang
  -> Goblins / Spear Goblins 61, Minion Horde -> Minions 39. A deploy's `units` is ~1 even for 15 skeletons, so the
  count rule (COMPOSITES) never fired;
- 84% of fakes are within 4 s of a real play: spawned units (Goblins near Curse, Barbarians near Barrel / Hut, Recruits
  near Delivery, hogs near Mother Witch) and the same play counted twice when its timing is ~2 s off;
- 151 seen plays were dropped as "first seen past the river": fast units (Hog Rider, Royal Hogs, Prince) and pocket
  plays, first seen 1-3 tiles past it within a second of the play;
- truly unseen (no detection of the card at all): ~750, led by Miner 87% missed, Goblin Drill 81%, the spirits, Mini
  P.E.K.K.A. Frames show the deploy clock and spawn glow covering the unit; spirits die within a second.
**Fixes** (tuned on held-back replays + dev only), held-back 1181 / 187 / 326 -> 1225 / 171 / 326, dev 430 / 47 / 215 ->
432 / 47 / 213 (right / wrong / fakes):
- `det_shadow` {"Rage": 3}: a detector-only Rage with another spell within 3 s is that spell's circle (ignoring "rage"
  outright lost 3 true Rages on dev that the spell stage misses); `det_ignore` ["clone"];
- `composite_peak` (spawns.composites_by_peak): peak units of the class in one frame near the deploy -> Skeleton Army at
  >= 5 skeletons (Skeletons <= 3, Army 4-13), Minion Horde at >= 4 minions: held-back +10 right / -12 wrong;
- replay timing refit on the 332 train-split replays (was 70): 38 of 73 cards moved >= 0.3 s (Wizard, Bowler, Hunter,
  Sparky ~3 s after the card leaves the hand, not the 1.2 s default): +3 right / -7 wrong;
- `half_margin` 3.1 tiles for replays (was 0.8): +28 right / +12 fakes; on POV recordings it lost, so replays only;
- `pocket` (spawns.towers_down; Cole: pocket = 4 tiles past the river): once our princess tower on a lane is down (its
  detections stop), new enemy units may be first seen 4 tiles past the river there: dev +2 right;
- spawners (Cole: cursed hogs / goblins appear where our troop died): Goblin Curse -> Goblins (dev -2 fakes), Mother
  Witch -> cursed hogs within her reach (7 tiles); a longer birth confirmation window changed nothing.
**Mining for detector labels.** `scripts/mine_failures.py`: at each never-seen play in the train-split replays, the
V5 detector re-run at conf 0.05 on six frames 0.4-2.4 s after the play; the best box is kept if called the true card's
class, or if it looks more like the card than like its own class (DINOv2 top-5 cosine to confident `unit_crops` of
each class, LanceDB vector search). Frames show why these plays are missed: the unit is under the deploy clock or in its
spawn glow, and spirits die within a second. Cole's review (scripts/build_mined_review.py, two rounds): round 1 kept 54
of 74 boxes (184 plays mined), round 2 57 of 66 (246 plays); Miner 4 of 11, Goblin Drill 1 of 4 (Drill found almost
nothing: it only shows as it surfaces). `scripts/label_mined.py` -> 105 full-frame labels (data/external/real_mined*):
the mined box SAM-tightened and named by the answer key (evo / hero form too), plus every other confident V4 box.
Spell clips: Cole confirmed 13 of 15 recording "no spell" clips that the replay-only spell_v2 called Goblin Curse /
Vines hold an opponent one (config/spell_clip_relabels.json, applied by scripts/spell_clips.py). Of 2,382 recording
negatives v2 flags 381, mostly Void, the over-call those clips were added to fix: the bigger cause of v3's weak Curse /
Vines on POV is that Cole's deck has neither, so there are no POV examples of them at all.
**Retrain (RunPod 4090 Secure, $0.89/h, pod rjkguya7kia80i, 08:56-10:48, ~$1.65; terminated):**
- detector V7 (`scripts/katacr/pod_v6.sh`): 4 epochs from V5 best on real_v3 x3 + real_v4 + real_v5r + mined x10, no synth
  (setup alone would cost an hour). Val mAP50 / 50-95 0.725 / 0.412 vs V5 0.742 / 0.434. Plays: dev 421 / 48 / 228 vs V5
  432 / 47 / 213; held-back replays (both run fresh on the 48 games) 1146 / 192 / 251 vs V5 1166 / 194 / 255. **Not
  adopted**: dropping synth cost more than 105 frames gained. Mined labels kept for the next full retrain.
- spell_v4 (v3 data + the 13 relabels, 14 epochs, val macro-F1 0.804 vs v3 0.778): worse on whole games in every mix
  (held-back / dev right: v2+v4 1221 / 428, full average 1218 / 417, v4 alone 1213 / 409, v2+v3+v4 1223 / 425, vs
  the v2+v3 mix 1225 / 432). **Not adopted**; 13 clips are too few to rise above run-to-run noise.
**Final check** (scored once; defaults = the rule fixes above, V5 detector, v2+v3 spell mix), right / wrong / fakes:
- replay test (30): 780 / 120 / 196 (64.6%) -> **792 / 119 / 193 (65.6%)**; with V4 detections 806 / 95 / 243 (66.8%)
- benchmark (116): 72 / 7 / 31 -> 72 / 7 / 30
Smaller than the held-back gain (+44 right): the rules were tuned on held-back with V4 detections. V4 vs V5 on replays:
held-back (1225 / 171 / 326 vs 1166 / 194 / 255) and test agree that V4 names more plays right with fewer wrong cards
but more fakes; V5 stays the default for POV recordings (dev, benchmark).
Budget: ~$5.25 of RunPod's $10 spent, ~$4.75 left.

## 2026-10-16 (afternoon) · Cutting fake plays (Cole: "I just want to get the hallucinations down")
Where fakes come from now (default rules): held-back replays 326 fakes of 1,722 logged plays (19%): 232 an extra card
within 4 s of a real play, 36 the same card twice, 35 within 4-10 s, 16 ability calls, 7 invented. Dev recordings
213 of 692 (31%): 66 invented > 10 s from any play, mostly spells (Goblin Curse 31, Vines 22, Void 17: the spell
model over-calls on POV), then two-part cards (Goblinstein 17, Spirit Empress 15, Tombstone 8).
**Elixir check** (`bot77.elixir`, rule `elixir_check`, off): simulate the opponent's elixir (5 at the start, 10 when a
recording joins mid-match; +1 per 2.8 s x the on-screen multiplier; cap 10; Mirror = copied card + 1, abilities =
cards.ability_cost) and drop a play when the logged plays overspend. The model holds on the answer keys (43 of 48
held-back games never below -1.5; the rest are key errors, e.g. one Mirror listed twice), but our play times and costs
(wrong cards) are rougher: every drop policy (fewest votes, latest play, weaker of a tight pair) cut ~80 fakes for
~45 right on held-back. Kept off; its elixir estimate is a feature of the filter below.
**Learned fake filter** (`bot77.fakefilter`, scripts/train_fake_filter.py, rule `fake_filter`): gradient-boosted
P(fake) per logged play from 31 features (votes and their mix, units, delay, spot, card cost / type / deck rank,
neighbour gaps, same card nearby, plays within 4 s, simulated elixir before / after, clock, multiplier). Trained on the
332 train-split replays (10,638 plays, 21% fake; re-scored with today's rules); AUC held-back 0.868, dev 0.648 (trained
on replays only: the POV gap again). Threshold sweep, right / wrong / fakes:
- held-back: off 1225 / 171 / 326; 0.7 1220 / 166 / 268; **0.6 1212 / 160 / 224**; 0.5 1200 / 155 / 185
- dev: off 432 / 47 / 213; **0.7 430 / 45 / 202**; 0.6 423 / 46 / 183; 0.5 411 / 48 / 164
Default thr 0.6 on replays (fakes 19% -> 14% of logged plays for -1% right), 0.7 on POV recordings; the pipeline
reproduces the sweep (held-back 1212 / 160 / 223, dev 430 / 45 / 203).
**Keyed POV footage without a friend** (Cole records each live game, then its in-game replay through the same
iPhone Mirroring setup): `bot77.pairing.pair_live_replay` pairs a live recording with its replay by the game itself
(Cole's live POV plays must reappear as the replay's bottom-hand plays at the match-clock offset; >= 70% of >= 5 plays
by card within 1.5 s), and `write_live_truth` moves the replay's top-hand plays onto the live time line ->
opponent_truth_live_<match>.json (`eval_dev.py --set live`). Checks: a replay paired with itself -> offset 0, 48/48
plays agree, the same 45-play key; with three other games -> no pair. `scripts/ingest_live_replays.py` converts and
processes a folder of <name>_live / <name>_replay recordings; it needs replay layouts for mirroring-size frames
(cole_replay_bottom / _top), calibrated on Cole's first replay recording. Deck advice for the session: put the cards
we're worst on in Cole's own deck (own plays are labelled exactly): Goblin Curse, Vines, Void; Goblinstein, Spirit
Empress, Tombstone; Miner, Goblin Drill, the spirits, Mini P.E.K.K.A; plus Hog EQ games (B's deck).

## 2026-10-17/18 · Cole's keyed live games, full retrain (V8, spell v5), fake filter v2
**Data.** Cole played ~20 ladder games recorded with iPhone Mirroring (3 recordings, 74 min; decks with Goblin Curse /
Vines / Void and the two-part cards, as suggested). `scripts/replay_runner.py` recorded the 25 newest Battle Log replays
in BlueStacks over ADB (one "Connection lost" mid-run: Retry Login, rerun with --skip 11). Live recordings processed as
cole_2026-10-17_{a,b,c} (20 matches), replays by ingest_replays.py (bl_2026-10-17_r01-r25).
**Pairing** (`scripts/pair_live_replays.py`, bot77.pairing.pair_live_replay): all 20 live games paired, 96-100% of
Cole's own plays agreeing; order matched Cole's account (newest replay = last game of the last recording). The match
clock shows whole seconds, so its offset was 1-2 s off and 4 games first missed the 1.5 s window: the offset is now
refined on the plays (+-3 s in 0.1 s steps, plays within 0.5 s), which also makes the keys' times exact.
**Split.** Live test (config/live_test_games.json, never trained or tuned on): recording b, 10 games, AND the replays of
those same games (bl r05-r14; same match, so training on them would leak the test). bot77.holdout.held_out() lists
every held-out game and every training builder now refuses them. Live training games: recordings a + c (10).
Note: V5 trained on 46 of the 48 held-back replays (real_v5r), so held-back can't pick detector checkpoints; rule tuning
on held-back uses V4 detections (V4 never saw them).
**Training data added:** 770 spell clips from the new games (Vines 36, Goblin Curse 29, Void 18 in Cole's view; opponent
spells from the keys); 442 clock-spot detector labels (308 replays with the reference check, 134 live -> Cole's review
kept 109); the 105 mined boxes. Spell clips: `spell_clips.py --only-new` (keeps the existing 16,337).
**Pod** (4090 Secure, $0.89/h, uu3z2043yui7hq, 10:45-16:09, ~$4.80; terminated): synth_v3 regenerated (48,015), spell
v5 (14 epochs on 17,107 clips, val macro-F1 0.781), detector V8 (`scripts/katacr/pod_v8.sh`: real_v3 x3 + real_v4 +
real_v5r + synth + mined x5 + live x3 + replays 10-17, 12 epochs from V5 best; best epoch 7 val mAP50 / 50-95 0.734 /
0.445 vs V5 0.742 / 0.434), then v2 / v3 / v5 spell probabilities for all 130 streamed games.
**Spell choice** (right / wrong / fakes): held-back v2+v3 1212 / 160 / 223 vs v2+v5 1206 / 160 / 218; dev v2+v3 430 /
45 / 203 vs v2+v5 433 / 45 / 203 (live-train favours v5 but v5 trained on those games). Default now split by source:
v2+v5 for POV recordings (spell_mix25), v2+v3 for replays (REPLAY_SPELL_PROBS).
**Detector choice:** dev with today's defaults V5 430 / 44 / 182 vs V8 429 / 40 / 190: no gain, V5 stays the POV
default (pre-registered rule: adopt V8 only if it beats V5 on dev). V8 is reported on the test sets for information.
**Fake filter v2:** + the 10 live training games (scored with the filter off, so their fakes are visible): dev AUC
0.648 -> 0.695, held-back 0.869. POV sample weight x5 = same, x15 worse. Default thr 0.6 for both: dev 434 / 47 / 214
(off) -> 430 / 44 / 182; held-back 1225 / 171 / 326 -> 1213 / 160 / 221.
**Final check (scored once; defaults: V5 for POV, v2+v5 spells on POV / v2+v3 on replays, fake filter v2 at 0.6)**,
right / wrong / fakes (share of logged plays that are fake):
- live test (Cole's recording b, 10 games, 290 plays): yesterday's setup 130 / 23 / 49 (44.8%, 24% fake) ->
  **148 / 13 / 51 (51.0%, 24% fake)**; V8 dets 147 / 19 / 44 (50.7%)
- replay test (30): 792 / 119 / 193 (65.6%, 17% fake) -> **785 / 110 / 134 (65.0%, 13% fake)**; V4 dets 806 / 95 / 243 ->
  799 / 89 / 162 (66.2%, 15% fake); V8 781 / 105 / 148
- benchmark (116): 72 / 7 / 30 -> **70 / 6 / 24**; V8 69 / 4 / 22
The filter cuts replay fakes by ~30% for ~1% of right plays; on Cole's own games the gain is the spells (+18 right, wrong
23 -> 13) while fakes stay ~5 per game. V8 matches V5 (not adopted). Pod total for the project so far ~$10.05.
**POV fakes, targeted (Cole: "target them the way the rage fix worked"), dev + live-train with today's defaults:**
205 fakes: 182 on dev, 23 on the 10 live-train games (2.3 per game). By recording: the 09-29 friendlies vs Cole's
friend 5-9 per game, 10-11 up to 11.5, the 10-17 ladder games 1.7-2.6. Checks:
- not our own plays: 3 of 64 Curse / Vines / Void fakes have our same card within 8 s;
- not key gaps: 11 of 182 dev fakes have an unnamed play in the opponent's own hand within 3 s; 128 have nothing;
- the opponent owns the spell (54 of 64) but didn't play it then; no link to the last real cast (1.5-117 s before);
- frames: at 4 of 6 Curse / Vines fakes the POV player is dragging a card (placement preview: ghost unit, range circle,
  card name, red no-deploy overlay). But hud "selected" is equally common at real spells (16-22% vs 18-27%), and only
  ~20 fakes vs ~11 real spells fall < 2 s before an own play: a weak signal, no clean rule;
- spell mix without v2-only Curse / Vines: dev -5 fakes / -6 right (kept as is);
- champions / two-part cards (Goblinstein, Spirit Empress, Tombstone; all in the friendlies, 18-80 s after a real
  play): wider continuation (any class, 18 tiles, 60-90 s) dev -13 fakes / -37 right; rejected;
- the weak own-play signal as filter features (own_next / own_prev / own_same_near, fake filter v3): dev AUC 0.697 vs
  0.695, held-back equal; v2 stays (v3 code kept; older models load with the first 31 features).
No Rage-like fix exists for these: the remaining POV fakes are spread over many causes. The lever is more keyed POV
games for the filter (10 games moved its dev AUC 0.648 -> 0.695).
**2026-10-19/20 session:** Cole recorded 12 games (cole_2026-10-19_a, 46 min, mostly Hog EQ). BlueStacks' ADB was at
127.0.0.1:5555 this time (emulator-5554 missing; `--device 127.0.0.1:5555`). The runner stopped after 2: BlueStacks had
lost its connection and showed a stale Battle Log (Cole); force-stopping and relaunching Clash refreshed it. With the
same deck every game the "recorded before" thumbnail check can misfire, so `replay_runner.py --no-known` skips it.
ingest_replays.py skips files younger than 2 min, so the replay finished seconds before an ingest needs a second pass.
All 12 keyed (96-100% of own plays agreeing): live training games 10 -> 22. Waiting for Cole's next session before one
pod run for spell probabilities and the fake filter retrain.
**2026-10-20 session + fake filter v4.** Cole recorded 11 games (cole_2026-10-20_a: log bait / Goblin Curse, then
E-Giant); 12 replays recorded after relaunching Clash in BlueStacks. 8 paired at once; games 1-3 had the same play
sequence as their replays but the gap drifted 8-13 s over a match in small jumps (the iPhone Mirroring recording
stalled). Pairing now (a) seeds the offset from the plays too, not only the clock, and (b) when no single offset fits,
maps time piecewise between in-order card anchors, accepted only if the gap drifts smoothly (<= 4 s between anchors,
median within 20 s of the clock offset). A first, looser version paired same-deck games with the wrong replays (any
game with the same 8 own cards lines up by card order) and later replays overwrote earlier keys; now the gap check
plus a one-to-one choice (best agreement first, each live match and replay once; old keys of the video deleted first)
gives 11 / 11 with 11 candidates. Live training games: 33 (+ the 10 test games). Spell probabilities for the 24 new
games on a 10-minute pod (c7arnzrs25daoa, ~$0.15, terminated).
Fake filter v4: replays + the 33 live training games (scored with the filter off), POV sample weight 3 (x1 and x6
worse). Dev AUC 0.695 (v2, 10 games) -> 0.724; held-back 0.865. At thr 0.6: held-back 1213 / 160 / 221 -> 1215 / 160 /
208, dev 430 / 44 / 182 -> 425 / 46 / 165. Live test (recording b, scored once): v2 148 / 13 / 51 (24% of logged plays
fake) -> **v4 148 / 14 / 29 (15% fake)**: fakes -43% at no cost in right plays.

## 2026-10-20/21 · Fourth session, spell v6, fake filter v5 (not adopted)
**Data.** cole_2026-10-20_b: 12 Hog EQ games, 12 replays (r23-r34; recorded with --no-known after relaunching Clash; the
LaCie had to be plugged in for the live file). 12 / 12 paired. Game 12's replay recording started ~9 s late (first play
missing) and the gap drifted 39 s with one 8.3 s jump, which broke the greedy in-order anchors (the missing first play
pulled the match onto a later copy of the card). Anchors are now a longest-common-subsequence alignment, and the
piecewise map is accepted on a steady-gap test measured on true vs false pairs: >= 85% of plays aligned, median step
<= 1.5 s, >= 60% of steps <= 2 s, max step <= 15 s (true pair: 94%, 0.7 s, 76%, 8.3 s; five same-deck false pairs: <= 72%,
4.3-9.0 s, 14-29%, 18-34 s). Re-pairing cole_2026-10-20_a with it: 11 / 11, still no extra candidates.
Live training games: 45 (+ 10 test). Spell clips: +2,381 from 71 games (19,488 total).
**Pod** hrwfzngjymhgq8 (4090 Secure, 20:56-01:07, ~$3.70: the 10 GB upload was slow; terminated): spell v6, 14 epochs,
val macro-F1 0.791 (v5 0.781); v2 / v3 / v5 / v6 probabilities for all 168 streamed games.
**Spell choice** (right / wrong / fakes): dev v2+v5 425 / 46 / 165 vs v2+v3 420 / 46 / 164, v2+v6 418, v2+v5+v6 417: POV
keeps v2+v5. Held-back v2+v3 1215 / 160 / 207 vs v2+v6 1219 / 159 / 208 vs **v2+v5+v6 1224 / 152 / 201**: the replay
default is now v2+v5+v6 (REPLAY_SPELL_PROBS = spell_mix256).
**Fake filter v5** (383 train replays rescored with the new replay spells + 45 live games, filter off): held-back AUC
0.878, dev 0.709-0.711 (POV weight 1 / 3 / 6). v4 scored on the same tables (`train_fake_filter.py --model-in`): held-back
0.869, dev 0.724; at 0.6 v4 1224 / 152 / 202 and 425 / 46 / 165 vs v5 (w1) 1223 / 153 / 214 and 426 / 42 / 181. **v4
stays**: 12 more games did not help this time (dev is 695 plays; the filter may have learned most of what these features
carry about POV fakes).
**Test (scored once after the replay spell change):** replay test V5 dets 785 / 110 / 134 -> 774 / 117 / 128, V4 dets
799 / 89 / 162 -> 800 / 88 / 169 (within noise; held-back had +9). POV defaults unchanged, so the live test stays
148 / 14 / 29 (15% of logged plays fake) and the benchmark 70 / 6 / 24.
Note: the session scratchpad lost its held-back id list and shards mid-run; both are rebuilt from config (heldback_replays,
holdout) whenever needed.

## 2026-10-21 · Step A starts: move predictor v1 (Cole: "make the commit, and build it out")
Committed C (dc6b2c7). Correction to what I told Cole the night before: per the 09-24 plan, A is the **pro move
predictor** (what a strong player plays next; B's starting policy), not an opponent-play predictor. For the acting
player their own hand, elixir and plays are exact; C only supplies the opponent's history.
A v1 predicts which of the 4 hand cards is played (docs/move-predictor-spec.md). 24,142 decisions from both players of
471 keyed replays (`move_samples`). Held-back: random 24.0%, best rule 35.8%, A v1 44.2% top-1 / 72.9% top-2; adding
response rates (card chosen right after the opponent's / own last card, cross-fitted by game) -> **46.6% / 76.0%**.
Test, scored once: **44.0% / 72.9%** (top ladder 42.0% / 73.1%). With the opponent's history from C instead of the
key: held-back 43.6 -> 43.9%, test 43.4 -> 41.1% top-1, top-2 unchanged: C's noise costs A little, because A v1
mostly uses the player's own state. Next: when, where, and board state / opponent elixir from C.

## 2026-10-21 · Step B v0 built offline (Cole: "build everything ... verify it works as best u can without using bluestacks")
Cole is making a brand-new account for the bot (ToS: automated play can get an account banned; the September plan
says throwaway accounts only — this one starts at 0 trophies, which also tracks progress). Built `bot77.live`:
- **state.py:** one frame -> hand, next, elixir, clock, multiplier, enemy units (the pipeline's own readers + V5
  detector); the deck is learned from the hand, then the reader restricts to it; a briefly unreadable slot keeps its
  last card. Checked on Cole's recordings vs the stored HUD reads: 426/480 and 430/480 slot reads agree, **0 wrong
  cards** (the rest are cards animating in); 115 ms per read. OpponentTracker: an enemy card not seen for 6 s is a play.
- **A at play time:** card features moved into `bot77.moves.candidates` (shared by training and live) and
  `MovePredictor` loads the model with its response-rate tables: reproduces 46.6% top-1 on held-back exactly.
- **policy.py:** when = defend if enemy units are on our half, else at 9.5 elixir, or at 7 after 6 s idle (1.2 s
  minimum gap); which = A v1 ranking among affordable cards with a sensible spot; where = rules by role (win conditions
  at the bridge, buildings centre, Miner-type on a tower, defenders in front of the threat, spells on the biggest
  group — on our half while defending). Shadow and dry runs found and fixed: Barbarian Barrel aimed at a group on the
  enemy side while defending; Earthquake / Mighty Miner sent to a tower during a defence; Hog counter-pushes with
  < 7 elixir.
- **device.py / loop.py:** ADB taps (select card, tap spot), screen recognition (battle from the HUD, end screen from
  the replay runner's VS / OK templates, home from a Battle-button template cut from a BlueStacks screenshot), and the
  loop with safety stops (data/live/STOP, max games / hours, 4 min unrecognised screen, screenshot failures). Actions
  and games logged to LanceDB (bot_actions, bot_games).
- **Calibration:** `scripts/live_calibrate.py` derives the BlueStacks battle layout with bot77.autolayout, then fits
  the arena crop to the princess towers (`bot77.live.arena`: the HUD fit places the arena only to ~50 px). Offline on
  two of Cole's recordings: arena within ~5 px of the hand-made layout, our towers off by 0.06-0.08 tiles.
- **Checks:** `scripts/live_shadow.py` (Cole's game, his plays fed as the bot's history): deck read 8/8, card agrees
  9/20 (45%, = A's rate), opponent tracker 10 of 23 matches the key. `scripts/live_dryrun.py` (the whole loop on a
  BlueStacks replay video): battle and end screen recognised, 40 decisions, 0 taps off screen.
  tests/test_live_policy.py: 9 tests pass (placement rules, gap, tap geometry). tests/test_cards.py::test_curated_coverage
  fails before and after this work (card metadata grew); left as is.
Next on BlueStacks: calibrate during the new account's first battle, then one watched game (`live_play.py --games 1`).

## 2026-10-22 · Playbook rules from the play reviews (Cole: "Just use em")
Cole reviewed 60 of the bot's plays from its lost games; Claude reviewed the other 84 with the principles distilled
from his (skill `.claude/skills/review-bot-plays`: contact sheets of the frame at each play and 2 s later, 4 plays per
image) and Cole approved all of them. 73 of 144 plays were mistakes. Commonest: Giant at the bridge with nothing going
on; idle plays at 7 elixir; Arrows on a lone Knight / Valkyrie / Giant / building; Musketeer or Archers on top of the
attacker; defending before the attacker crossed; answering a couple of Archers; nothing behind our Giant.
- **`config/playbook_rules.yaml` + `bot77.live.rules`** (the policy's default; `Policy(rules=False)` is the earlier B v0):
  no idle plays (play at 9.5 elixir only, Giant behind the king tower, never the bridge); defend only units on the
  bridge or past it (y >= 15.2), not cheap ones alone (Archers, Skeletons, spirits, Bats under 4 elixir), one answer
  per attacker per 6 s (the bot dropped Minions twice on the same Minions); per-attacker answer lists (tanks: Mini
  P.E.K.K.A > Musketeer > Minions; Valkyrie: Minions first; ...), never a Giant as a defender, no ground troops on
  air, spells only on units they kill and worth >= 2/3 of the spell; Mini P.E.K.K.A / Minions on the target,
  ranged troops in front of the king tower, other melee on top of the attacker; one support play behind our tank
  while it walks to the bridge. A ranks the cards the rules leave open.
- **Offline check** (`scripts/eval_rules_on_reviews.py`, states rebuilt from bot_actions: hand, elixir and the enemy
  units logged on our half; our own units weren't logged, so support can't fire there): the rules change 70/73 bad
  plays (32/47 to exactly the reviewer's card or "wait"); of 71 OK plays, 43 keep the card, 15 become waits (mostly
  opening cycles and units logged just short of the bridge, which the live loop answers a tick later) and the rest
  switch to the listed answer (e.g. Mini P.E.K.K.A instead of Archers on a tank).
  Tuned on the same 144 plays, so this is a sanity check, not a held-out score: the next session's reviews are.
- `tests/test_live_rules.py`: 11 tests, one per review finding.

## 2026-10-22 · Board state for A, a placement model (Cole: "We can do all 3 ... similar moment search should be held back")
1. **Full board logged:** every bot play now saves all detected units (`bot_actions.units`: card, class, team, x, y,
   conf) and the placement model's spot (`model_spot`); the loop adds new columns to the existing table.
2. **Board states + A v2.** The replays' per-frame detections already existed (`unit_detections_v5l_e15`, every
   0.25 s, all 524 keyed replays), so `scripts/build_board_states.py` takes the last frame >= 0.25 s before each of the
   24,142 decisions (none missing) -> table `board_states`, in the deciding player's frame (`bot77.board`: the top
   player's board turned around about y 27.5, not 2 x river: unit boxes sit ~1.3 tiles above the feet, measured from
   the two players' answer positions). Features (shared with live play): board summary (enemy elixir on our half per
   lane, nearest attacker, air / tank / swarm coming, our units, towers standing, the main threat's card) and card vs
   board (can it hit the threat, air answer when air is coming, spell value of its best target, the answer rate to
   the threat's card, roles from `config/card_roles.yaml` -- written from general knowledge, for Cole to correct).
   `train_moves.py --board` -> data/models/moves_v2. **Held-back 46.6% -> 48.4% (McNemar p 0.03), but test 44.0% ->
   44.0%** (only with an attacker on our half 47.3% -> 48.9%). Not adopted: the bot keeps A v1; v2 stays saved. The
   board mostly decides *whether* to play and *where*, which the rules / placement model now handle, while A's card
   choice is driven by the hand and cycle.
3. **Placement model** (`bot77.place`, `scripts/build_placements.py`, `scripts/train_place.py` -> data/models/place_v1):
   14,080 plays with a spot and a board (45% of plays: the rest weren't matched to a detection). It ranks candidate
   spots like A ranks cards: 32 fixed spots (k-means of where players put cards) plus spots anchored on the board (on
   the main attacker, just in front of it, on other enemy units, behind our tank, a spell's best group, the enemy
   towers), with per-spot features (distance and lane vs the threat, enemy value nearby, our / their units and towers
   in that lane). Taps go FEET_DY = 1.27 tiles below the box centre. Lane-context features took held-back from 4.8 to
   4.1 tiles and same lane 64% -> 74%.
   **Test (scored once):** median 4.1 tiles from the real spot, 27% within 2 tiles, same lane 77% (nearest candidate,
   the ceiling: 0.9 tiles). On the answers the playbook covers: model 4.3 tiles / 79% same lane vs the playbook's
   typical spot 5.0 / 76%, the playbook rules 7.1 / 50%, the earlier rules 7.7 / 51%. The rules are poor as a
   general placer (they were written for the bot's deck and Cole's principles, and pick a lane by a heuristic).
   **In the bot: shadow only** -- each play logs where the model would have put the card; the rules still choose the
   tap. Next: compare both on the next session's reviews, then let the model place where the rules have no principle,
   with the rules as guard rails (no Giant at the bridge, ranged away from the attacker).
4. Similar-moment search held back (Cole). The `interactions` table (20,643 answer pairs) already exists and feeds the
   playbook and the placement comparison.

## 2026-10-22 · Training-data rebuild, phases 0-3 (Cole: "everything feels glued together and half assed")
Plan: docs/training-architecture.md; progress: ROADMAP.md. Bot sessions paused until it's through.
- **Phase 0:** inventory (docs/data-inventory.md) and `media` (every video file: laptop or LaCie). 401 experiment
  tables and 27 GB of RunPod bundles / old review pages to the Trash (Cole: bin them). LanceDB 80 → 18 GB: `unit_crops`
  carried 58 GB of version manifests (one per append) — `optimize(cleanup_older_than=0)` after big append jobs.
- **Phase 1:** `matches` gets source / split / pov side / pair / truth set / league / result (split once, from the
  config lists; agrees with A's samples on every row); `detections` (all 6 detector runs, `run_id`, tiles) + `runs`;
  `plays` (hand / detected / bot).
- **Phase 2:** `moments` (replay plays, 116k could-have-played waits, bot plays), `labels`, `reviews`. A's rows rebuilt
  from them are identical to `move_samples` (96,568 rows); A v1 46.6%. The bot's own replays (53) were about to become
  imitation data (A dropped to 45.5%): now `replay_bot`, excluded by default.
- **Phase 3:** `bot77.features` + registry (88 features, docs/features.md). Every model reproduces exactly through
  it given the same card-roles file; the newer roles (Cole's win-condition definition, newer-card guesses) improve the
  placement model 4.1 → 3.9 tiles held-back (`place_v2`, now the bot's shadow placer). A v2 unchanged (not adopted).

## 2026-10-22 · Phases 4-6: data browser, models registry, card + timing model (A v3 / v4)
- **Data browser** (`uv run python -m bot77.browser`): FastAPI over LanceDB + React / TypeScript / Tailwind. Scalar
  indexes took a match page from ~5 s to 0.1 s. Bot play times were session time: now video time (game start + a
  measured 1.5 s recording lag); the bot's index-less .mkv recordings are remuxed to .mp4 on first view.
- **Models registry** (`bot77.models`, `models` table): 39 versions on disk seeded with their history from this log;
  one adopted per role; the bot loads adopted models; training scripts register what they save.
- **A v3 (card + timing):** "wait" as a fifth candidate at every play and could-have-played moment of the human
  replays (129,844 moments), board + timing features. Held-back: card at plays 51.5% (A v1 46.6%), play-vs-wait AUC
  0.78 (elixir 0.63, the rules' trigger 0.60). Test, scored once: card 50.2% (A v1 44.0%), AUC 0.78, at the rules'
  false-play rate 60% of real plays caught vs the rules' 40%. The play threshold is set on held-back so the model plays
  as often as players do (it catches 45% of plays and keeps 88% of waits there; the rules 40% / 79%).
- **Reviews as labels (A v4):** the bot's games paired with their Battle Log replays (`pair_bot_replays.py`; the
  replay's match clock runs 3-5 s ahead of the bot's clock reads, so each pair finds its offset): 49 of 53 paired, 96%
  median agreement; 118 of 144 reviewed plays land on a replay moment, 94 usable as labels (the rest are spot-only
  mistakes or a fix card not in hand). Weight chosen by holding out a third of the reviewed games at a time: agreement
  with the reviewer 0.33 (weight 0) / 0.44 (5) / 0.50 (20) / 0.53 (50) / 0.53 (100); held-back card 51.5 → 51.2% at 50.
  A v4 = weight 50; test card 48.8%, AUC 0.78. Small sample (94 moments, 7 games): the next session's reviews are the
  real check. In the bot A v4 runs in shadow (`bot_shadow`), the rules still play.

## 2026-10-23 · Session 2 processed; Claude's pre-review vs Cole; simulator plan
- Session 2 (61 games, new deck): all 61 replays keyed (0 errors); 55 of 61 bot games paired with their replays.
- Review round 1 (6 lost games, 142 plays) pre-reviewed by Claude (skill review-bot-plays). Cole reviewed 47 and
  discarded the rest: Claude agreed on 26 of 44 (59%), called 17 of Cole's 30 mistakes OK, high confidence right 5 of 13.
  Claude misses plays whose judgement needs the next seconds (dead Giant: reset; Minions outrun the Giant; kiting).
- So: a simulator, to search the next ~10 s from every moment (`docs/simulator.md`). Evaluated two (code read, not
  run): ClashRoyaleEnv (itzik123, MIT, C++ / pybind11, 132 cards incl. all of ours, 734 tests, state injection and
  snapshot fork; Windows-only build) and Jason-XII's clash-royale-simulator (no license, no Goblin Hut spawning, no
  tests). Decision: fork ClashRoyaleEnv's engine only (not its perception / game-control code), validate it against our
  replays before trusting it; LanceDB holds everything going in and out.
- Cole's blind re-review of 20 round 1 plays: 14 of 20 verdicts the same as his first review (70%; 5 of 6 flips got
  stricter), same fix on 9 of 12 shared mistakes; new notes mostly about timing. Claude on the same 20: 60%. So the
  simulator search's gate (S3) is set at ≥ 65% verdict agreement (ceiling ~70%) + Cole's fix in the top 2 on half his
  mistakes. Cole: exact placement and timing is the goal; S2 measures and flags but doesn't gate.
- Correction: one blind flip (`015212:4`, missed Arrows) was Cole not seeing the miss on the frame; corrected. Self-agreement 15 of 20 (75%); gate stays 65%.

## 2026-10-23 · Simulator S0: the ClashRoyaleEnv fork builds on the Mac (gate passed)
Plan: docs/simulator.md. Forked itzik123/ClashRoyaleAi (MIT) as github.com/cole-zoom/clash-royale-env, submodule
`third_party/clash-royale-env`; credited in README.md and docs/references.bib. Took ~1 hour of the half-day box.
- **Build:** no source changes were needed for Apple clang 21 / Python 3.12 arm64. Added a scikit-build-core
  `pyproject.toml` in the fork and made the `.pyd` copy Windows-only; our pyproject depends on it as a path source, so
  `uv sync` builds it (~10 s) and `uv run python -c "import clash_royale_env"` works. cmake 4.4 + ninja 1.13 are dev
  dependencies (for building the Catch2 suite by hand into `build/sim`).
- **One portability bug, fixed in the fork with a test:** `getAllCardIds()` iterated an `unordered_map`, whose order
  differs between MSVC and libc++; the Python probes take the first 8 ids as a deck, which on the Mac put a Hero in
  slot 0, so the whole Python suite failed at conftest. Now sorted (`tests/core/test_card_order.cpp`). Seeded random
  decks can still differ across platforms (`<random>` distributions aren't specified bit-for-bit); we don't use them.
- **Tests:** Catch2 **735 passed + 1 expected fail of 736** (the fork's pinned wedge-collision defect,
  `test_navigation_wedge.cpp`). Python (`python_ai/tests`, run with gymnasium / tensorboard overlaid, not added to our
  deps): **1,007 passed, 2 failed, 3 skipped** (99.8%). The 2 failures are bit-equality checks in their PPO network's
  row compaction under torch 2.14 / ARM; training code we don't use.
- **Speed** (`scripts/sim_bench.py`, 20 seeded random-play matches, session 2 deck both sides, one core of the M5):
  median **2.3 ms per match** (max 3.4 ms; 1,398 ticks median, Python driving every tick), 1.8 µs per tick;
  `snapshot()` mid-match median **2.4 µs** (max 14 µs); a 10 s rollout from a fork in one call 0.03 ms. Faster than the
  fork's own figures (~5 µs / tick, 10 ms / match on an i5 laptop).
- **Gate:** imports under 3.12 ✅; ≥ 95% of tests pass ✅ (Catch2 100% excluding the expected fail, Python 99.8%);
  match < 50 ms ✅. Search budget: 8 cards × 600 spots × 100 ticks ≈ 0.5 M ticks ≈ 1 s per moment at this speed.
- Noticed for S1: `inject()` takes `hp` and `deploy_ticks`, so unit HP can be set at injection (the plan assumed
  full HP only).

## 2026-10-23 · Simulator S1: adapter `bot77.sim`
`src/bot77/sim/` (geometry, cards, engine, scenario); nothing else imports the engine. ~2 h of the 1-day estimate.
- **Coordinates, measured from our detections (v5l_e15):** x maps 1:1 with the engine's half-cell offset (princess
  towers 3.45 / 14.50 in ours, both teams, vs 3.0 / 14.0: same 11-tile spread, no perspective in x). **Our y is
  foreshortened:** 0.7305 of our units per engine row, from the princess towers (box centres 5.94 / 21.28, 21 rows
  apart), through the river (ours 14.9 = engine 16.5). Checks: that puts both princess boxes 1.29 above their tower
  centre (FEET_DY for units is 1.27); Giant / Golem / Knight / Valkyrie / Mini P.E.K.K.A walk 0.73-0.74 of our units
  per engine tile with no trend between y 8 and 18; units cross the river at x 3.7 / 14.2 (the engine's bridges span
  [2, 4] / [14, 16] in ours). **Known misfit:** the king towers imply 0.76 (~0.45 units off behind each king);
  S2 refines the map on unit tracks.
- **Teams:** the bottom / deciding player is engine team 0, in `bot77.sim.geometry` only.
- **Cards** (`docs/simulator-cards.md`, `scripts/sim_cards.py`): all 127 of ours map by name except **Minion
  Giant** (211 of 31,389 human-replay plays, 0.7%); the 4 tower troops are the engine's TowerTroopType. Evolutions and
  Heroes map by the play's `form`; 1,473 plays (4.7%) use a form the engine lacks (Hero Valkyrie, Hero Bowler, Hero Ice
  Wizard, ... Evo Elite Barbarians, Evo Electro Giant) and fall back to the plain card.
- **Engine changes (fork, no gameplay change):** `get_units()` read-only entity view + test. Nothing else needed:
  plays go through the engine's own playCard (`Sim.play(force=True)` puts the card in hand, tops up elixir, moves an
  illegal spot to the nearest legal one within 2 tiles, and records each), `snapshot()` forks.
- **`scenario_from_moment`** (~10 ms): clock, our hand and elixir, both decks (matches.deck + forms seen), towers
  standing, tower troops seen, units (box -> feet; squads clustered within 2.5 tiles, one card per cluster, so a
  half-dead squad comes back whole), opponent elixir from their last play + real regeneration (2.9 s / elixir, measured in S2).
  Not carried over: unit HP (full; `inject` can take HP once we have it), what units were doing, units still deploying,
  spells on the board, queue order past `next`, king wake-up from damage. Tower HP waits for S2's `tower_hp`.
- **Found for S2:** our replays' elixir multiplier is 2x from 2:00 to **4:00** and 3x after; the engine goes 3x at
  3:00. And the engine's elixir runs 2.857 s / elixir vs the real 2.8.
- Tests: `tests/test_sim.py` (7): coordinate round trip and fixed points, card map, a Giant from our bridge stops at
  box y 8.19 (real 7.97) and hits the tower ~10 s after (real walking speed predicts ~9.5 s), Arrows kill a Skeletons
  group, forced plays, snapshot independence, a scenario from a moment. pytest now only collects `tests/`.

## 2026-10-24 · Simulator S2: measured against the replays (no gate; flags)
Full write-up: docs/simulator-validation.md. Held-back only (48 games, 346 windows); knobs chosen on 557 train windows.
- **Windows** on (near-)empty boards, both players' exact plays; spots recovered from the detections where `plays` had
  none (`bot77.sim.spots`, matches stored spots to median 0.00 / p90 1.12; play coverage ~46% -> ~76%).
- **Numbers (held-back, empty start):** position error 0.30 at deploy, 1.0-1.25 of our units from +2.5 s (noise floor:
  building jitter median 0.05); same lane 90% (98% from near-empty starts); fight winner right **57.6% / 62.2%**
  (plan's rough expectation ~80%: missed); sim units outlive replay ones by **+1.75 / +2.25 s**; tower damage where the
  replay shows > 0.1 princess tower: sim / real median **0.02-0.55** (expectation within ~20%: missed).
- **Engine fixes (fork, tests):** triple elixir from 4:00 (586 replays; was 3:00); first-hit load times for 98 cards
  (RoyaleAPI cr-api-data; units hit 0.1-0.3 s after locking on, real 0.4-1.4 s); tower re-levelling API; read-only
  `get_units`; sight switch (no effect measured, left off). Towers set to level 11 to match the cards (RoyaleAPI).
- **Tower HP OCR** (`tower_hp`, Apple Vision on the tower bars) and **matches.result** from the end screen ("Winner!"
  position; 20 / 20 by hand on held-back; the first try from tower detections was wrong on 4 of 12 because TV
  replays' `t_end` is past the end screen).
- **Flags for S3:** lifetimes, big tower damage, squad formations, fast units' spots, siege reach near the river.
- **S2b:** the bot's units land 0.55 of our units below its aim (1,791 plays): fixed (`AIM_DY`). Decision-to-field delay
  isn't measurable from clock reads (the two clocks differ by 3-5 s); open.
- Estimate was 2-3 days; took ~1 day of session time.

## 2026-10-24 · Simulator S3: the search fails its gate (stopped here, as agreed)
- **Prerequisites done:** `matches.result` for 585 of 586 replays (574 from the end screen; 20 / 20 by hand on
  held-back). Win model `sim_value_v1` (`bot77.sim.value`, models registry role sim_value): **weak**. Held-back log loss
  0.686 vs 0.693 for a constant, AUC 0.58 (crowns + clock alone 0.58). The default boosting settings memorised games
  (382 train games behind 105k moments; log loss 0.750, worse than a constant), hence heavy regularisation. Why weak:
  these pro games are close; only ~15% of moments after 150 s have a crown difference, and most games end on the
  first tower to fall in overtime, which the state features can't see coming.
- **Search** (`bot77.sim.search`): each card in hand x legal spots on a 1.5-tile grid x timings (now, +0.5, +1, +2,
  +3 s) + wait, refined around the best spots; 10 s rollouts with the opponent's actual next plays; score = change in
  win chance. ~3,000 candidates in ~1 s per moment.
- **Verdict margin, set from the pros before scoring Cole's plays:** 80th percentile of (best - the pro's actual play)
  over 300 train-split pro moments = **0.178**.
- **Gate (91 of 107 reviewed plays scoreable; 16 have no paired replay):** verdict agreement **37.4%** (34 / 91; gate
  >= 65%, Claude's pre-review 59%); clean set **4 / 14** (gate 12 / 15; one of the 15 has no replay); Cole's fix in the
  top 2 **38.3%** (23 / 60; gate >= 50%); timing notes saying "later": the search's best delays on 0 of 3. **Failed.**
- **No signal, not a bad margin:** the gap (best - the bot's play) has median 0.054 on plays Cole called mistakes and
  0.065 on plays he called OK. No margin separates them (margin 0 = "everything is a mistake" = 65.9%, only because
  60 of 91 are mistakes).
- One measurement bug fixed after the first run, without changing the margin: the bot's taps just behind a princess
  tower are illegal in the engine (the game places the card at the nearest free spot); they're now snapped like the
  game does. First run (59 scored): 40.7%; after the fix (91 scored): 37.4%.
- **Not changed, per the plan ("report where it disagrees, with the frames, before changing anything"):** the 57
  disagreements are on a local page (`scripts/sim_disagreements.py` -> data/reviews/sim_s3_v1/index.html) for Cole.
  Likely causes, in order: the win model is close to blind over a 10 s window; the sim's fights (S2: winners right
  58-62%, tower damage often missed); the opponent's actual plays answer what really happened, not the candidate.
## 2026-10-24 · Overnight: simulator fixes, self-play test, more pro replays (Cole: "do 1 2 and 3")
1. **Simulator fixes, measured on the replays** (engine fork, tests per fix):
   - Card audit vs RoyaleAPI level 11 (docs/simulator-card-audit.md, `get_card_stats`). Fixed: spells hit crown towers
     for their published fraction (Fireball / Arrows / Zap / Lightning / Freeze / Snowball 30%, Rocket 25%, Log 20%;
     were 100%), the Miner 25% (replays: ~20%); squad formations fitted on 200 replays (Skeletons get depth; Royal Hogs,
     Wall Breakers, Guards, Skeleton Dragons, Bats wider). Leads not acted on: 2023 data vs the fork's 2026 balance.
   - **Tower level, by measurement:** a lone Giant / Knight / Royal Ghost hit takes 0.0832 / 0.0665 / 0.0867 of a
     princess tower on the replays; engine damage / 3052 = 0.0829 / 0.0662 / 0.0855 (RoyaleAPI's 3584: 0.071-0.073).
     Towers now default to level 11 at 3052 / 109.
   - **tower_hp fixed:** per-side max (the bot's games have different tower levels), bars re-read every 5 s, the HP
     is the largest number on the bar not above the tower's HP, a reading contradicted within 15 s is rejected.
     Impossible mid-match drops 160 -> 19 (546 matches). Win model on it: held-back AUC 0.58 -> 0.62.
   - **The fight-winner metric's ceiling:** the replay agrees with itself 1.5 s earlier on only 69% of windows, so
     the sim's 58-64% is close to what the metric can show; S2's ~80% expectation wasn't reachable with it.
   - Held-back after the fixes: tracks unchanged (0.99-1.21 at +5 s); big tower damage, sim / real median 0.50 (own)
     / 0.86 (opp) from near-empty starts (was 0.30 / 0.55); empty starts still poor (n 9-11).
2. **Self-play test:** the fork's PPO stack for our deck (scripts/selfplay_train.sh, supervised), 02:09-07:15:
   25,950 episodes against its scripted teacher on the 11-rung ladder (16 meta decks), all in phase 1. Rung 0 for
   the first ~10k episodes at 25-40% (the cold start the fork's notes describe), then rungs 1-5 by 07:00 (rung 5 = a
   4 s lookahead teacher; 14-44% there). Then **20 real games** with it (`live_session.py --policy selfplay`,
   bot77.live.selfplay: the screen mirrored into an engine once a second; towers and unit HP not mirrored; taps with
   the new AIM_DY): **10 W / 10 L (50%), trophies 606 -> 614** (peak 664). Rules bot, session 2: 47%; before the
   rules: 36%. 20 games is a small sample (50% +- 11). Its 20 Battle Log replays were pulled for analysis.
   Gotchas: SIGINT is ignored under nohup, so the deadline stop never landed (a checkpoint from 07:15 was used);
   the "Exit Clash Royale?" dialog appears if Back is pressed on the home screen.
3. **More pro replays:** TV Royale recorder fixed (ADB device; scrcpy ignored SIGINT under nohup and left orphans);
   ingest ~5-6 min per replay under load = recording speed: ~10-11 replays an hour on one Mac. **55 TV Royale
   replays** recorded overnight (Ultimate Champion first); being ingested and rebuilt into the tables
   (scripts/after_session_2026-10-24.sh). The ceiling is real-time playback on one emulator (~250 a day at most).
## 2026-10-26 · Option 3 step 1: the imitation agent in real games (20 games)
- **Agent:** bc_v2 (scripts/train_bc.py: the fork's MicroRoyaleNet cloned from 966 pro trajectories, 483 games;
  held-back card at plays 40%, play/wait AUC 0.755, cell 4.1 tiles), driven by bot77.live.selfplay, sampling. Cole's
  new deck (Giant, Battle Ram, Mini P.E.K.K.A, Archers, Goblin Hut, Fireball, Arrows + one 3-elixir card the hand
  reader has no art for; never played).
- **Result: 7 W / 13 L (35%)**, Oct 25 23:37 - Oct 26 01:20. For comparison: self-play agent 50% (20 games) then
  ~36% (afternoon, 22 games); rules bot 47%. Small samples (+-11) and a different deck, so not a clean comparison.
- **What the games showed was mostly plumbing:**
  - The agent decided every ~4.5 s, not every 1 s as it was trained: `adb screencap` takes 3.5-5 s on BlueStacks
    (raw or PNG alike), and the hand reader took 1.2 s a read (it never learned the 7-known-card deck, so it matched
    every card). Fixed the reader (read against the 7 known: 0.12 s) and added `--stream` (bot77.live.stream:
    Android screenrecord H.264 over ADB, ~6 frames/s). When it did decide, it chose to play ~50% of the time: it
    wasn't holding elixir on purpose.
  - The new "Play Again + OK" end screen wasn't recognised (stuck 4 min): matched by its buttons now.
  - Results were logged "?": OCR misses the stylised "Winner!". Matched by its bright lettering now (yellow, cyan or
    white by who won); 102 / 102 past end screens with known results right.
  - Two scrcpy recorders from the stuck run were still recording an hour later (killed).
- Next: confirm `--stream` gets one decision a second, then step 2 (RL from bc_v2 against pro clones).
- **`--stream` check (2 games, 01:02-01:09):** one decision every 1.3 s (median; 90th percentile 1.5 s) instead of
  ~4.5 s; p(wait) 0.70 on average; 25 plays a game (the screenshot games: 3-25). 1 W / 1 L. Stream lag from a tap to
  the change on the stream: 0.14 s. Its speed exposed a navigation race (home seen mid-animation, Battle tapped onto
  the arena, which opened the trophy road): Battle is now tapped once home has held for 1 s, and a game that never
  starts is given up after 45 s. Trophies 600 -> 630 over the 20 test games (read before the 2 stream games).
- **The imitation agent at full speed (2026-10-27, `--stream`, all 8 cards readable): 8 W / 9 L (47%) in 17 clean
  games**, 24-41 plays in most games; one decision every 1.3 s (median over the run). Slow-pace games (Oct 26): 35%.
  Rules bot: 47%; self-play agent: 50% then ~36%. 17 games: +-12, so "about the rules bot", not better.
  Games left out (bugs, not the agent): 6 from the first stream run (a scrcpy recorder alongside stalled the stream
  30 s+ at game start) and 6 more (stand-ins for unreadable slots pushed a real card out of the agent's deck and it
  stopped deciding mid-game; 4 cards played on one stale frame while the stream restarted). Also fixed on the way:
  Mega Minion art override (the in-game art no longer matches the API icon), the trophy road's OK (template), the
  learned deck kept through the end screen, the result read up to 4 s after the end screen appears.
## 2026-10-27 · Option 3 step 2: RL against pro clones
- **Checked first: did the imitation data lose Mega Minion?** No. In the pro replays the card reads fine (the replay
  viewer draws the art the reader knows): the 38 player-sides that play it have it in hand on 26,389 reads, with
  unreadable slots no more often than everyone else (29.8% vs 31.3%), and their plays reach training with a card
  label as often (4.8% unknown vs 5.2%). The live-screen art change was a BlueStacks-only problem. Nothing to fix.
- **The setup** (scripts/rl_vs_clones.py, bot77.rl.clone_env): the trainee starts from bc_v2 and plays Cole's deck
  (Giant, Battle Ram, Mini P.E.K.K.A, Mega Minion, Archers, Goblin Hut, Fireball, Arrows); team 1 is a frozen bc_v2
  playing a pro's deck, one of 750 from the train-split replays (scripts/pro_decks.py; Heroes as their plain card and
  Champions without abilities, as the clone has no ability heads; 126 decks with a card the engine lacks left out).
  The loop is the fork's phase-2 trainer, subclassed (its python_ai/ untouched), without its league. Two changes for
  fine-tuning from imitation: a critic warm-up (only the value and auxiliary heads learn for the first 500 episodes:
  the untrained value head made the first update clip 57% of samples) and lr 1e-4 (fork: 3e-4). Smoke run (200
  episodes, 4 workers): clip fraction 0.14-0.24 after the warm-up, 45% wins against the clone.
- Run: 20,000 episodes, 7 workers, started 2026-10-27 ~14:15. Real test after it: 20 `--stream` games vs the
  imitation agent's 47%.
- **Progress to 10,000 episodes** (2026-10-27, paused at ~10,000 for the real test): win rate against the clone
  45% at the start -> ~55% (3k) -> 66-74% (4k-8k) -> 78% (10k); noisy (100-episode window swings ~+-10). The trainer's
  "losing trend" re-boost fires on that noise below 0.50, since against a copy of itself ~0.50 is the null.
- **What it changed** (12 sampled games per checkpoint, no scenarios): it drops the expensive cards. Plays per game,
  bc_v2 -> 2.5k -> 6k episodes: Giant 1.2 -> 0.2 -> 0.6, Goblin Hut 2.7 -> 3.0 -> 1.5, Fireball 1.6 -> 1.1 -> 0.6;
  Archers, Mega Minion, Arrows, Battle Ram, Mini P.E.K.K.A up. Distinct cards per game 7.3 -> 6.2. The fork's own notes
  say the engine undervalues big units and Fireball (median catch one unit), so this may be a simulator bias; left
  alone (forcing card use cost the fork's runs win rate) until real games say.
- **Drift from the pros** (held-back pro trajectories, 40, scripts/train_bc.py evaluate): card at plays 38.8% (bc_v2)
  -> 37.7% (8k), play/wait AUC 0.745 -> 0.696, placement 4.1 -> 6.0 tiles from the pro's cell. Card choice barely
  moved; timing and placement moved to what the simulator rewards. 8,000 episodes is ~1.2M decisions, ~6x the
  imitation data.
- **Real test:** 20 `--stream` games with the 8,000-episode snapshot, training paused (running both slowed nothing
  measured yet, but the live bot's decision pace is what spoiled earlier tests). +-11 points at 20 games: it can show a
  big change against bc_v2's 47%, not a small one.
- **On LanceDB** (Cole asked): the option-3 loop uses it as an embedded table store (filtered reads, Arrow, appends,
  binary columns: ingest -> imitation data, the pro deck pool, the live bot's game log). Vector search is used by the
  detector's crop mining / auto-labelling (bot77.mine, bot77.autolabel), not by option 3. A hosted store (turbopuffer)
  would break "no game frames to third parties" for the crop tables.
- **Detector at scale (Cole: what would a near-perfect detector take?).** Measured: real labelled data is 18,157
  frames / 329,312 boxes / 219 classes (+48,004 synthetic); median 173 boxes per class, 64 classes <= 100, 6 with none.
  V7 / V8 (more mined + live data on top of V5) gained nothing, so data volume isn't the current limit: label quality
  (spot-checks 59-72% of boxes right), the 4 fps detection rate (Zap 0/27, Arrows 0/16: ~0.3 s effects fall between
  frames) and the rare-card tail are. Tail arithmetic from the 526 replays (a card appears in a median 7.4% of games,
  ~3.7 plays when present; rarest 0.76%): ~1,000 distinct plays per card needs ~3,600 games (median card), ~7,100
  (90% of cards), ~36,000 (every card seen); at ~250 recorded replays a day, 2 weeks / 4 weeks / 5 months. 7,100
  games at one deduplicated frame a second is ~1.5M frames, ~200-350 GB (130-250 KB a frame). LanceDB fits the
  curation (dedupe by embedding, class gaps, hard-example mining, versioned label sets); Ultralytics trains from image
  folders, so curated versions would be exported. Not now: better labels, faster detection around spells and
  targeted recordings of thin classes come first, and only if the rebuilt board proves to be what costs games.
- Real test, RL 8k snapshot (results re-checked with the crowns): **13 W / 5 L in the first 18 clean games (72%)**;
  2 replacement games running. 1.3 s decisions.
  Games 6 (bot_2026-10-27_154333) and 12 (bot_2026-10-27_160053) excluded: Wi-Fi dropped, Cole restarted Clash mid-game
  (data/live/excluded_games.json); 2 games are added at the end to keep 20 clean.
- **Direction (Cole, 2026-10-27):** no new detector for now; push the RL as far as it goes, then a video, then maybe
  crowdsourced replays. Plan: finish this real test; add the trainee's own past snapshots to the opponent pool with
  the clone (the league the fork already has, switched off here; the plateau at ~72-78% against the one clone looks
  like learning that copy's weaknesses); 20-40k-episode rounds; 20 real `--stream` games per promising snapshot; keep
  recordings, training curves, card-use and drift numbers for the video.
- **Result reading fixed: crowns, not "Winner!".** Game 6 above (excluded) was a 3-crown loss logged as a win: the
  "Winner!" lettering had faded out of the end-screen capture and its match landed lower on the screen. Crowns
  (`device.crowns`: gold vs cushion in each banner's three slots) agree with 149 of 159 recorded results; the 3
  disagreements were all that misread (losses logged as wins), and 7 crown ties fall back to "Winner!". Corrected:
  bot_2026-10-26_004442 (slow-pace imitation test: **7-13 -> 6-14, 30%**), bot_2026-10-27_121850 (already excluded),
  bot_2026-10-27_154333 (excluded). The full-speed imitation tally (8-9) is unchanged. The RL test in progress
  started before the fix, so its results are re-checked with the crowns at the end.
- **RL 8k real test, final: 13 W / 5 L in 18 clean games (72%)** (2 Wi-Fi games excluded; the 2 replacements were
  dropped when the deck changed). Imitation agent: 8 W / 9 L (47%). The gap (25 points) has a 95% interval of about
  +-32 at these sample sizes, so likely better, not proven; the two tests ran on different days and trophy counts.
- **Deck change (Cole, 2026-10-27): Bomber for Archers** ("we were facing a lot of splash"; "the bot should learn to use
  different decks"). Every result above is with Archers; from here on, Bomber. Compare across the change with care.
- **Round 2 (rl_clones_v2):** from rl_clones_v1's checkpoint (~10,100 episodes), deck Giant, Battle Ram, Mini
  P.E.K.K.A, Mega Minion, Bomber, Goblin Hut, Fireball, Arrows; half the episodes against the clone (pro decks), half
  against past snapshots (v1's and its own, PFSP-weighted, playing our deck); critic warm-up 300; 20,000 episodes.
  Smoke test: 40 resets drew the clone 24 times and 5 different snapshots 16 times. (The league code went in with
  commit fb0426e, swept up by a `commit -a`.) Also fixed today: an OK button found by its white lettering on any
  colour (a nudge opened Choose Decoration: 30 min stuck).
- Round 2 at ~2,100 episodes, win rate by opponent (the trainer's EMA estimates, from the checkpoint): clone 0.60 (741
  games; round 1 was ~0.78 against it with Archers, so the deck change costs for now), round-1 snapshots 0.57 (10k),
  0.57 (8k), 0.60 (6k), 0.69 (4k), 0.72 (2k): it beats older versions more, as expected.
- Round 2 at ~3,000 episodes: distinct cards per game 5.98; plays per game (12 sim games vs the clone, Bomber deck):
  Giant 0.2, Goblin Hut 0.5, Fireball 0.3, Bomber 5.0, Battle Ram 4.3, Mini P.E.K.K.A 4.2, Mega Minion 4.5, Arrows 3.9
  (11/12 wins). Effectively a 5-card deck; the rare cards still get cycled out now and then. The 8k snapshot that went
  13-5 in real games already leaned this way (Giant 0.6, Hut 1.5, Fireball 0.6). Left alone until the real test.
- Round 2 at ~4,650 episodes (Bomber): clone 0.70 (1,621 games; 0.60 at 2k), round-1 snapshots 0.79 / 0.81 / 0.70 /
  0.68 / 0.57 (2k / 4k / 6k / 8k / 10k), its own round-2 snapshots 0.55 / 0.52 (near-copies: ~0.5 is the null).
  Distinct cards per game 5.8-5.9.
- Round 2 at ~8,800 episodes: clone 0.73 (3,045 games), round-1 snapshots 0.89 / 0.86 / 0.77 / 0.75 / 0.67
  (2k-10k), its own round-2 snapshots 0.61 / 0.54 / 0.54 / 0.53 (2k / 4k / 6k / 8k). Every older version is beaten
  more than before; distinct cards per game 5.8-6.0.
- Round 2 at ~11,900 episodes: clone 0.67 (4,151 games; 0.73 at 8.8k, within the estimate's swing), round-1
  snapshots 0.91 / 0.87 / 0.75 / 0.75 / 0.70, round-2 snapshots 0.69 / 0.67 / 0.62 / 0.56 / 0.56 (2k-10k): the margin
  over every older version keeps growing. The 54% headline at 12k was the short window on the newest opponents.
- Card use at ~13k (20 sim games vs the clone, 17 wins): Giant 0.3, Fireball 0.5, Goblin Hut 0.8 plays per game;
  Arrows 3.5, Battle Ram 3.9, Mini P.E.K.K.A 4.2, Mega Minion 4.2, Bomber 4.5. A five-card cycle; the three
  4-5 elixir cards are mostly cycled out.
- **Hand clog (Cole: "does it just not use Giant?").** Share of decisions with the card in hand (10 sim games vs the
  clone, 9 wins, ~14k): Fireball 93%, Giant 91%, Goblin Hut 80%; Arrows 41%, Battle Ram 34%, Mini P.E.K.K.A 26%,
  Mega Minion 20%, Bomber 15%. The three rarely-played cards hold three of the four slots, so the agent mostly plays
  whatever reaches the one free slot: a one-slot cycle that wins in the simulator. Options put to Cole: test it for
  real first (recommended), shape training against holding cards, or change the deck.
- Not new: the same measure (10 sim games each, Bomber deck): in hand Fireball / Giant / Goblin Hut = imitation 66 /
  62 / 54%, round 1 at 8k 87 / 87 / 72%, round 2 at ~14k 93 / 91 / 80% (an even split is ~50%). The 8k snapshot that
  went 13-5 in real games already clogged its hand; round 2 goes slightly further. Training was never restarted:
  bc_v2 -> round 1 (10.1k, Archers) -> round 2 (from round 1's weights, Bomber).
- **Real games disagree in degree** (Cole: "in game testing it always played giant"). The 8k snapshot's 18 clean
  real games (Archers): 33 plays a game; Giant 1.3, Goblin Hut 2.4, Fireball 0.9 per game; in hand Fireball 85%,
  Giant 77%, Goblin Hut 64%. Same snapshot in the simulator (Bomber deck): 23 plays, Giant 0.7, in hand 87%. So it
  does play the Giant about once a game, but holds it most of the time; the lean is milder in real games (longer,
  more plays). Card use is measured on real games from here on.
- **Round 2 finished** (20,000 episodes, ~7.5 h of training; last checkpoint at 19,575, copied to
  rl_clones_v2/final.pth): clone 0.74 (6,859 games), round-1 snapshots 0.93 / 0.85 / 0.78 / 0.75 / 0.72 (2k-10k),
  round-2 snapshots 0.74 / 0.69 / 0.56 / 0.57 / 0.61 / 0.58 / 0.57 / 0.54 / 0.55 (2k-18k). Distinct cards per game
  ~6.0 throughout. Next: the real test with Bomber (check the hand reader reads Bomber first).
- **The training scenarios, measured** (Cole asked how it wins them). The fork injects 30% of episodes into one of
  four (python_ai/envs/scenarios.py): bridge_push (a win condition on a bridge, 15 s), bridge_push_supported (+ a
  ranged support, 25 s), fireball_swarm (3-5 cheap bodies in our half, 12 s), fireball_tower_value (2-4 enemies on
  their tower, 15 s; they walk over). Final round-2 model, 25 runs each, engine towers 2,534 / princess: our tower HP
  lost (median) 1,480 / 3,680 / 2,534 / 1,149; first card, bridge push: Battle Ram 9 of 25 (a counter-push, not a
  defence); Fireball first in 2-3 of 25 even in the two Fireball scenarios. It answers at once but often with the
  wrong card. The log's ScenDef ~0.98-1.00 only checks the episode reward stayed above -0.5, which a big hit in a
  short window rarely trips: it hid this. Lever for round 3: weight the scenarios up and judge them on tower HP lost.
- **Openings (Cole: "every first play I've seen is just a bum rush").** Real games: round 2 (Bomber, 15 games) opens
  ~5 s in at ~8 elixir with Mini P.E.K.K.A 6 / Battle Ram 5 / Mega Minion 2 / Bomber 1 / Arrows 1, mostly at the
  bridge; the 8k snapshot (Archers) opened with Mega Minion 5 / Goblin Hut 4 / Mini P.E.K.K.A 3 / ... Round 2 got
  more aggressive. **Next (Cole):** design scenarios together to punish the rush and teach defence and counter-attack:
  per scenario the setup (units, positions, elixir, hand), what good play looks like (Cole), and scoring on tower HP
  lost / elixir trade / damage after the defence (not the fork's reward > -0.5). Built in bot77.rl.clone_env, the
  fork untouched.
- **Real test, round 2 final (Bomber deck): 9 W / 9 L (50%) in 18 games**, stopped early by Cole ("we've seen
  enough"); every result confirmed by the crowns; the reader read Bomber from the first game; 1.3 s decisions. Against
  the 8k snapshot's 13-5 (Archers) it looks worse, but the deck changed and the 13-5 run raised the trophy count (so
  stronger opponents); both samples are small. Simulator win rates against the clone (74% vs ~75%) didn't predict the
  difference. Next: the defence / counter-attack scenarios with Cole.
## 2026-10-28 · Defence situations, phase one (Cole's scenarios 2, 3, 4, 7)
- bot77.rl.situations: hog_rider (Hog at the bridge, 4-6 elixir, 15-20 s), battle_ram (16-19.5 out, 4-6, 20-26 s),
  melee (Mini P.E.K.K.A / Knight / Prince, 3-5, 15-20 s), supported_push (Giant + Musketeer / Wizard / Executioner /
  Minions near the bridge, or a Golem + Wizard / Musketeer from the back; 6-8, 25-40 s). Lane, positions, elixir and
  hand order randomised; the hand always holds an affordable answer (and a support answer on the supported push); the
  enemy is already moving (deploy 0); runs until the threat is dead (window minimum, timeout maximum). Score =
  -(tower HP lost / princess max) - 0.04 x elixir spent [+ 0.02 x surviving defenders' elixir, supported push]
  - 0.3 if unresolved; no credit for enemy tower damage. On a situation's steps the game's shaping is zeroed and its
  end is final (no bootstrap, no draw penalty); the fork's own scenarios are off when ours are on.
- **Exam** (scripts/situation_exam.py, 20 held-out seeds per template; engine towers 2,534; the clone keeps playing
  during the window, so states aren't identical across players). Scores (higher is better):
  hog nothing -1.23 / rush -1.38 / scripted defence -0.61 (3.6 elixir) / round-2 model -0.86 (8.7 elixir);
  battle ram -1.26 / -1.43 / -0.93 (3.6) / -1.40 (10.8); melee -1.23 / -1.42 / -0.70 (2.5) / -0.79 (8.2);
  supported push -2.81 / -3.06 / -1.89 (6.3) / -1.94 (14.7). Cole's check holds (a sensible defence beats rush and
  doing nothing on all four); the model often saves more tower HP but spends 2-4x the elixir.
- Smoke (situations only, 4 workers, ~340 episodes): runs; critic loss 0.68 -> 0.10 as the value head learns the
  new scores. Waiting on Cole's review of the scripted behaviour before the round.
- Situation replays (scripts/situation_replays.py) published for Cole as a private page (top-down, scripted defence vs
  the round-2 model from identical starts). Seen in them: the model answers a Battle Ram with 9-13 elixir (once with its
  own Ram plus three cards) and still loses most of a tower; the scripted defence is too simple for the supported push
  (one tank answer + one support answer, then nothing: 4,800-5,800 tower HP lost) — that template needs Cole's
  reference defence; a Hog that did 243 damage but outlived the timeout was marked unresolved (the rule may be harsh).
- **Situations end at neutral** (Cole: "the opponent isn't gonna just let the mini pekka take tower"): a situation now
  ends as soon as the threat is dead and no enemy unit is on our side (3 s minimum, the template's timeout maximum);
  the old minimum windows let surviving defenders walk over and hit an undefended tower. Surviving defenders count
  (+0.02 / elixir of value, cap 10) in every template. Counter-attacking stays its own later scenario (Cole's 8).
  Re-recorded replays: situations now end 5-14 s in (supported Golem still runs to its 40 s limit).
- **Scripted defence fixed** (Cole: "our scripted defense is ass?" on Battle Ram 2). It picked the first answer in hand
  order and met a Ram with a lone Mega Minion. Now one shared `situations.scripted_defence` by priority: Mini
  P.E.K.K.A first against Hog and Ram (Goblin Hut first at x 8.5, y 9 failed to pull the Hog: placement for Cole),
  Bomber on the Ram's Barbarians, Mega Minion first against Prince / Mini P.E.K.K.A, Bomber first against Knight.
  Also a bug in the neutral rule: the Ram's Barbarians appear ~1 s after it breaks and the board looked empty for that
  second (one model run ended at 5 s, its damage missed): neutral must now hold 2 s. Replays now: Hog -0.12 / -0.08,
  Ram -0.35 / -0.12, Prince -0.68, Knight -0.44; the supported push still fails (-2.1 / -3.2, unresolved): Cole's
  reference defence needed. A Mega Minion stood idle in one replay; in a re-run it attacked (watching for it).
