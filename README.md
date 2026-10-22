# bot-77

A Clash Royale AI built from gameplay footage, in three steps:

- **C — analysis engine.** Turns recorded games into a play-by-play event log in LanceDB: both players' cards, times and
  positions. ([docs/play-detector-spec.md](docs/play-detector-spec.md), [docs/opponent-plays-spec.md](docs/opponent-plays-spec.md))
- **A — move predictor.** Learns from strong players' games what a player plays next: which card (built), when and
  where (next). ([docs/move-predictor-spec.md](docs/move-predictor-spec.md))
- **B — the bot.** Plays in BlueStacks using A, and records every game so its own games become training data.

Every decision, measurement and review is logged in [decisions.md](decisions.md). Related work: KataCR (Wu et al. 2025,
arXiv:2504.04783) — the unit detector and spell classifier build on its approach.

> **Account warning.** An automated player breaks Supercell's terms of service; accounts that run bots can be banned.
> Run the bot only on a throwaway account, never your main one.

---

## Setup

```bash
uv sync                                   # Python env (Python 3.12, uv)
brew install scrcpy                       # records the Android screen
```

- **BlueStacks** (Mac): Settings → Advanced → turn **Android Debug Bridge (ADB)** on. The bot drives the game through
  ADB (`/Applications/BlueStacks.app/Contents/MacOS/hd-adb`), so your mouse and screen stay free. The device is
  usually `127.0.0.1:5555` (older installs: `emulator-5554`).
- **Data stays local.** Game footage, frames, models and LanceDB tables live under `data/` and are gitignored.
  Cold videos can live on an external drive behind symlinks (see decisions.md, "Storage").

---

## Running the bot (B)

### 1. Calibrate once per screen setup

Start a battle in BlueStacks yourself (any game mode), then while it runs:

```bash
uv run python scripts/live_calibrate.py
```

It takes ~15 s of screenshots, derives the battle layout (`config/layouts/bluestacks_live.json`), fits the arena to
the towers, prints what it reads (hand, elixir, clock), and saves `data/raw/live_calib/bluestacks_live_check.jpg` with
every region drawn — check the boxes sit on the cards. If a card in hand never reads (its in-game art changed), cut an
art override from a frame (`data/cards/art_overrides/<Card>__normal.png`, see decisions.md 2026-10-21).

### 2. Play

```bash
uv run python scripts/live_play.py --games 1                 # one game — watch it the first time
uv run python scripts/live_play.py --games 40 --hours 4      # a session
```

The bot taps **Battle** from the home screen (or joins a battle already on screen), plays, taps through the end
screen and rewards, and repeats.

**Stop it any time:** `touch data/live/STOP` (it finishes the current step and exits).

It also stops by itself after `--games` / `--hours`, when screenshots keep failing, or when a screen stays
unrecognised for 4 minutes (it never taps blindly: between games it only presses buttons on an allow-list — Choose,
OK, Claim, Continue, … — never anything about gems, buying, offers or clans; otherwise it taps empty space or the
Battle tab).

**What it saves, per game** (`data/raw/live_games/<game id>/`): `game.mkv` (screen recording), a frame at every play
and 2 s after it. Every play and game is logged to LanceDB (`bot_actions`, `bot_games`). Screens it didn't recognise
between games go to `data/raw/live_ui/unknown/`.

### 3. Pull the replays (answer keys)

The Battle Log keeps the last 25 games, so pull them during long sessions (the session script does this every ~20 games):

```bash
uv run python scripts/replay_runner.py -n 20 --no-known --device 127.0.0.1:5555   # record the newest 20 replays
uv run python scripts/ingest_replays.py                                           # process them into answer keys
```

A replay shows both hands, so each one is an exact play-by-play of both players — the training data for A and the
truth for C.

### 4. Review the bot's plays

```bash
uv run python scripts/build_bot_review.py --since 2026-10-21    # -> data/reviews/bot_<date>/review_<n>.html
```

Each card is one bot play: the frame at the play (the bot's tap in yellow) and 2 s later. Mark **OK** or **Mistake**;
for a mistake, click where the card should have gone, pick the card it should have played (or "wait"), add a note.
**Export** → JSON. These become rules in the playbook (below).

### Offline checks (no BlueStacks needed)

```bash
uv run python scripts/live_shadow.py cole_2026-10-20_b_m03          # the bot's choices on a recorded game vs what was played
uv run python scripts/live_dryrun.py data/raw/replays/2026-10-20/replay_24.mp4 --layout replay_bottom   # the whole loop on a video
uv run pytest tests/test_live_policy.py                              # placement rules and tap geometry
```

---

## Architecture

```
            BlueStacks (ADB)                      LanceDB (data/lancedb)
                 │ screenshot ~0.3 s                    │
                 ▼                                      │
  ┌─────────────────────────────┐                       │
  │ state   (bot77.live.state)  │  hand, next, elixir,  │
  │  hand / elixir / clock read │  clock, enemy units   │
  │  unit detector (V5)         │                       │
  │  opponent-play tracker      │                       │
  └──────────────┬──────────────┘                       │
                 ▼                                      │
  ┌─────────────────────────────────────────────────┐   │
  │ policy  (bot77.live.policy)                     │   │
  │  1. rules file   config/playbook_rules.yaml  ◄──┼── Cole's reviewed mistakes        (planned)
  │  2. playbook     answer to the opponent's    ◄──┼── `playbook` table: mined from   (planned)
  │                  last play, placed relative     │   471 keyed replays
  │                  to the threat                  │
  │  3. A (move predictor) — which card otherwise ◄─┼── data/models/moves_v1 (built)
  │  when / where: rules today (learned later)      │
  └──────────────┬──────────────────────────────────┘   │
                 ▼                                      │
  device (bot77.live.device): tap card → tap spot ──────┼──► bot_actions, bot_games, frames, recording
                                                        │
  after a session: replays → keys → retrain C / A / playbook
```

**Today (built):** the policy decides *when* by rules (defend when enemy units are on our half; otherwise at full
elixir, or after a quiet spell), *which card* with A v1 (a gradient-boosted ranker over the 4 cards in hand: 44% exact
on held-out test games vs 23% random), and *where* by rules per card role (win conditions at the bridge, buildings in
the centre, defenders in front of the threat, spells on the biggest enemy group).

**Planned — the playbook** (not built yet):

1. **`interactions` table (LanceDB).** From both players of every keyed replay: each opponent play and the other
   player's next play within 8 s — card, delay, and placement relative to the opponent's unit (positions from C's
   deploy detections).
2. **`playbook` table.** Aggregated per opponent card: the most common answers, their share, typical delay and
   placement; filterable to top-ladder games. The bot looks it up when the opponent plays something (~1 ms): the
   best answer that is in hand and affordable is played at the playbook's position relative to the threat; otherwise
   A decides.
3. **Rules file** (`config/playbook_rules.yaml`). Short forbid / prefer / place rules written from Cole's reviewed
   mistakes, checked before every play. Readable, easy to delete.
4. **Next for A:** similar-moment search — every moment of the replays stored as a vector (board, elixir, hand,
   clock); at play time LanceDB returns the most similar moments from strong players' games and what they did next.

**The loop:** the bot plays → its games are recorded and their replays pulled → keys made by the same pipeline as
Cole's games → C (detector, spell model, fake filter), A and the playbook retrain on them → the bot plays better.

---

## The other steps

| Step | Entry points |
|---|---|
| C: process a video | `uv run bot77 process --video <file> --layout <id>`, `scripts/run_all.py <jobs.json>` |
| C: opponent plays + scoring | `scripts/eval_dev.py`, `scripts/eval_v2.py` (benchmark), `src/bot77/opponent.py`, `src/bot77/playrules.py` |
| C: live games keyed by replays | `scripts/pair_live_replays.py` |
| C: failure mining | `scripts/build_failures.py`, `scripts/mine_failures.py`, `scripts/train_fake_filter.py` |
| A: train / evaluate | `scripts/build_moves.py` → `scripts/train_moves.py` |

Tests: `uv run pytest`.
