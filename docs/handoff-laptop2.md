# Handoff: the second laptop records TV Royale 24/7

**Goal (2026-11-17):** grow the pro-replay set from 484 matches to about 5,000. v10 overfits after one epoch on what
we have, so more data is the lever. The second laptop records TV Royale (top-ladder games) around the clock onto the
LaCie. The main laptop later reads the recordings into training data.

- **Rate:** replays play in real time, about 10 an hour, so 200-250 a day running 24/7. 5,000 takes 3-4 weeks.
- **Space:** about 180 MB a replay. The LaCie has about 900 GB free (cleaned 2026-11-17), so the full 5,000 fits.
  The runner stops on its own below 30 GB free.
- **Checkpoints:** retrain and score at about 1,000, 2,000 and 4,000 matches. Keep recording only while the score
  still climbs.

## What the second laptop needs

1. **Its own Android phone, not the Galaxy A16 the bot plays on.** BlueStacks can't run Clash since the 11-09 update
   (decisions.md, 2026-11-10), so recording needs a real phone. Best is a second Galaxy A16, because the screen
   templates were cut on one. Another phone needs Android 12L or newer, and its templates may need re-cutting.
2. **Its own Clash account.** One account can't be signed in on two devices. If this phone signs in as bot-77, the
   live bot gets kicked out. TV Royale shows the same top games to every account, so a fresh free account works.
3. **The LaCie, plugged in for the whole run.** The laptop's own disk has no room. Recordings go to
   `/Volumes/LaCie/bot-77-data/raw/tv_royale/<date>/`.
4. **A Mac, kept awake on power.** Closing the lid pauses everything unless it runs in clamshell mode on a display.

## Setup (once)

```bash
# tools
brew install scrcpy android-platform-tools uv
git clone https://github.com/cole-zoom/bot-77.git && cd bot-77
git checkout laptop2-tv-recording
uv sync

# phone: Settings > About > tap Build number 7x, then Developer options > USB debugging on; plug in, accept the prompt
adb devices                          # note the serial
adb -s <serial> shell wm size 1080x1920 && adb -s <serial> shell wm density 320   # the templates assume this
#   (undo later: wm size reset / wm density reset)

# screen templates + thumbnails of the 445 battles already recorded, so none get recorded again.
# laptop2_seed.tar is in the main laptop's data/ folder (gitignored); copy it over (AirDrop / USB), then:
tar -xf laptop2_seed.tar -C /Volumes/LaCie/bot-77-data/raw
```

Open Clash on the phone, signed in to the recording account, on the home screen.

## Settings (every terminal that runs it)

```bash
export BOT77_RAW=/Volumes/LaCie/bot-77-data/raw          # where recordings go
export REPLAY_UI=$BOT77_RAW/replays/_ui_a16              # the A16's screen templates
export DEVICE=<serial>                                   # from adb devices
export BOT77_MIN_FREE_GB=30                              # stop recording below this much free space
```

`BOT77_ADB` picks the adb binary. Without BlueStacks installed, the runner finds platform-tools' `adb` by itself.

## First test (do this before leaving it alone)

**TV Royale has not been recorded on the A16 yet.** The last TV recording was on BlueStacks (2026-10-24). The A16's
TV templates (`tv_title`, `tv_top_tab`) were cut on 11-13 but never run. Record two first:

```bash
uv run python scripts/replay_runner.py --source tv -n 2 --device $DEVICE
ls -lh $BOT77_RAW/tv_royale/$(date +%F)/   # expect replay_01.mp4 + replay_01_entry.png, ~150-250 MB each
```

Watch the phone while it runs. It should open TV Royale, go to the top league, press Watch, wait for the replay to
end, and press OK. If it gets stuck on a screen, take a screenshot (`adb -s $DEVICE exec-out screencap -p > s.png`)
and bring it back to the main laptop session to cut the missing template.

## Run 24/7

```bash
caffeinate -i -s bash scripts/tv_loop.sh 720 1800 60
#   720 h (30 days) max, a new pass every 30 min, all 60 leagues from the top down
```

- **Progress:** `ls $BOT77_RAW/tv_royale/*/replay_*.mp4 | wc -l`, and the log in `data/logs/tv_runner_<date>.log`.
- **Stop:** Ctrl-C, or `pkill -f tv_loop.sh; pkill -INT -f "scrcpy -s"`. Wait for the current replay to finish first if
  you can, since a cut-off replay is wasted.
- **If the LaCie gets unplugged:** the run errors out, and the next pass also fails until the drive is back. Restart
  tv_loop once it's plugged in. A replay cut off mid-recording ends up as a short or broken file; the main laptop's
  ingest skips those.
- **Don't edit scripts while tv_loop is running.** Bash reads the file as it goes.
- **Watch for:** "no more battles in the list" on every pass means TV Royale isn't listing new games fast enough. Note
  how many new replays arrive a day. That number sets how long this takes.

## Back on the main laptop (with the LaCie)

When the LaCie comes back for a read (plan for around 1,000 total, then every ~1,000 after):

1. Link each new date folder into the repo:
   `ln -s /Volumes/LaCie/bot-77-data/raw/tv_royale/<date> data/raw/tv_royale/<date>`
2. `uv run python scripts/ingest_replays.py` (HUD truth; macOS only, Apple Vision OCR). Duplicates get split
   `'duplicate'` (bot77.replay_dedupe).
3. Read them through the live path into new shards (decisions.md, 2026-11-16 step 5: scripts/pod/pack.sh ->
   read_replay_boards.py -> build_bc_live.py). About 44 matches an hour on the laptop, or a 4090 pod.
4. Retrain from scratch on the bigger set, score with scripts/shadow_agreement.py, and log the curve in decisions.md.

## Rules that still apply

- Recordings stay local (LaCie). They are never committed. data/ is gitignored and the repo is public.
- The runner only navigates replay menus. It never plays a battle.
- Commit or push from the second laptop only through this branch; the main laptop's session merges it.
