# Giant Double Prince
Source: Double Prince with Giant! (Tips were Touched) · Sweaty Clash · 2025-09-26 ·
https://www.youtube.com/watch?v=VIy02pJXmtk (main, gameplay with commentary at ~15k trophies)
Also: The ONLY Guide Needed to MASTER Giant Double Prince · giodelion · 2025-10-15 ·
https://www.youtube.com/watch?v=p8Pflb3338M (short archetype explainer, no fixed list)

Placements are (x, y) in the engine frame: board 18 x 34, cells centred on integers. The bot (team 0) defends low
y, this opponent (team 1) defends high y; river band y 15.5-17.5. Lanes x 3.0 (left) / 14.0 (right), bridges
x ≈ 2.5 / 14.5, middle x 8.5. Opponent placements: "bridge" = (lane, 18-19), "back" = y 31-33, "pocket" =
(lane, 24-25), "middle" = (8.5, 24-25) as in `_tempo_opponent` (src/bot77/rl/situations.py). The bot's bridge is
y 14-15; its back is y ≈ 1-3.

## Deck
Giant 5 · Prince 5 · Dark Prince 4 · Baby Dragon 4 · Minions 3 · Goblin Gang 3 · Vines 3 · Zap 2 ·
avg 3.6 · 4-card cycle 11 (Zap, Goblin Gang, Minions, Vines) · sim: all 8 in the engine, costs as listed.

Sweaty plays Evo Baby Dragon (gives nearby allies a speed boost, Sweaty 0:00, 12:29) and Evo
Zap (Sweaty 2:08). The scripted opponent uses the normal forms. The list is read off the gameplay (the video has no
deck in its description); the captions mis-spell Goblin Gang. giodelion's archetype swaps in Electro
Wizard, Mega Minion or Musketeer and Fireball or Arrows for the flex slots (giodelion 1:42-2:17); we use Sweaty's 8.

## Game plan
Defend with the Princes, let them survive, then drop the Giant in front of them for a 10+ elixir counter-push
(giodelion 0:35, 2:52). Splitting is the other weapon: Prince in one lane and Dark Prince in the other, so one of
them connects (Sweaty 17:24). It is scared of being out-cycled into a Giant in the back with no elixir, of clumped
troops eating a big spell (giodelion 3:58), and of splash/swarm defence like Witch (Sweaty 22:21).

## Card roles
- Giant: win condition and tank; goes in front of surviving defenders or in the pocket to tank for the Prince (Sweaty 14:40).
- Prince: tank killer on defence (Giant, Valkyrie, Hog per giodelion 3:26) and the main damage dealer on offence.
- Dark Prince: splash defender vs swarms, shield; bridge pressure card (giodelion 3:26, Sweaty 8:07).
- Baby Dragon: air splash support; cycled in the back to start pushes (Sweaty 3:45, 20:08).
- Minions: cheap air DPS; shred an enemy Giant while our Giant kites (Sweaty 11:22).
- Goblin Gang: cheap defence, distraction and split-lane pressure (Sweaty 13:02, 14:06, 19:02).
- Vines: 3-elixir control spell, the strongest defensive card in the list per Sweaty (4:52, 22:21).
- Zap: clears swarms off the Prince/Giant, finishes spawns (Sweaty 2:08, 14:40, 21:48).

## Opening
- **Waits for the bot (Cole, 2026-11-04).** It never plays first before 0:30 of game time: heavy decks play reactive in single
  elixir. Until the first play of the game, DEFAULT is off and it sits at 10 elixir (leaking) rather than open. The
  moment the bot plays, it reacts as below. If the bot still hasn't played at 0:30, it makes the first play:
- First play at 0:30 with no info: Baby Dragon in the back ((7.5, 32) or (9.5, 32)) to cycle and start a push (Sweaty 3:45,
  20:08); alternatives seen are Goblin Gang (Sweaty 13:02) or Dark Prince at the bridge (Sweaty 10:16).
- If the bot plays first: answer it cheaply (Goblin Gang / Dark Prince / Prince on its tank) and counter-push.
- Never opens Giant in the back: only when ahead on elixir or in double elixir (giodelion 3:58). Never opens
  Prince + Dark Prince together (giodelion 4:31).

## Against our deck
Near-mirror: both are Giant beatdown. Its Prince is our Mini P.E.K.K.A, its Minions/Baby Dragon are our Mega
Minion, its Goblin Gang is our Bomber/Hut. **[inferred]** except where cited.
- **Giant:** ignore it until it is at the bridge, then Prince + Goblin Gang in the pocket ((lane, 24-25)) so the Giant
  takes Prince hits while gobs swarm; Vines if supported. Then Giant in front of the surviving Prince.
- **Battle Ram:** Dark Prince placed in front (Sweaty 3:12) or Goblin Gang in its path; the ram dies cheaply.
- **Mini P.E.K.K.A:** Goblin Gang on it (it one-shots gobs one at a time) or Vines to root it under tower fire.
  Never a lone uncharged Prince against it at close range.
- **Mega Minion:** Minions (3 for 3) or Baby Dragon; Zap does not kill it.
- **Bomber:** Dark Prince or Prince walks through it; never Goblin Gang near it.
- **Goblin Hut:** the Giant is pulled to it if the hut is centred; push the other lane, or Zap the spear gobs and
  let the Prince charge through.
- **Fireball / Arrows:** it spaces Baby Dragon away from Dark Prince (Sweaty 10:48; giodelion 3:58) and plays
  Minions/Goblin Gang after the bot's Arrows is out of cycle (Sweaty 2:40).
- **Our cards that hurt it:** Arrows (Minions + Goblin Gang), Mini P.E.K.K.A placed before the Prince charges,
  Goblin Hut (pulls Giant and Princes), Bomber + Mega Minion behind our own Giant (its air answers are Minions/Baby
  Dragon only).

## Punish rules (checked top to bottom, every tick)
"Bot elixir" is the opponent's count of it. "Committed in a lane" = elixir of bot cards played in that lane in the
last 6 s that are still alive.
1. IF the bot has ≥ 7 committed in one lane AND bot elixir ≤ 3 → defend that lane with the cheapest stop (Goblin
   Gang under the Giant at (lane, 24) + Vines on its support when it reaches the tower range), and within 1 s play
   Prince at the opposite bridge ((lane, 18.5)), followed by Goblin Gang or Dark Prince behind it ((lane, 21)) if own
   elixir ≥ 3 after the defence. Take some damage on one side, more on the other (Sweaty 0:32). **[inferred
   thresholds]**
2. IF the bot plays Giant in the back (y ≤ 4) in single elixir AND own elixir ≥ 5 → Prince at the opposite
   bridge immediately, Goblin Gang behind it when affordable (Sweaty 5:27; giodelion 3:58 calls a back Giant
   without an elixir lead a mistake). Defend the Giant at our bridge later with Prince/Goblin Gang only if its
   own Prince is still alive on offence, otherwise Goblin Gang + Vines.
3. IF the bot just played Mini P.E.K.K.A (seen, so out of hand for 3 more bot plays) AND own Giant or Prince is
   alive → push the other lane: Prince at the bridge, Giant in front if own elixir ≥ 5 (Sweaty 4:52-5:27: Giant
   baits the tank killer, then hit the opposite lane hard).
4. IF the bot just played Arrows (out of hand) AND own elixir ≥ 8 → Giant at the bridge (or back in double
   elixir) + Minions behind it 2 s later (Sweaty 2:40).
5. IF the bot plays Battle Ram → Dark Prince 2 tiles in front of it (lower y) in its lane (Sweaty 3:12); if Dark Prince
   is not in hand, Goblin Gang in its path.
6. IF a defence just ended with own Prince or Dark Prince alive at ≥ 50% HP on its own side AND own elixir ≥ 5
   AND (bot elixir ≤ 4 OR Mini P.E.K.K.A out of the bot's hand) → Giant 1 tile in front of the survivors (1 lower y)
   (giodelion 2:52; Sweaty 9:43).
7. IF bot elixir ≤ 2 for ≥ 1 s (it just spent) AND own elixir ≥ 4 → Dark Prince or Prince at the bridge of the
   lane with the lower-HP bot tower (Sweaty 11:54: the opponent is out of elixir, so go in). **[inferred
   thresholds]**
8. IF the bot's push is Giant + Mega Minion → Minions on the Mega Minion as soon as it crosses, Prince on the Giant
   at (lane, 24) after it locks the tower. **[inferred]**
9. IF the bot has Mini P.E.K.K.A in hand (not seen in its last 3 plays) AND bot elixir ≥ 4 → do not start a
   same-lane Giant + Prince push; cycle Baby Dragon / Goblin Gang instead (Sweaty 15:11: can't attack while the
   tank killer is up). **[inferred mapping: Sweaty's case was Inferno]**
DEFAULT:
- Single elixir: cycle (Baby Dragon back, Goblin Gang, Zap; Sweaty 3:45, 19:02), defend with Princes, counter-push
  per rule 6. Never both Princes on defence of one push (giodelion 4:31). Never Giant without support behind it.
- Double elixir: Giant in the back ((lane, 32)) at 10 elixir, then Prince + Baby Dragon behind it at the bridge (Sweaty 8:38,
  17:24); or split Prince one lane / Dark Prince the other (Sweaty 17:24, 21:14).
- Overtime: ahead on towers → defend only, cycle cheap cards; behind → split pushes every time elixir ≥ 9.

## Scenarios
- **ALL-IN PUNISH** · 0:30 · bot 9 / opp 9 → bot plays Giant + Mini P.E.K.K.A at the left bridge (bot 0) · board
  empty · opp hand Prince, Goblin Gang, Vines, Zap, next Giant · learn: in single elixir don't stack 9 in one lane
  without elixir left for the other; Giant alone + cheap support is enough · punished by rule 1 (Goblin Gang
  defends, Prince at the right bridge hits an undefended tower).
- **GIANT IN THE BACK** · 0:50 · bot 10 / opp 7 · empty · opp hand Prince, Dark Prince, Goblin Gang, Zap, next
  Minions · learn: in single elixir, at 10 elixir play cheap (Bomber/Hut) or Giant at the bridge, not 5 in the
  back · punished by rule 2 (Prince at the opposite bridge while the bot is at 5).
- **COUNTER-PUSH WINDOW** · 1:40 · bot 4 / opp 6 · opp Prince (80% HP) and Dark Prince just defended our Ram on
  its side · opp hand Giant, Minions, Vines, Zap, next Baby Dragon · learn: hold Mini P.E.K.K.A for the Giant-led
  counter-push; Arrows its Minions; don't spend on a second Ram · punished by rules 6/7 (Giant in front of the
  Princes, 13 elixir on our tower).
- **ARROWS SPENT EARLY** · 1:10 · bot 6 / opp 9 · bot just Arrowsed a lone Goblin Gang at the bridge · opp hand
  Giant, Minions, Prince, Zap, next Vines · learn: save Arrows for Minions/Goblin Gang behind a Giant, kill lone
  gobs with tower or Bomber · punished by rule 4 (Giant + Minions with no Arrows to answer).
- **SPLIT IN DOUBLE** · 2:20 (double) · bot 7 / opp 8 · empty · opp hand Prince, Dark Prince, Goblin Gang, Baby
  Dragon, next Giant · learn: against Prince one lane + Dark Prince the other, Mini P.E.K.K.A on the Prince and
  Bomber/Goblin Hut on the Dark Prince side, not both answers in one lane · punished by DEFAULT double split.

## Open questions for Cole
- Is Sweaty's 8 (Evo Baby D, Evo Zap, Vines, Goblin Gang, Minions) the list you see most, or the giodelion-style
  E-Wiz / Mega Minion / Fireball list? The rules assume Sweaty's.
- Rule 1 thresholds (≥ 7 in one lane, bot ≤ 3): too eager or too lax at your level?
- Does GDP really ignore our Giant until the bridge, or does it Goblin Gang it early at our side?
- Rule 9: should our Mini P.E.K.K.A in hand stop its push like an Inferno would, or does it just pre-place Goblin
  Gang to distract it and go anyway?
