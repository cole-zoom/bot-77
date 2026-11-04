# Golem Beatdown (Golem Hog hybrid)
Source: Master Golem In 21 Minutes! · Sweaty Clash · 2026-09-30 · https://www.youtube.com/watch?v=PCC-EB8YIXU

## Deck
Golem 8 · Barbarians 5 · Hog Rider 4 · Valkyrie 4 · Firecracker 3 · Dart Goblin 3 · Skeletons 1 · Heal Spirit 1.
Average 3.6; cheapest 4-card cycle 8 (Skeletons, Heal Spirit, Firecracker, Dart Goblin). sim: all 8 in the engine.

The description has no deck list; this is pieced together from the cards he names across the six games (Golem 0:00,
Dart 0:31, Hog / Barbarians / Heal Spirit 1:06, Skeletons / Firecracker 1:39, Valkyrie 2:43). It is **not** the
classic Golem build (Night Witch, Baby Dragon, Lightning, Tornado). It has **no spells**: Firecracker and Dart Goblin
are the only answers to air and swarms (18:08, 20:21). The video uses evolutions (several Barbarian sets at 2:43,
evos cycled in single at 3:15 and 8:42; the captions suggest Barbarians and Firecracker or Skeletons) and an
unnamed card with an "ability" (5:26, 11:58). The scripted opponent uses normal forms of all 8.

## Game plan
Survive single elixir with both towers and no big elixir deficit, chipping with Hog (0:00, 1:06). In his words at 3:15,
"Single time is for hogs and hogs alone"; in double, Golem in one lane and Hog in the other to split
the defense. Defense comes first (2:43). Scared of: Mini P.E.K.K.A (answers both Hog and Golem, 18:41), cheap
spells on Firecracker and Dart Goblin (20:21), and air it can't reach with two ranged troops (18:08).

## Card roles
- Golem: win condition in double elixir; placed in the back (9:14). Rarely in single (11:23).
- Hog Rider: single-elixir chip; the second-lane threat during a Golem push (1:06, 3:15).
- Barbarians: main defense vs ground win conditions (Battle Ram 1:06, Giant 17:02); bridge support behind Golem.
- Valkyrie: splash defense; cleans support behind a push (2:43, 9:14).
- Firecracker: ranged splash, air defense; protected and kept alive (6:32).
- Dart Goblin: ranged, air defense; kept away from spells (19:49, he wanted his Dart placed lower).
- Skeletons: 1-elixir distraction and cycle, often off to the side (3:48).
- Heal Spirit: heals Barbarians back to full for 1 (7:36); offsets poison/chip damage (16:27).

## Opening
**[inferred]** The video doesn't show 0:00 plays. Waits to ~9-10 elixir. If the bot hasn't played by then: Hog +
Skeletons at a bridge, or Firecracker behind a princess tower to cycle. If the bot plays first: answer at the tower
with the cheapest fitting card, then Hog in the other lane. Never opens Golem (no Golem in single, 7:36) and never
opens Barbarians on offense.

## Against our deck
- Giant: Barbarians when it's 2-3 tiles from the tower (17:02, he punished a Giant into Barbarians); Firecracker or
  Dart behind the tower picks off Bomber. **[inferred]** placement.
- Battle Ram: Barbarians in its path; without Barbarians in hand, the Ram connects (1:06). Skeletons as a fallback.
- Mini P.E.K.K.A (sim name "Mini PEKKA"): Skeletons or Barbarians to surround it **[inferred]**. Our Mini P.E.K.K.A is the card that hurts
  it most: it kills Hog and chews Golem (18:41), and a Mini P.E.K.K.A counterpush took a whole tower (14:12, 17:02).
- Mega Minion: Dart Goblin or Firecracker **[inferred]**. Mega Minion behind Mini P.E.K.K.A on a Golem is the ugly
  matchup: only two troops can hit air (18:08).
- Bomber: Dart Goblin outranges it, Valkyrie walks through it **[inferred]**.
- Goblin Hut: pulls Golem and Hog (both target buildings); Valkyrie on the spawns, Hog in the lane away from the Hut
  **[inferred]**.
- Fireball: kills Firecracker or Dart Goblin outright (20:21). It keeps them 4+ tiles apart and away from its towers
  so one Fireball never gets two **[inferred]** spacing; the spreading idea is from 6:32.
- Arrows: hits Skeletons, Heal Spirit, Dart, Firecracker. He rates Arrows near the worst card to face (5:59).

## Punish rules (checked top to bottom, every tick)
Engine frame (`ArenaLayout.h`): 18 x 34, cells centred on integers. Bot = team 0, defends low y; this opponent =
team 1, defends high y. River y 15.5-17.5. Lanes x 3.0 (L) / 14.0 (R), bridges x 2.5 / 14.5, middle x 8.5. Opponent
princess (3, 27) / (14, 27), king (8.5, 30.5); bot princess (3, 6) / (14, 6). Its bridge spot (bridge x, 18.5), its
back (lane x, 32.5), behind its king (8.5, 33), its defensive middle (8.5, 24.5). "Other lane" = the lane the bot didn't commit to. "Not in hand" = the bot
played that card fewer than 4 plays ago.

1. IF the bot spent ≥5 in one lane within 5 s and the bot's elixir estimate ≤3 → defend that lane at the tower with
   Barbarians or Valkyrie (+ Skeletons to distract a Giant), then Hog Rider at the other bridge ((2.5 or 14.5, 18.5)),
   Heal Spirit or Skeletons right behind it, 0.5-1 s after the defender lands (1:06 pushes the opposite lane, 4:21).
2. IF the bot's Mini P.E.K.K.A is not in hand → Hog at the bridge now (out-cycling the Mini P.E.K.K.A, 15:19). IF
   it is in hand and its own elixir < 8 → don't send Hog alone; cycle Skeletons/Firecracker instead **[inferred]**.
3. IF a bot Battle Ram or Giant crosses the river → Barbarians in its path in front of that princess tower ((lane x, 25))
   (1:06, 17:02); Firecracker behind the tower ((lane x, 29)) if the bot added Bomber/Mega Minion. Heal Spirit on the
   Barbarians if they drop below half (7:36).
4. IF Golem is at its back (it spent 8, < 6 s ago) and the bot attacks the other lane → defend with the cheapest
   ground answer (Valkyrie, Skeletons, Barbarians) at (lane x, 25) or the middle (8.5, 24.5) + Firecracker; never a second big card in the Golem lane until
   the defense ends (9:14, he defended everything while 8 was in the other lane). If the defense would cost > 6 and
   that tower is < 30% → let it go and support the Golem instead (2:12, he was ready to give up a tower).
5. IF double elixir and own elixir ≥ 9 and no bot troop on its side → Golem at its back in the lane
   where the bot's tower is lower ((3 or 14, 32.5)), or behind the King ((8.5, 33)) to stay hidden (2:43 golem on king
   tower).
6. IF its Golem reaches its bridge (y ≤ 18.5) → Firecracker or Dart Goblin 3 tiles behind it (y + 3), Barbarians at
   (bridge x, 18.5) as it crosses with Heal Spirit on them, units spread out so one spell can't hit all (5:26, 6:32). When the bot commits
   a defender to the Golem lane, Hog at the other bridge (3:15).
7. IF the bot's Fireball and Arrows are both not in hand → stack Firecracker + Dart together behind Golem; otherwise
   keep them ≥4 tiles apart **[inferred]**.
8. IF single elixir, it leads on tower damage, and the bot's elixir estimate ≤ 3 → Golem in the back anyway (11:23);
   it ties up the bot's elixir for the rest of single (12:31).
DEFAULT: single elixir, defend at the tower and send Hog + a 1-elixir card at the bridge when at ≥ 8 elixir or right
after a positive defense; never Golem except rule 8 (7:36). Double elixir, rule 5 then rule 6, one Golem push per
cycle; Hog in the other lane. Overtime, Hog at the lowest bot tower whenever one Hog hit finishes it (15:55), else
same as double.

## Scenarios
- **Punish the back Golem** · 2:05 (double) · bot 9, opp 1 · opponent Golem just placed at its R back (14, 32.5), nothing
  else on the board · opp hand Barbarians, Valkyrie, Skeletons, Firecracker; next Dart Goblin · Learn: when it spends
  8 in the back, attack the *other* lane right away (Battle Ram + Bomber, or Giant at the bridge). Bad play: dropping
  Mini P.E.K.K.A + Mega Minion on the Golem in its own half → Golem dies late, the opponent rebuilds to 7+, and the
  Barbarians + Firecracker counterpush with Hog in the other lane (rules 4, 6) **[inferred]**.
- **Defend the support, not the tank** · 2:40 · bot 8, opp 2 · Golem on the bot's R bridge (14.5, 15), Firecracker and
  Dart Goblin 3 tiles behind it (y ≈ 18) · opp hand Barbarians, Hog, Heal Spirit, Skeletons · Learn: Fireball
  the Firecracker +
  Dart pair (or Arrows if it's spread), kite or take Golem hits on the tower, Mini P.E.K.K.A on Golem only after the
  support is dead (20:21: spells on his two ranged troops leave Golem with nothing). Bad play: Mini P.E.K.K.A on
  Golem first → Firecracker/Dart kill it, Barbarians land at the bridge (rule 6).
- **Save the spells for double** · 1:20 (single) · bot 7, opp 6 · bot Fireball used 10 s ago on Hog + tower · opp
  hand Golem, Firecracker, Dart Goblin, Heal Spirit · Learn: in single, answer Hog with Mini P.E.K.K.A or a centre
  Goblin Hut, keep Fireball for the Golem push support **[inferred]**. Bad play: Fireball on Hog chip → rule 7 stacks
  Firecracker + Dart behind the first Golem.
- **Opening all-in** · 0:05 · bot 10, opp 10 · empty board · opp hand Barbarians, Hog, Skeletons, Valkyrie; next
  Heal Spirit · Learn: open with a cheap or split play (Bomber or Goblin Hut), not Giant + Battle Ram in one lane.
  Bad play: > 5 elixir in one lane → Barbarians defend, Hog + Skeletons hit the other tower (rule 1).
- **Respect the Mini P.E.K.K.A cycle** · 1:00 · bot 6, opp 8 · bot just used Mini P.E.K.K.A on a Hog in L ·
  opp hand Hog, Skeletons, Firecracker, Valkyrie · Learn: after spending Mini P.E.K.K.A, keep 4 elixir for a Hog in
  the other lane (Goblin Hut or Bomber + tower). Bad play: Mini P.E.K.K.A counterpush with Mega Minion (7 total) →
  Valkyrie defends, Hog at R while Mini P.E.K.K.A is out of hand (rules 1, 2).

## Open questions for Cole
1. The video's deck is a Golem Hog hybrid with no spells. Is that the Golem opponent you want, or a classic
   Golem / Night Witch / Baby Dragon / Lightning list (or both, as two opponents)?
2. Rule 8 (Golem in single when ahead): what lead is enough? I used "leads on tower damage + bot ≤ 3 elixir".
3. Which card had the "ability" (5:26, 11:58)? The scripted opponent ignores it either way.
