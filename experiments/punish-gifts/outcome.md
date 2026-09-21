# punish-gifts: outcome

## PUNISH (2026-09-19): DEPTH20 + 120 updates, one-ply rules punisher at 0.5 on every board

Owner: "the model we are playing against is not punishing for very stupid moves so model is
not learning to avoid them at all" -> "go ahead with it". `punish_epsilon 0.5`,
`punish_threshold 3`, `punish_boards all`; everything else as DEPTH20. Driver `run_punish.sh`.
Audit and matrix run the bare network (no punisher), so every number below is what the
network learned, not what the punisher did for it.

| dial | DEPTH20 (start) | PUNISH |
|---|---|---|
| free-piece gifts per 100 own plies (vs self) | 9.6 | **6.6** |
| own punish rate on free pieces (vs self) | 0.54 | **0.72** |
| allowed mates per 100 plies (vs self) | 0.87 | 0.83 |
| own punish rate on allowed mates | 0.13 | **0.28** |
| gifts per 100 plies vs SL / SL punishes | 8.3 / 0.49 | 7.4 / 0.61 |
| head to head vs DEPTH20, decisive | - | **14 to 8** for PUNISH |
| prior_score (20 games) | 0.41 | **0.53** (first >= 0.5 since the pool runs) |
| mate-in-1 / 2 / 3 / 4 / endgame (held-out top-1) | 0.804 / 0.723 / 0.683 / 0.580 / 0.751 | 0.772 / 0.698 / 0.656 / 0.561 / 0.733 |
| own-game mate-in-one | 0.352 | **0.257** |
| finishing d2 / d10 / d20 | 0.40 / 0.20 / 0.13 | 0.36 / 0.18 / 0.17 |
| technique Q / R / RR / QR (held-out, level 3) | 0.20 / 0.12 / 0.16 / 0.22 | 0.10 / 0.00 / 0.12 / 0.22 |
| game entropy | 0.288 | 0.303 (up) |
| training draw rate (last update) | 0.70 | 0.48 |
| punish_moves / punish_picks per update | - | ~850 / ~420 (fires 0.5 as designed) |

Reading.
- **The mechanism works.** With the punisher off at evaluation time the network hangs a third
  fewer pieces and takes a hung piece 0.72 of the time instead of 0.54; allowed mates get
  delivered twice as often. It beats its own start 14-8 and is at par with the prior (0.53,
  20 games). Entropy went *up*, not down; training games are far more decisive (draws 0.70 ->
  0.48) because gifts now end games.
- **The cost.** Every puzzle set fell by ~0.03 (consistent across five sets: real), own-game
  mate-in-one fell 0.35 -> 0.26, technique held-out fell. The old shaping lesson in a new
  coat: a hint that pays for captures teaches piece-grabbing over mating.
- **A design flaw that explains the mate numbers.** `punish_actions` returns mates *and* free
  captures as one set and the punisher draws uniformly from it. Whenever a mate and a free
  piece exist together (common in won positions) the punisher takes the piece half the time
  instead of mating, and the mixture log-prob makes that a legitimate sample. The label
  demonstrator already orders its rules (mate first); the punisher must too.
- Caveat: the SL prior "punished" our gifts 0.61 instead of 0.49 without changing. The gifts
  we still make are the ones a one-ply check finds most clearly; some earlier "gifts" were
  poisoned pieces the audit cannot tell apart.

Verdict: hypothesis confirmed (fewer gifts, more punishment, stronger head to head), with a
fixable side effect. Next: mates take priority over captures in the punisher (spec change
within this change), rerun 120 updates from DEPTH20 as PUNISH2, compare on the same table.

## PUNISH_LONG (2026-09-19): DEPTH20 + 600 updates, punisher 0.5, mates before captures

Owner: "make the fix and run a longer run". Same as PUNISH except the punisher offers the
mating moves when any exist and free captures only otherwise. Training took 3.6 h (games are
shorter with a punisher on both sides). Bare network at evaluation throughout.

| dial | DEPTH20 | PUNISH (120) | PUNISH_LONG (600) |
|---|---|---|---|
| free-piece gifts per 100 plies / own punish rate | 9.6 / 0.54 | 6.6 / 0.72 | 7.4 / 0.71 |
| allowed mates per 100 / own punish rate | 0.87 / 0.13 | 0.83 / 0.28 | 0.79 / 0.30 |
| mate-in-1 / 2 / 3 / 4 / endgame | 0.804 / 0.723 / 0.683 / 0.580 / 0.751 | 0.772 / 0.698 / 0.656 / 0.561 / 0.733 | 0.784 / 0.706 / 0.678 / 0.555 / 0.732 |
| own-game mate-in-one | 0.352 | 0.257 | 0.262 |
| finishing d2 / d10 / d20 | 0.40 / 0.20 / 0.13 | 0.36 / 0.18 / 0.17 | 0.29 / 0.17 / 0.09 |
| technique Q / R / RR / QR | 0.20 / 0.12 / 0.16 / 0.22 | 0.10 / 0.00 / 0.12 / 0.22 | 0.14 / 0.06 / 0.16 / 0.24 |
| game entropy | 0.288 | 0.303 | 0.333 (rising all run) |
| vs SL, decisive | 14-11 | 14-7 | 12-15 |
| vs DEPTH20, decisive | - | 11-11 | 15-12 |

Five-way matrix (30 games per colour): every row mean between 0.48 and 0.52. No checkpoint
is distinguishable from another at this sample size; PUNISH_LONG beats LONG3 15-11 and
DEPTH20 15-12 and loses to SL 12-15 and PUNISH 11-13.

Reading.
- **The punisher fixes exactly what it measures and nothing more.** Gifts down a quarter to a
  third, own punishment up, and those gains appear within 120 updates and do not grow with
  600. Game strength stays at par with everything (the self-play plateau since LONG2).
- **The mate-priority fix did not bring the mates back.** Own-game mate-in-one stayed at 0.26,
  and depth-2 finishing (find the mate in two against our own defence) fell further to 0.29.
  So the cost is not the uniform draw; it is the exploration shift itself: about 8% of learner
  plies per update are punisher rows, and half of those are captures the network did not
  choose. The batch leans toward material play; the mating skill that took LONG2 and LONG3
  1,200 updates to build decays in 120.
- Entropy rose 0.29 -> 0.33 over the run (the punished side keeps losing material in new ways),
  the opposite of the drift we worried about, and probably part of why the puzzle dials
  slipped: a flatter policy on a fixed-size network.
- Diagnostic (6 self-play games, 822 plies): a punishing move exists in 18% of positions, a
  mate in one in 1.3%; punish_moves ~85 per update after the label boards are excluded.

Verdict: hypothesis confirmed as a mechanism, not as a lever on strength. As configured
(epsilon 0.5, all boards) the punisher costs more finishing and mating skill than it buys.
`punish_epsilon` stays 0 by default. Options recorded for the owner: (1) epsilon 0.2 on game
boards only (opening + mid-game), curriculum boards clean; (2) punisher on the pool
opponents only (the original option 2), which never touches the learner's batch; (3) park it
and record the lesson.

## POOL_REF (2026-09-21): DEPTH20 + 120 updates, referee on the five pool opponents only

Owner: "okay go 1 now". `pool_referee true`, `punish_epsilon 0`: the frozen pool opponent plays
a rules mate, else a free capture >= 3, whenever one exists (11% of its plies were replaced:
178 of 1606 per summary window). Learner plies untouched. Bare network at evaluation.

| dial | DEPTH20 | PUNISH (learner-side, 120) | POOL_REF (opponent-side, 120) |
|---|---|---|---|
| free-piece gifts per 100 plies / own punish rate | 9.6 / 0.54 | 6.6 / 0.72 | 8.0 / 0.53 |
| allowed mates per 100 / own punish rate | 0.87 / 0.13 | 0.83 / 0.28 | 0.65 / 0.15 |
| mate-in-1 / 2 / 3 / 4 / endgame | 0.804 / 0.723 / 0.683 / 0.580 / 0.751 | 0.772 / 0.698 / 0.656 / 0.561 / 0.733 | 0.800 / 0.730 / 0.688 / 0.576 / 0.755 |
| own-game mate-in-one | 0.352 | 0.257 | 0.328 |
| finishing d2 / d10 / d20 | 0.40 / 0.20 / 0.13 | 0.36 / 0.18 / 0.17 | 0.39 / 0.17 / 0.17 |
| technique Q / R / RR / QR | 0.20 / 0.12 / 0.16 / 0.22 | 0.10 / 0.00 / 0.12 / 0.22 | 0.12 / 0.04 / 0.18 / 0.26 |
| game entropy | 0.288 | 0.303 | 0.270 |
| vs DEPTH20, decisive | - | 14-8 | 17-15 |
| vs PUNISH, decisive | 8-14 | - | 11-18 |
| training score vs the (refereed) prior member | 0.4-0.5 | - | 0.08 |

Reading.
- **The cost is gone.** Puzzles, own-game mate-in-one, finishing and technique all sit on
  DEPTH20's numbers (within noise). Confirms the diagnosis: PUNISH's losses came from the
  learner's own exploration being shifted, not from punishment as such.
- **The benefit is smaller.** Gifts fell 9.6 -> 8.0 (PUNISH: 6.6) and own punishment did not
  move, because only the five pool boards carry the lesson and the learner never plays the
  punishing move itself. Head to head it is level with DEPTH20 (17-15) and loses to PUNISH
  (11-18).
- **The refereed prior crushes the learner (0.08).** Punishment at rate 1.0 on those boards is
  a very different opponent; the learner is still losing those games after 120 updates. A
  longer run might turn that around, but the held-out dials give no reason to expect a
  strength gain.
- **A pattern across the three arms:** PUNISH is the checkpoint that wins head to head
  (14-8 vs DEPTH20, 18-11 vs POOL_REF, best row in the five-way matrix) while being the
  worst on every mate dial. Among networks at this level, material discipline decides games
  more than mating skill does. The held-out dials measure what we chose to teach, not what
  wins games between these networks.

Verdict: opponent-side punishment is free but weak; learner-side punishment is effective but
costly. Neither moves the network off the plateau. Both stay off by default; the change is
recorded and ready to archive after the comprehension check. Next lever: capacity (SL192).
