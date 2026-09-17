# Exploration noise, arms NOISE and CONTROL (2026-09-17)

Both 120 updates from the FIN3 checkpoint (shaping off; m1 0.638, m2 0.545),
layout 4 puzzle / 6 finishing / 2 mid-game, threshold 0.4. NOISE draws moves on
puzzle and finishing boards from 0.75 pi + 0.25 Dir(0.3); CONTROL draws from pi.

| Dial | FIN3 start | NOISE @120 | CONTROL @120 |
|---|---|---|---|
| Lichess m1 top-1 | 0.638 | **0.597** | **0.653** |
| Lichess m2 top-1 | 0.545 | 0.471 | 0.571 |
| Own-game mate positions top-1 | 0.12 | 0.13 | 0.12 |
| Held-out conversion d2 / d10 / d20 | 0.20 / 0.04 / 0.06 | 0.13 / 0.08 / 0.07 | 0.16 / 0.12 / 0.06 |
| Own-game mate rate, 40 sampled games | 0.13 | 0.10 (8/80) | 0.08 (5/66) |
| Normalised entropy, puzzle / game boards | 0.10 / 0.33 | **0.27 / 0.52** | 0.10 / 0.31 |
| Curriculum depth | 6 | 2 | 6 |
| Mate-mass histogram, Lichess m1 (<0.05 / mid / >0.8) | 0.30 / 0.11 / 0.59 | 0.28 / 0.28 / 0.44 | 0.30 / 0.13 / 0.57 |

Reading: the noise hurt. The confident-wrong share it was aimed at barely moved
(0.30 -> 0.28) while the confident-right share fell (0.59 -> 0.44) and entropy
rose on every board, including the opening boards that had no noise. CONTROL,
plain continued training, kept the slow climb (m1 +0.015, m2 +0.026) and kept
the sharp policy. About a quarter of noisy-board picks were moves the network
gave < 5% (`explore_offprior_share` 0.24 -> 0.29), so the device sampled what it
was meant to sample; the update turned that into a flatter policy rather than a
corrected one.

Working explanation (to test, not established): advantages are normalised to
mean zero per minibatch, so on the long, draw-heavy finishing boards about half
of the random picks receive a positive advantage by construction and are
reinforced; PPO's clip only silences the negative-advantage noise picks. On the
one-move puzzle boards a random miss is unambiguously negative, so the damage
most likely came from the finishing boards. Candidate fixes: noise on puzzle
boards only; no advantage normalisation on noisy samples; label-guided behaviour
(play the known mate with probability eps) where labels exist. The clean arm
(SL -> outcome-only, 600 updates, running) gives an unpolluted base first.

# Arm PUZNOISE (2026-09-17): noise on the one-move puzzle boards only, from CONTROL

Mechanism test for the flattening. From the CONTROL checkpoint (m1 0.653, m2
0.571), 120 updates, eps 0.25 on puzzle boards only.

| Dial | CONTROL start | PUZNOISE @50 | @100 | @120 |
|---|---|---|---|---|
| Lichess m1 top-1 | 0.653 | 0.629 | 0.631 | 0.628 |
| Lichess m2 top-1 | 0.571 | 0.529 | 0.534 | 0.532 |
| Own-game mate positions | 0.12 | 0.09 | 0.13 | 0.12 |
| Entropy, puzzle / game boards | 0.10 / 0.31 | 0.12 / 0.32 | 0.16 / 0.33 | 0.17 / 0.34 |

Reading: milder than NOISE but the same sign. Noise confined to boards where a
random miss is unambiguously negative still cost 2-4 points and raised puzzle
entropy; the game-board entropy barely moved, so the finishing boards explain the
size of the NOISE flattening but not its existence. Behaviour noise with an
importance-corrected, clipped PPO update is out as a remedy here. What is left
for the confident-wrong third: search as the teacher (AlphaZero), or
label-guided behaviour on labelled boards.

# Arm GUIDE (2026-09-17): label-guided exploration, from CONTROL

120 updates from CONTROL (m1 0.653, m2 0.571), guide_epsilon 0.25 on puzzle and
finishing boards: the behaviour policy plays the demonstrator's move (puzzle key
move / human line / rules mate) one time in four; honest mixture log-prob in the
PPO ratio; update unchanged.

| Dial | CONTROL start | GUIDE @50 | @100 | @120 |
|---|---|---|---|---|
| Lichess m1 top-1 | 0.653 | 0.666 | 0.670 | **0.685** |
| Lichess m2 top-1 | 0.571 | 0.565 | 0.579 | **0.605** |
| Own-game mate positions top-1 | 0.12 | 0.13 | 0.17 | **0.17** |
| Held-out conversion d2 / d10 / d20 | 0.16 / 0.12 / 0.06 | 0.21 / 0.12 / 0.07 | 0.23 / 0.09 / 0.08 | **0.24 / 0.14 / 0.08** |
| Curriculum depth | 6 | 6 | 12 | **14** |
| Entropy, puzzle / game boards | 0.10 / 0.31 | 0.10 / 0.32 | 0.10 / 0.33 | 0.10 / 0.35 |

Reading: the first exploration device that helped, and on every dial at once:
+3 points m1, +3.5 m2, +5 own-game, +8 on two-ply conversion, and the depth
curriculum, stuck at 4-6 all day, climbed to 14. The policy stayed sharp
(puzzle entropy 0.10 throughout), unlike both noise arms. Training-side success
rates (puzzle m1 0.74, depth-0 0.80) are inflated by the guidance itself and are
not comparable; the held-out numbers above are the reading. About 70% of guided
picks were the demonstrated move (the network's own mass plus the 0.25).
Mechanism confirmed: guided picks are always the good move and always rewarded,
so nothing random is reinforced; the only cost is a small importance ratio on
the rarest ones.

Probe after GUIDE: Lichess m1 histogram <0.05 / mid / >0.8 = 0.25 / 0.12 / 0.62
(CONTROL 0.30 / 0.13 / 0.57): the confident-wrong bucket shrank five points and
the confident-right bucket grew five, the first arm to move the histogram in the
wanted direction. Human-mate d0: <0.05 0.47 -> 0.35, top-1 0.46 -> 0.48, value
+0.32 -> +0.49. Own-game m1 <0.05 0.80 -> 0.70, top-1 0.12 -> 0.17.

# Arm CLEAN (2026-09-17): outcome-only from the pure SL checkpoint, 600 updates

Owner's question: "isn't the FIN3 start already polluted with previous evaluation
using material also?" Answer arm: SL (m1 0.17, m2 0.13) -> PPO with shaping off
from update one, FIN3 layout, no guidance, 600 updates (169 min).

| Update | CLEAN m1 / m2 (outcome only) | Shaped ppo_from_sl m1 / m2 |
|---|---|---|
| 100 | 0.443 / 0.337 | - |
| 200 | 0.531 / 0.423 | - |
| 300 | 0.560 / 0.465 | 0.524 / 0.437 |
| 400 | 0.595 / 0.516 | - |
| 500 | 0.630 / 0.545 | - |
| 600 | **0.647 / 0.565** | 0.570 / 0.510 |

Own-game mate positions 0.05 -> 0.11; held-out conversion d2 / d10 / d20 at 600:
0.25 / 0.09 / 0.12 (best d2 and d20 of any arm); depth 6; entropy sharp.

Reading: shaping never helped from the prior. Outcome-only is ahead at 300 and
at 600, and CLEAN at 600 matches what the shaped path needed 1,020 updates for
(FIN3 + CONTROL: 0.653 / 0.571). Still rising at 600 (+1.7 points over the last
100). This is the unpolluted base: future arms start from CLEAN or its
continuation, not from the shaped lineage.

# Arm GUIDE_LONG (2026-09-17): 300 more guided updates from GUIDE

Named exception to the standard budget. From GUIDE (m1 0.685, m2 0.605),
guide_epsilon 0.25, switch to 4 finishing / 2 mid-game / 2 opening at 200.

| Update | m1 | m2 | own-game positions | conversion d2 / d10 / d20 |
|---|---|---|---|---|
| 0 (GUIDE) | 0.685 | 0.605 | 0.17 | 0.24 / 0.14 / 0.08 |
| 100 | 0.691 | 0.602 | 0.18 | 0.25 / 0.13 / 0.09 |
| 200 | 0.721 | 0.624 | 0.22 | 0.32 / 0.09 / 0.07 |
| 300 | **0.734** | **0.630** | **0.23** | 0.27 / 0.10 / 0.10 |

Curriculum depth 14 -> 18; puzzle entropy 0.09 (sharp); 74% of guided picks were
the demonstrated move. Still rising at 300: guidance has not saturated. Total
over 420 guided updates from CONTROL: m1 +8 points, m2 +6, own-game +11, depth
6 -> 18. Conversion from 10-20 plies out remains near 0.10: the human line helps
the attacker start, but the defender is the same network and deviates, after
which only a rules mate-in-one is demonstrated.

Probe after GUIDE_LONG: Lichess m1 histogram <0.05 / mid / >0.8 = 0.19 / 0.14 /
0.67 (CONTROL 0.30 / 0.13 / 0.57; supervised upper bound reached 0.90 top-1).
Human-mate d0 top-1 0.46 -> 0.58, value +0.32 -> +0.76. Own-game m1 <0.05
0.80 -> 0.59, top-1 0.12 -> 0.23. 40 sampled games: mate found 8/33 (0.24), 8 decisive,
against 7/100 and 7 decisive for the original SL+PPO checkpoint (small samples).

# Arm CLEAN_GUIDE (2026-09-17, night): guidance on the clean base

From CLEAN (SL -> 600 outcome-only; m1 0.647, m2 0.565), 420 guided updates so
the lineage totals 1,020 updates like GUIDE_LONG, none of them shaped.

| Update | m1 | m2 | own-game positions | conversion d2 / d10 / d20 |
|---|---|---|---|---|
| 0 (CLEAN) | 0.647 | 0.565 | 0.11 | 0.25 / 0.09 / 0.12 |
| 100 | 0.675 | 0.586 | 0.18 | 0.26 / 0.06 / 0.09 |
| 200 | 0.707 | 0.598 | 0.16 | 0.33 / 0.16 / 0.10 |
| 300 | 0.722 | 0.613 | 0.20 | 0.27 / 0.12 / 0.16 |
| 400 | 0.745 | 0.637 | 0.22 | 0.38 / 0.07 / 0.08 |
| 420 | **0.752** | **0.638** | **0.22** | 0.28 / 0.18 / 0.07 |

Versus GUIDE_LONG at the same 1,020 total updates (shaped lineage): 0.734 / 0.630
/ 0.23. The clean lineage is ahead on both puzzle sets and equal on own-game
positions; depth 20 (GUIDE_LONG 18); puzzle entropy 0.09; 75% of guided picks
were the demonstrated move. Still rising at the end. Best checkpoint of the
project so far, with a clean history: `experiments/exploration-noise/CLEAN_GUIDE`.
Conversion from 10-20 plies out is noisy at 100 games (0.07-0.18) and shows no
trend: unchanged verdict, that is the search problem.

Probe after CLEAN_GUIDE: Lichess m1 histogram <0.05 / mid / >0.8 = 0.17 / 0.16 /
0.67 (day start 0.30 / 0.11 / 0.59). Human-mate d0 top-1 0.52, value +0.76.
Own-game m1 top-1 0.22, <0.05 share 0.62 (from 0.80). 40 sampled games: 20 decisive (13 black, 7 white), 15 draws, mean 148 plies;
mate found in 20 of 54 chances (0.37). Day start: 7 decisive, 7 of 100.

# Greedy-attacker check (2026-09-17, night)

Is deep conversion low because sampling loses a ten-move plan? No. With the
attacker playing argmax (AlphaZero evaluates greedily) and the defender still
sampling: CLEAN_GUIDE d2 / d10 / d20 = 0.27 / 0.17 / 0.13 (sampled 0.28 / 0.18 /
0.07); day-start checkpoint 0.13 / 0.04 / 0.08. The policy does not know the
conversion; it is not a sampling artefact. Note also that d2 is capped well
below 1 even for a good player: the human defender often blundered into the
mate, so after the network's different reply a forced mate need not exist.

# Arm M34: mate-in-3 and mate-in-4 rungs (2026-09-17, night)

Owner: "in puzzles if defensor doesnt reply the same then the mate should come
even faster. Maybe we should add more puzzles than mate in 2." Puzzle mix m1-m4
at 0.25 each, the Lichess solution line as the demonstrator (rules mate-in-one
and forcing-move fallbacks), guidance 0.25, 120 updates from CLEAN_GUIDE. 40 min.

Held-out top-1 (baseline = CLEAN_GUIDE before the arm):

| update | m1 | m2 | m3 | m4 | own-game m1 |
|---|---|---|---|---|---|
| 0 | 0.752 | 0.638 | 0.454 | 0.375 | 0.22 |
| 50 | 0.737 | 0.637 | 0.522 | 0.436 | 0.23 |
| 100 | 0.744 | 0.639 | 0.541 | 0.441 | 0.25 |
| 120 | 0.741 | 0.634 | **0.546** | **0.442** | 0.22 |

m3 +9 points and m4 +7 in 120 updates, most of it in the first 50; m1 and m2
held within a point; own-game unchanged. Finishing conversion d2 / d10 / d20 =
0.32 / 0.16 / 0.12 (CLEAN_GUIDE 0.28 / 0.18 / 0.07): bystander, as predicted.
40 sampled games: 15 decisive (11 white, 4 black), 22 draws, 3 capped; mate found
15 of 51 (0.29; CLEAN_GUIDE 0.37). Probe: Lichess m1 confident-wrong share 0.16
(unchanged), own-game m1 0.61 (unchanged).

Win matrix (see `experiments/win-matrix/`): M34 vs SL prior 12-11 decisive
(0.49 score), while its parent CLEAN_GUIDE lost to SL 4-22 in the same session and
lost to M34 1-15. The deeper rungs did not move the puzzle dials much beyond their
own sets, but they repaired whole-game strength back to par with the prior. Working
reading: mate-in-3/4 positions are ordinary middlegame tactics, so training on them
pulled the policy back toward positions the prior was good at, whereas mate-in-1/2
and finishing boards are end-of-game only. Needs confirming with a second seed
before it is a lesson.

## M34 second seed (seed 1, same recipe), 2026-09-17 23:02

Held-out top-1 at 120: m1 0.732, m2 0.638, m3 0.535, m4 0.464, own-game 0.26
(seed 42: 0.741 / 0.634 / 0.546 / 0.442 / 0.22). Puzzle learning reproduces:
m3 +8, m4 +9, m1 / m2 within 2 points. Finishing 0.29 / 0.08 / 0.09. Games: 14
decisive of 40, mate 14 of 43 (0.33).

Win matrix (fresh game seeds): SL vs M34_s1 19-11 (score 0.57 for SL); SL vs
CLEAN_GUIDE 22-2 in the same session; CLEAN_GUIDE vs M34_s1 13-14.

Verdict on the "middlegame tactics repair the drift" reading: partly. Both seeds
close most of the gap to the prior (CLEAN_GUIDE loses 4-22 and 2-22; M34 12-11 and
11-19), so the direction is real, but seed 1 is still behind the prior and only
even with its parent. The deeper rungs slow or partly reverse the drift; they do
not by themselves restore the prior's strength. The opponent pool with the prior
in it remains the direct fix; M34 (seed 42) is the starting checkpoint.
