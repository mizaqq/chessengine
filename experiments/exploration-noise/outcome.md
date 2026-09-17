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
