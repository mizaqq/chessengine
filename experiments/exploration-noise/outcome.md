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
