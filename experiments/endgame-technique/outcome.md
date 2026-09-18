# Arm TECH: endgame-mate puzzles + generated technique boards (2026-09-18, 08:59-09:37)

120 updates from POOL_LONG, default config: 16 boards, 4 puzzle (five sources at 0.2,
endgame_mate new), 2 technique boards (Q / R / RR / QR, caps 40 / 60 / 40 / 40), 5 mid-game,
5 opening, pool on 5 game boards, guidance 0.25 on curriculum boards (rules mate-in-one
and forcing-move demonstrator on technique boards). 37 min. Owner before the result:
"my suspicion is that 120 updates will not be enough to get strong signal."

## Held-out dials (50 generated positions per set, both sides sampling, no guidance)

| set | POOL_LONG | 50 | 100 | 120 |
|---|---|---|---|---|
| Q (cap 40) | 0.00 | 0.04 | 0.04 | 0.00 |
| R (cap 60) | 0.00 | 0.02 | 0.02 | 0.02 |
| RR (cap 40) | 0.06 | 0.06 | 0.10 | 0.10 |
| QR (cap 40) | 0.04 | 0.12 | 0.10 | 0.04 |

Other dials at 120: m1 0.777, m2 0.690, m3 0.643, m4 0.549 (POOL_LONG 0.785 / 0.690 /
0.639 / 0.541); endgame-mate 0.698 (baseline 0.7105); own-game 0.28; finishing d2 / d10 /
d20 0.35 / 0.18 / 0.12 (unchanged); prior_score 0.39, 0.41, 0.54 at 50 / 100 / 120 (16
decisive games each, noisy). 40 sampled games: 20 decisive, mate 20 of 72.

## Training-window success (with guidance), 12-20 attempts per set per 20 updates

Q 0.00-0.14, R 0.00-0.09, RR 0.17-0.67, QR 0.09-0.40. Far above the held-out rates.

## Reading

- Owner's suspicion confirmed: the held-out technique dial did not leave zero for the
  single pieces and moved within noise for the pairs. 120 updates gave about 75 games
  per set from a zero start; that is not enough for the sparse win to be found and
  reinforced.
- The gap between training success (up to 0.67 for two rooks) and held-out success
  (0.10) is the demonstrator: on the boards, the rules mate-in-one / forcing-move
  fallback plays the finishing move a quarter of the time whenever one exists, so the
  wins were produced by the teacher, not by the network. Without the teacher (held-out)
  the network still cannot finish. The wins entered the batch with the honest
  behaviour ratio, but 75 games were too few for the policy to absorb them.
- The rest of the run behaved: puzzle dials held or rose slightly, the pool score
  against the prior did not fall, the endgame-mate rung starts at 0.71 and did not move
  in 120 updates (it overlaps the m1/m2 sets heavily).

## Options next (owner decides)

1. **Tablebase demonstrator + technique curriculum.** Syzygy 3-4 piece tables (a few
   MB, python-chess reads them) give the perfect move in every technique position and
   a distance to the win; generate starts at increasing distance (Florensa 2017 again,
   this time with a ruler) and let the demonstrator play the tablebase move at
   guidance 0.25. Turns a sparse reward into a dense good-move signal. Recommended.
2. **Longer run as is** (300-600 more updates). Cheap in code, expensive in time; the
   training-window rates suggest wins will keep coming mainly from the teacher.
3. **More technique boards** (4 instead of 2) to double the games per set; same
   sparse-signal shape, faster.

Files: `TECH/run.json, games.json, finishes.json, probe.txt, baseline_end.json,
baseline_technique.txt`, `run_tech.sh`, `run_tech.log`, `prepare_endgame.log`.
