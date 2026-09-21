# width: outcome

## Question
Is the 128x10 network the ceiling? Owner (2026-09-21): compare 192 vs 128 with the same recipe;
"the actual test will be long run, but we should fix data there" (the priors were data-starved).

## Priors (supervised, same recipe, GPU)

| prior | data | held-out human top-1 | eval CE | m1 / m2 puzzles | vs old SL, decisive |
|---|---|---|---|---|---|
| sl (128) | 299k positions | 0.341 | 2.301 | 0.169 / 0.132 | - |
| sl192 | 299k | 0.338 | 2.300 | 0.161 / 0.144 | 8-6 |
| slbig128 | 1.43M | **0.405** | 1.981 | 0.261 / 0.224 | 24-1 |
| slbig192 | 1.43M | 0.404 | 1.975 | 0.267 / 0.253 | 18-5 |

Width changed nothing in the supervised stage at either data size; five times the data moved
every dial (train 0.40 vs eval 0.31 on the small set was the tell: data-limited, not
capacity-limited). slbig128 vs slbig192: 15-10, par.

## 120-update comparison from the small-data priors (CTRL128 vs RL192)
Puzzles 0.478 vs 0.494 (m1), own-game m1 0.205 vs 0.071, head to head 13-15. A wash; both lose
to their priors at 120 updates as every short run from a prior has. RL192 took 9 min on the GPU
vs 45 min for CTRL128 on the CPU.

## Long runs from the big-data priors (600 updates each, today's full recipe)

| dial | LONG3 (old lineage, ~2,000 updates) | LONG128 | LONG192 |
|---|---|---|---|
| mate-in-1 / 2 / 3 / 4 / endgame | 0.808 / 0.723 / 0.691 / 0.590 / 0.747 | 0.688 / 0.626 / 0.566 / 0.492 / 0.662 | 0.749 / 0.672 / 0.619 / 0.535 / 0.677 |
| m1 trajectory (updates 50 -> 550) | - | 0.59 -> 0.69, flat from 250 | 0.56 -> **0.73, still rising** |
| own-game mate-in-one (40 games) | 0.232 | 0.284 | 0.267 |
| finishing d2 / d10 / d20 | 0.43 / 0.13 / 0.11 | 0.33 / 0.16 / 0.18 | 0.43 / 0.20 / 0.20 |
| technique Q / R / RR / QR | 0.16 / 0.06 / 0.10 / 0.38 | 0.04 / 0.06 / 0.18 / 0.18 | 0.10 / 0.08 / 0.20 / 0.38 |
| gifts per 100 plies / own punish | 9.6 / 0.54 (DEPTH20) | 9.9 / 0.50 | 8.2 / 0.62 |
| game entropy / approx KL | 0.283 / 0.039 | 0.286 / 0.034 | 0.367 / 0.022 |
| vs own prior, decisive | 6-12 (vs sl) | 8-18 | 12-10 |
| vs LONG3, decisive | - | **30-9** | **32-2** |
| LONG128 vs LONG192, decisive | | 16 | **28** |
| wall clock (GPU) | 217 min (CPU) | 38 min | 44 min |

Five-way matrix row means: SLBIG128 0.585, LONG192 0.587, SLBIG192 0.569, LONG128 0.471,
LONG3 0.288.

## Reading
- **Strength was data-limited, not recipe-limited.** A big-data prior with zero RL beats LONG3,
  the strongest checkpoint of the old lineage, 24-0 and 29-2. Two thousand updates of curricula
  on a starved prior never made up for the missing human games. Balduzzi's warning in numbers:
  the old lineage kept beating its own past selves while staying weak against anything else.
- **Width matters in RL, not in SL.** Same prior quality, same recipe, same budget: LONG192 beats
  LONG128 28-16, is higher on every puzzle set (+0.04 to +0.06), on finishing at every depth, on
  gifts and on own punishment, and its mate-in-1 curve is still rising at update 550 where
  LONG128's flattened at 250. LONG192 stays at par with its prior (12-10) while LONG128 drifts
  below its own (8-18, prior_score 0.61 -> 0.35 over the last 200 updates). Entropy 0.37 vs 0.29
  and KL 0.022 vs 0.034: the wider network changes less per update and keeps more options open.
- **The held-out puzzle dials and game strength disagree.** LONG3 has the best puzzle numbers
  in the project and loses 32-2 to LONG192. The puzzle sets measure what the curriculum taught;
  games measure what the prior knew. Read puzzles as skill dials, not as strength.
- Correction to earlier reports: LONG3's "own-game mate-in-one 0.387" was the eval hook's
  self-play probe (`selfplay_top1` = 0.38666666666666666); `games.json` (40 games, seed 0) says 0.232. The table
  above uses games.json for all three.

## Verdict
Adopt the 192-filter network on the big-data prior as the main line: `experiments/width/LONG192`
is the new best checkpoint by games (and its curves are still climbing). The old lineage
(LONG3 / DEPTH20 / PUNISH*) is retired as a starting point; its lessons (lambda 1, depth
uniform over [0,20], mixed finishing slot, pool, SIL, punisher off) carry over in the recipe.
Counts unchanged per the owner (120 / 600) until told otherwise.

Next (owner decides): (1) continue LONG192 for 600 more (curves still rising); (2) redo the
key ablations on the new line only where the old result may not transfer (lambda, punisher);
(3) more data still: one more Lichess month roughly doubles the positions.

## LONG192B (2026-09-21): LONG192 + 600 updates (owner: "lets go with 1 and 2")

| dial | LONG192 (600) | LONG192B (1,200) |
|---|---|---|
| mate-in-1 / 2 / 3 / 4 / endgame | 0.749 / 0.672 / 0.619 / 0.535 / 0.677 | **0.775 / 0.701 / 0.659 / 0.571 / 0.714** |
| own-game mate-in-one (40 games) | 0.267 | **0.492** (project best) |
| decisive share, own games | 0.60 | 0.75 |
| finishing d2 / d10 / d20 | 0.43 / 0.20 / 0.20 | 0.45 / 0.19 / 0.19 |
| technique Q / R / RR / QR | 0.10 / 0.08 / 0.20 / 0.38 | 0.30 / 0.12 / 0.20 / 0.26 |
| gifts per 100 plies / own punish | 8.2 / 0.62 | 8.1 / 0.69 |
| game entropy / KL | 0.367 / 0.022 | 0.310 / 0.024 |
| vs slbig192 (own prior), decisive | 20-11 | **30-5** |
| vs LONG192, decisive | - | 22-19 |
| wall clock | 44 min | 56 min (GPU shared with pretraining) |

Reading: every puzzle set up again (+0.03 to +0.04), own-game mate-in-one nearly doubled, the
network now beats its prior 30-5 where the old lineage never got past par with its own.
Head to head with LONG192 only 22-19: game strength is saturating against a same-lineage
opponent while the skill dials still climb. Finishing at depth 10/20 flat at ~0.19 (the one
dial that has not moved on the new line either). Entropy 0.37 -> 0.31, still above the old
lineage's 0.28. LONG192B is the current best checkpoint. Meanwhile the two-month 192 prior
(sl2m192, 2.85M positions) reached top-1 0.432 and beats slbig192 21-6: the next long run
should start from a better prior rather than continue this one.

## LONG256 (2026-09-21): slhi256 (high-Elo, 256 filters) + 600 updates

| dial | LONG192B (1,200 upd, 1500-Elo prior) | LONG256 (600 upd, 2000-Elo prior) |
|---|---|---|
| mate-in-1 / 2 / 3 / 4 / endgame | 0.775 / 0.701 / 0.659 / 0.571 / 0.714 | **0.802 / 0.728 / 0.666 / 0.573 / 0.753** |
| m1 trajectory | 0.75 -> 0.78 | 0.74 -> 0.81 (flat from 350) |
| own-game mate-in-one | **0.492** | 0.408 |
| finishing d2 / d10 / d20 | 0.45 / 0.19 / 0.19 | **0.53 / 0.34 / 0.20** |
| technique Q / R / RR / QR | 0.30 / 0.12 / 0.20 / 0.26 | 0.28 / 0.14 / **0.32 / 0.56** |
| gifts per 100 plies / own punish | 8.1 / 0.69 | 8.6 / 0.47 |
| game entropy / KL / clip | 0.310 / 0.024 / 0.14 | **0.198** / 0.040 / 0.18 |
| vs own prior, decisive | 30-5 | 31-11 |
| vs LONG192B, decisive | - | **28-22** |
| vs LONG192 | 22-19 | 32-11 |
| wall clock | 56 min | 52 min |

Four-way row means: LONG256 0.631, SLHI256 0.561, LONG192B 0.422, LONG192 0.386. The bare
high-Elo prior beats LONG192B 31-6.

Reading.
- **The new best checkpoint by every game measure and most dials.** In 600 updates LONG256
  reaches LONG3's puzzle numbers (which took the old lineage ~2,000 updates) and the first
  real movement on finishing depth 10 in the project (0.19 -> 0.34). Technique QR 0.56.
- **The prior is most of it.** slhi256 with zero RL beats LONG192B 31-6: a better start beats
  1,200 updates of curriculum on a worse one, for the third time today.
- **Warning signs.** Game entropy 0.198 is the lowest ever recorded (old lineage 0.28, LONG192B
  0.31); KL and clip fraction are up; own-game mate-in-one is below LONG192B and own punishment
  fell to 0.47. The 256 network sharpens fast; with m1 flat from update 350 this looks like the
  policy collapsing onto a narrow repertoire rather than still learning. Continuing without an
  entropy guard is the risk to discuss before the next long run.

Verdict: LONG256 is the main line. Before continuing it, decide on an entropy floor (e.g. raise
the entropy coefficient or stop decaying it) and measure on 120 updates.

## Entropy guard arms (2026-09-21): LONG256 + 120 updates each

| dial | LONG256 | ENT03 (entropy_coef 0.03) | LR5 (lr 5e-5) |
|---|---|---|---|
| game entropy (end) | 0.198 | **0.156** | **0.154** |
| KL / clip fraction | 0.040 / 0.18 | 0.052 / 0.17 | 0.019 / 0.11 |
| mate-in-1 / 2 / 3 / 4 / endgame | 0.802 / 0.728 / 0.666 / 0.573 / 0.753 | 0.812 / 0.739 / 0.673 / 0.587 / 0.746 | 0.780 / 0.726 / 0.659 / 0.572 / 0.720 |
| own-game mate-in-one | 0.408 | 0.207 | 0.361 |
| finishing d2 / d10 / d20 | 0.53 / 0.34 / 0.20 | 0.51 / 0.28 / 0.23 | 0.48 / 0.25 / 0.30 |
| prior_score (20 games) | 0.64 | 0.44 | 0.45 |
| vs LONG256, decisive | - | 21-24 | 25-22 |
| ENT03 vs LR5 | | 30-17 | |

Reading: **neither guard moved entropy** — both arms kept sliding (0.20 -> 0.15) in 120
updates; tripling the entropy coefficient did nothing visible, and halving the learning rate
halved KL as expected but not the slide. Puzzles held or rose slightly (ENT03 best puzzle
numbers in the project); own-game mate-in-one and prior_score fell in both arms, within
the 40/20-game noise bands but in the same direction. Head to head both arms are level with
LONG256. LONG256's own entropy series was a monotone slide from 0.56 (update 10) to 0.20
(update 580), so the 256 network sharpens continuously under this recipe regardless of the
two knobs we turned by these amounts.

Open question (owner): is a sharp policy a problem in itself? The dials say no yet (puzzles
up, strength level); the risk is the AlphaGo-SL lesson that a narrow policy explores less
and will plateau. Options: entropy_coef 0.1 (a real step), a KL-target / early stop per
update, or accept and watch prior_score on the next long run.

## ENT10 (2026-09-21): LONG256 + 120 updates, entropy_coef 0.1

| dial | LONG256 | ENT10 |
|---|---|---|
| game entropy | 0.198 | 0.216 (stopped sliding, barely up) |
| mate-in-1 / 2 / 3 / 4 / endgame | 0.802 / 0.728 / 0.666 / 0.573 / 0.753 | 0.783 / 0.719 / 0.643 / 0.571 / 0.721 |
| own-game mate-in-one | 0.408 | 0.357 |
| finishing d2 / d10 / d20 | 0.53 / 0.34 / 0.20 | 0.45 / 0.26 / 0.22 |
| prior_score | 0.64 | 0.48 |
| vs LONG256, decisive | - | 18-29 |

Reading: at 0.1 the bonus finally registers — the slide stops (0.198 -> 0.216) — but every
skill dial pays for it and LONG256 beats the arm 29-18. The guard rule in the driver
(entropy +0.03 and m1 not down > 0.01) rejected it; LONG256B runs without an entropy guard,
with prior_score as the alarm. Lesson: in this loss the entropy term becomes visible around
0.1 in normalised units, and at that size it costs more skill than the sharpness costs strength.
