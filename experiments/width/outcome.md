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

## LONG256B (2026-09-21): LONG256 + 600 updates, no guard — COLLAPSED

| dial | LONG256 | LONG256B |
|---|---|---|
| mate-in-1 / 2 / 3 / 4 / endgame | 0.802 / 0.728 / 0.666 / 0.573 / 0.753 | **0.542 / 0.494 / 0.458 / 0.403 / 0.502** |
| own-game mate-in-one | 0.408 | 0.135 |
| finishing d2 / d10 / d20 | 0.53 / 0.34 / 0.20 | 0.21 / 0.14 / 0.09 |
| technique Q / R / RR / QR | 0.28 / 0.14 / 0.32 / 0.56 | 0.02 / 0.00 / 0.06 / 0.06 |
| gifts per 100 / own punish | 8.6 / 0.47 | 11.9 / 0.18 |
| game entropy / KL / clip | 0.198 / 0.040 / 0.18 | 0.426 / 0.058 / 0.26 |
| prior_score | 0.64 | 0.23 |
| vs LONG256, decisive | - | **3-48** |

Trajectory (LONG256B logs): held-out m1 0.79 flat to update 250, then 0.75 (350), 0.67 (450),
0.62 (550). prior_score 0.47 / 0.56 / 0.35 / 0.38 / 0.19 / 0.23 at 50..550 — the alarm we
named (below par twice) tripped at 350 but nothing stopped the run. Under the hood the drift
is slow and monotone from ~update 170: approx KL 0.04 -> 0.06, clip fraction 0.16 -> 0.26,
policy loss from -0.01 to +0.02, training puzzle rate 0.70 -> 0.45, entropy 0.15 -> 0.51,
draw rate 0.3 -> 0.8, SIL positive share 0.43 -> 0.28. Value loss did not explode.

Reading: not a blow-up but PPO drifting out of its trust band. LONG256 spent 600 updates at
lr 1e-4 with KL 0.03-0.04 and clip 0.15-0.18 (healthy); the continuation kept the same lr
(lr_decay_interval 1000 = no decay in a 600-update run, floor 1e-4) and the sharp 256 policy
started moving too far per update; once it unlearned, the noisier games fed it noisier
targets and the slide compounded (the mirror image of the entropy collapse we feared: the
policy flattened while forgetting). The LR5 arm from the same start had KL 0.019 and clip
0.105 with puzzles held — the healthy band Schulman et al. 2017 and the baselines' target-KL
early stop aim for.

Verdict: LONG256B is discarded; LONG256 remains the best checkpoint (unchanged). Fixes on the
table: (1) lr 5e-5 for continuations of the 256 line (LR5 evidence); (2) target-KL early
stopping inside the PPO update (stop the epoch loop when approx KL > ~0.03; standard in the
reference implementations); (3) a training stop when prior_score is below par at two
consecutive evaluations, so an alarm actually halts the run.

## LONG256C (2026-09-21): LONG256 + 600 with the three guards — alarm fired at update 100

lr 5e-5, target_kl 0.03 (per-minibatch check), stop_below_prior 0.5 / patience 2.
- `mean_kl_stop` = 1.0 at every update, `mean_optimizer_steps` 6.5 of 16: the guard cut every
  update after about a third of its steps; reported KL over the steps taken 0.007, clip 0.06.
- prior_score 0.40 (update 50), 0.35 (update 100) -> alarm; weights restored to the start
  (no evaluation had passed). The saved LONG256C is LONG256 bit for bit: identical
  games.json / finishes.json / audit. Held-out m1 at update 50 was 0.793 (LONG256 0.802).
- The matrix then played LONG256 against its own weights: 17-27. That is the noise band of
  60 sampled games between identical policies — worth remembering when reading 60-game
  results (Henderson et al. 2018).

Reading: the alarm did its mechanical job (stop + restore) and the run cost 7 minutes instead
of 55. Whether it fired on a real decline is unclear: two 20-game matches at 0.40 / 0.35 from
a policy that scored 0.56-0.69 across twelve evaluations in LONG256 is unlikely but not
impossible noise; and the per-minibatch KL check at 0.03 turned the recipe into ~1.5 PPO
epochs, which is a large change made blind. Spinning Up checks the *epoch-mean* KL and stops
at 1.5x target; our check is per minibatch, whose KL varies far more than the mean.

Proposed fixes (owner decides): (1) KL check on the running mean of the current epoch, trip
at 1.5x target (faithful to the reference); (2) the alarm uses 40 prior games (noise band
+-0.15 instead of +-0.2), patience 2; (3) rerun LONG256C.

## LONG256D (2026-09-21): guards v2 (whole-batch KL, target 0.02; alarm on 40 games) — alarm at 100 again

- KL guard now fires on 40% of updates, 10.4 of 16 steps taken; KL 0.016, clip 0.08 (in band).
- Held-out puzzles at update 100: 0.810 / 0.732 / 0.669 / 0.579 / 0.751 — **above** LONG256.
- prior_score 0.46 (16-23 of 80 decisive) at update 50, 0.38 (16-35) at 100 -> alarm, weights
  restored to LONG256. LONG256's own evaluations at 450-600 were 19-6, 19-8, 16-11, 19-8 of 40;
  a fresh 200-game match LONG256 vs slhi256 from the opening: **78-43** (score 0.59). So the
  continuation really does lose to the prior within 50-100 updates while every puzzle set
  rises — the dials-vs-strength split, now inside one run.
- On the training pool boards (mid-game starts) the continued learner still scores 0.5-0.6
  against the same prior. The evaluation plays from the opening. Hypothesis: the continuation
  degrades opening play specifically. Both continuations (C and D, different guards) show it,
  LONG192 -> LONG192B did not; what a restart resets: Adam moments (not saved in checkpoints),
  the opponent pool (prior only for the first 50 updates), the SIL buffer.

Probe queued (PROBE256): LONG256 + 100 updates, lr 5e-5, guards on but alarm off, checkpoint
kept; then 200-game matches vs slhi256 from the opening and from mid-game starts.

## PROBE256 (2026-09-21): LONG256 + 100 updates (lr 5e-5, KL guard, no alarm), 200-game matches

Same seed/config as LONG256D minus the alarm, so it reproduces it exactly (prior evals 16-23,
16-35) and keeps the checkpoint.

| match (200 games, decisive W-L, draws) | from the opening | from mid-game starts (high-Elo) |
|---|---|---|
| LONG256 vs slhi256 | **78-43** (79 draws) | 68-58 (74) |
| PROBE256 vs slhi256 | 51-73 (76) | 56-62 (82) |
| LONG256 vs PROBE256 | **99-52** (49) | **111-46** (43) |

Held-out puzzles of PROBE256 at update 100: 0.810 / 0.732 (LONG256 0.802 / 0.728).

Reading: 100 updates of continuation made the network weaker in games everywhere — most from
the opening (score vs prior 0.59 -> 0.44), a little from mid-game (0.53 -> 0.48), and badly
against its own start (LONG256 wins 2:1 from either start) — while every puzzle set improved.
This is not the LONG256B drift (KL 0.016, clip 0.08, 10 of 16 steps: all in band), and it
is not opening-specific. Whatever a restart resets is the suspect: the Adam moments (never
saved: every continuation starts Adam from zero, so the first steps are the largest possible
per parameter), the opponent pool (prior only until the first snapshot at update 50), the SIL
buffer (empty). LONG192 -> LONG192B survived the same restart, so the 256 network at its
sharpness (entropy 0.20 -> 0.15) is the fragile case.

Next (owner decides): isolate the restart. From slhi256: A = 200 updates saving the optimizer
state; B1 = A + 100 with the optimizer state restored; B2 = A + 100 with a fresh optimizer;
200-game matches of A, B1, B2 vs the prior. If B2 loses and B1 holds, checkpoints must carry
the optimizer (code ready: save/load of the Adam state). If both hold, the fragility is
LONG256's alone and the pool / SIL resets are next.

## RESTART probe (2026-09-21 evening): is it the Adam reset?

A = slhi256 + 200 updates (lr 1e-4, target_kl 0.02, alarm off; optimizer saved). B1 = A + 100
with the Adam state restored; B2 = A + 100 with a fresh Adam. 200-game matches from the opening:

| match | decisive W-L | draws |
|---|---|---|
| A vs prior | 80-42 | 77 |
| B1 (Adam restored) vs prior | **93-50** | 57 |
| B2 (fresh Adam) vs prior | 77-49 | 74 |
| B1 vs A | 71-64 | 65 |
| B2 vs A | 62-74 | 64 |
| B1 vs B2 | 81-67 | 52 |

Row means: B1 0.553, A 0.536, B2 0.502, prior 0.409. Puzzles: A 0.777 / 0.702, B1 0.778 / 0.704,
B2 0.792 / 0.724. Guard dials in band throughout (KL 0.011-0.012, clip 0.09-0.10, kl_stop
0.3-0.7). A's entropy at 200 updates: 0.35.

Reading: **a restart is not fatal** — both continuations from a 200-update checkpoint hold or
improve against the prior. Restoring Adam helps a little (B1 > B2 by 81-67 and by row mean),
not decisively; the fresh-Adam continuation still beats the prior 77-49. So the collapse of
every continuation from LONG256 is about LONG256's state, not the restart mechanics: after 600
guard-less updates at lr 1e-4 it sits at entropy 0.20 (A at 200 guarded updates: 0.35), and
from there any further training loses games while puzzles rise. Adam persistence stays on
(cheap, mildly positive).

Verdict / next: stop continuing LONG256. The clean path is a fresh 600-update run from slhi256
with the guards on from update 1 (lr 1e-4, target_kl 0.02, alarm on 40 games) — LONG256G —
compared against LONG256 on the same table. A (its first 200 updates, in effect) is already
80-42 vs the prior with entropy 0.35.

## LONG256G (2026-09-21 night): fresh 600 from slhi256, guards on from update 1 — new best

lr 1e-4, target_kl 0.02 (whole-batch, 1.5x), alarm on 40 games (never fired), optimizer saved.

| dial | LONG256 (unguarded) | LONG256G (guarded) |
|---|---|---|
| mate-in-1 / 2 / 3 / 4 / endgame | 0.802 / 0.728 / 0.666 / 0.573 / 0.753 | **0.824 / 0.752 / 0.672 / 0.598 / 0.768** |
| m1 trajectory | flat from 350 | 0.73 -> 0.82, **still rising at 550** |
| own-game mate-in-one | 0.408 | 0.330 |
| finishing d2 / d10 / d20 | 0.53 / 0.34 / 0.20 | 0.51 / 0.29 / 0.17 |
| technique Q / R / RR / QR | 0.28 / 0.14 / 0.32 / 0.56 | 0.22 / 0.12 / 0.24 / 0.54 |
| gifts per 100 plies / own punish | 8.6 / 0.47 | **6.9 / 0.68** |
| game entropy at 600 | 0.198 | **0.268** |
| KL / clip / kl_stop / steps per update | 0.040 / 0.18 / - / 16 | 0.012 / 0.09 / 0.9 late / 11.2 |
| vs slhi256, 200 games from the opening | 78-43 | **87-59** |
| LONG256 vs LONG256G, 200 games | | **93-53 for LONG256G** |
| wall clock | 52 min | 56 min |

Training-time prior match (40 games) over the run: 28-14, 36-13, 35-13, 27-17, 27-24, 29-26 at
50..550 — the margin narrows in the last 200 updates while the guard fires on most updates
(kl_stop 0.8-0.9). Same shape LONG256 showed before its continuations failed; here the 200-game
match still says 87-59 and puzzles are still climbing.

Reading: with the KL guard the same 600 updates end with a policy that is stronger (93-53 head
to head), sharper on every puzzle set, gives away a fifth fewer pieces and punishes more, and
keeps entropy at 0.27 instead of 0.20. The guard trimmed about a third of the optimizer steps
over the run and more late on — the step budget adapts as the policy sharpens, which is the
point (Spinning Up). The dials that did not improve (own-game m1, finishing, technique) are
within their 40/100/50-game noise bands.

Verdict: LONG256G is the best checkpoint. The guarded recipe (target_kl 0.02, alarm, optimizer
persistence) is the default for the 256 line. Open question for the continuation: the late
narrowing of the training-time prior margin; the alarm exists for exactly that.

## LONG256H (2026-09-21 night): guarded continuation of LONG256G — alarm at update 300

Optimizer restored, lr 1e-4, target_kl 0.02, alarm on 40 games. Training-time prior match:
34-19, 36-21, 30-16, 30-22, **25-28, 20-36** at 50..300 -> alarm; weights restored to update 200
(the last passing evaluation). Held-out m1 during the run: 0.817 (50), 0.807 (150), 0.786
(250) — declining from the start. Entropy 0.26 -> 0.17 by update 210. The KL guard fired on
every update and left 4.8 of 16 steps: even at ~one pass per update the policy kept moving
in a direction that lost games and puzzles.

| 200-game matches from the opening | decisive W-L |
|---|---|
| LONG256G vs prior | 96-38 |
| LONG256H (weights of update 200) vs prior | 75-44 |
| LONG256G vs LONG256H | **87-61** |

Reading: the alarm did its job and kept a checkpoint that still beats the prior, but the
continuation never improved on LONG256G — it declined from its first evaluation. Two
continuations of the 256 line (from LONG256 unguarded, from LONG256G guarded) now show the
same picture: after ~600 updates from this prior, further PPO updates under this recipe make
the policy sharper (entropy < 0.2) and weaker, and the KL guard slows the descent without
changing its direction. LONG256G stays the best checkpoint; its own late-run prior margin
(29-26, 24-18 at 550/600) was the early warning.

What this points at (owner decides tomorrow): the ceiling is in the recipe, not the step
size — candidates are the opponent pool (snapshots of an already-sharp self), the
self-imitation buffer (replaying the sharp policy's own wins), and the learning rate floor
(no decay: min_lr = lr). A better prior (two high-Elo months, running tonight) raises the
ceiling without touching the recipe; the recipe question remains.
