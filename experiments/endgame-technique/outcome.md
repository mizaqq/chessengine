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

# Arm TECH2: 3 puzzle / 4 technique / 5 mid-game / 4 opening (2026-09-18, 09:47-10:24)

Owner: "tables is perfect environment, i dont like it. I think more technique boards
should be there ... make puzzle 3, technique 4, mid game 5, opening 4." Same recipe and
the same POOL_LONG start as TECH; only the layout differs. 37 min. Technique attempts
per set per 20 updates rose from 12-20 to 19-39 (about 150-200 per set over the arm).

| set | POOL_LONG | TECH 120 | TECH2 50 | TECH2 100 | TECH2 120 |
|---|---|---|---|---|---|
| Q | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| R | 0.00 | 0.02 | 0.06 | 0.04 | 0.02 |
| RR | 0.06 | 0.10 | 0.08 | 0.10 | 0.06 |
| QR | 0.04 | 0.04 | 0.14 | 0.14 | 0.12 |

Training-window success (guided): Q 0.00-0.15, R 0.00-0.14, RR 0.18-0.34, QR 0.32-0.47.
Other dials at 120: m1 0.782, m2 0.693, m3 0.626, m4 0.525, endgame-mate 0.702, own-game
0.30, finishing 0.34 / 0.11 / 0.10, prior_score 0.48 / 0.49 / 0.54. Games: 17 decisive
of 40, mate 17 of 59.

## Reading

- Doubling the technique games did not move the held-out dial for the single pieces;
  queen-and-rook may have gained (0.04 -> 0.12-0.14, 50 games, borderline). Training
  success stayed where it was, so more games from the same distribution produce the
  same teacher-made wins and the same non-transfer.
- The demonstrator acts only when a mate in one or a forcing mate in two is on the
  board, i.e. in the last one or two plies. The 30-plus plies before that are the
  network's own wandering, and a win at the end spreads its credit over all of them
  with the GAE decay; the approach pattern (drive the king to the edge, keep the
  queen a knight's move away) gets a faint, noisy signal.
- Puzzle boards down from 4 to 3: m1-m4 held (0.782 / 0.693 / 0.626 / 0.525).

## Next model-free levers (owner rejected tablebases)

1. **Start-state curriculum on the generator** (Florensa 2017 without a ruler): draw
   early starts with the weak king already on an edge or in a corner and the strong
   pieces within a few squares, so the mate is a few plies away and the network's own
   moves are what finish it; widen toward random placement as the held-out rate rises
   (adaptive threshold like the finishing depth curriculum). No new RL machinery.
2. **Self-imitation** (Oh et al. 2018, already cited): replay the technique wins the
   network does achieve with the (R - V)+ loss so a rare win is learned from many
   times instead of once. Queued change; fits the sparse-win shape exactly.
3. Longer run on the current recipe: the flat curves over 240 technique updates (TECH
   + TECH2 combined evidence) argue against it as the first move.

# Arm CURSIL: level curriculum + self-imitation (2026-09-18, 10:57-11:42)

Owner: "lets continue with self imitation and start-state curriculum." Same start
(POOL_LONG), budget (120) and layout as TECH2; technique levels 0-3 per set (advance at
0.6 over 30 boards), SIL 4 x 256 per update (weight 0.1, value 0.01, buffer 50k). 44 min
(+19% per update for SIL). Owner's prediction: none. Claude's: queen level 1-2 by 120,
held-out queen 0.05-0.20. Actual: queen at level 3 by update 60, held-out queen 0.02.

| set | TECH2 120 | CURSIL 50 | CURSIL 100 | CURSIL 120 | level reached |
|---|---|---|---|---|---|
| Q | 0.00 | 0.04 | 0.04 | 0.02 | 3 (by update 60) |
| R | 0.02 | 0.02 | 0.06 | 0.00 | 1 (at update 100) |
| RR | 0.06 | 0.08 | 0.06 | 0.12 | 3 (by 40) |
| QR | 0.12 | 0.12 | 0.16 | 0.12 | 3 (by 20) |

Training-window success (guided, mixed levels): Q 0.24-0.51, R 0.39-0.49, RR 0.29-0.55,
QR 0.44-0.68. Other dials at 120: m1 0.778, m2 0.688, m3 0.630, m4 0.544, endgame-mate
0.706, own-game 0.29, finishing 0.32 / 0.16 / 0.11, prior_score 0.44 / 0.50 / 0.53, game
entropy 0.34-0.36 (no collapse). Games: 16 decisive of 40, mate 16 of 82.
SIL: buffer full (50k) from update 60; sil_valid_share 0.96-0.99 throughout; policy loss
0.74-0.89 flat.

## Reading

- **The curriculum raced ahead of the skill.** Queen, two rooks and queen-and-rook hit
  the 0.6 threshold within a few windows and reached level 3, where the held-out dial
  says the network still cannot win. The threshold counted wins the demonstrator made
  (guidance 0.25 plays the mate-in-one / forcing move on these boards), so "60% won at
  level k" measured teacher plus network, not the network. The design's own warning
  ("levels race ahead of skill") came true. Rook v king, where the rules fallback rarely
  finds a forcing line, stayed at level 0 and is the honest one.
- **SIL valid share near 1 is not the alarm I said it was.** Plies are drawn proportional
  to (R - V)+, so the drawn plies have positive advantage by construction; the share of
  the whole buffer with positive priority is the informative number and is not logged.
  SIL's effect on the dials in 120 updates is not visible either way: puzzles, games and
  entropy unchanged. No lock-in, no gain.
- Net: held-out technique unchanged (Q 0.02, R 0.00, RR 0.12, QR 0.12); curriculum
  mechanics work, advancement rule wrong.

## Proposed fix (design change, owner decides)

Advance on **clean episodes only**: count an episode toward the window only if the
demonstrator never acted in it (the rollout knows which boards took a guided pick), or
equivalently measure advancement with guidance off. Raise the threshold to 0.7 over 40
clean boards. Also log `sil_positive_share` (share of the buffer with priority > 0).

# Arm CURSIL2: clean-episode advancement (2026-09-18, 12:26-13:11)

Owner: "Ok, agreed on those." Levels advance on clean episodes only, 0.7 over 40;
sil_positive_share logged. Same start, budget, layout as CURSIL.

Held-out at 120: Q 0.02, R 0.02, RR 0.12, QR 0.08 (unchanged). Levels: every set stayed
at 0 for all 120 updates. Training success at level 0 (all episodes, guided): Q 0.68 ->
0.92, QR 0.93 -> 0.94, RR 0.70 -> 0.80, R 0.52 -> 0.52. sil_positive_share 0.35-0.40
(a third of the buffer still beats the value estimate). Other dials held; prior_score
0.45 / 0.43 / 0.39; game entropy 0.33. Games 11 decisive of 40.

## Reading

- The level-0 positions are being learned: queen wins rose from 0.68 to 0.92 with only
  a quarter of the moves guided. That is the first evidence the network can acquire
  technique when the reward is near.
- The curriculum never advanced because of how "clean" was measured: an episode was
  flagged as teacher-made whenever the played move was in the demonstrator's set. Every
  win ends with a mating move, and the mating move is the demonstrator's move by
  definition, so every win was flagged and only losses and draws counted as clean. The
  window held failures only and the threshold could never be met.
- Fix (CURSIL3): sample the guided mixture by its components (a Bernoulli epsilon draw
  decides whether the demonstrator fires, then the move); flag the episode only when it
  fired. Same distribution and the same stored mixture log-prob, so the PPO ratio is
  unchanged; a network-chosen mating move now counts as the network's. Also log
  `technique_clean_attempts_<set>` and `technique_clean_success_rate_<set>`.

# Arm CURSIL3: clean flag fixed (2026-09-18, 13:14-13:59)

Demonstrator "fires" by an explicit epsilon draw; only then is the episode flagged.
Same start (POOL_LONG), budget (120) and layout.

Levels reached: Q 1 (at update ~100), QR 2 (1 by 20, 2 by 80), RR 1 (by 80), R 0.
Clean success per 20-update window (network alone, mixed levels 0..current):
Q 0.36, 0.53, 0.74 | advance | 0.33, 0.70, 0.21; QR 0.82 | 0.45, 0.68 | 0.33, 0.28, 0.20;
RR 0.37, 0.60, 0.65 | 0.38, 0.20, 0.39; R 0.12, 0.15, 0.17, 0.28, 0.35, 0.13.
Clean attempts per set per window 12-40 (about half of all technique episodes).

Held-out level 3 at 50 / 100 / 120: Q 0.08 / 0.04 / 0.06, R 0.02 / 0.02 / 0.04,
RR 0.06 / 0.04 / 0.04, QR 0.14 / 0.14 / **0.20** (TECH2 0.12, POOL_LONG 0.04).
Other dials at 120: m1 0.780, m2 0.686, m3 0.642, m4 0.546, endgame-mate 0.701, own-game
0.31, finishing 0.39 / 0.13 / 0.13, prior_score 0.41 / 0.48 / 0.49, game entropy 0.35.
Games: **21 decisive of 40** (TECH2 17, CURSIL2 11), mate 21 of 90. sil_positive_share
0.42 -> 0.34.

## Reading

- The curriculum now behaves like a curriculum: clean success climbs at a level (queen
  0.36 -> 0.74 at level 0), the level advances, success drops as harder starts enter the
  mix, and climbs again. This is the shape the design wanted and the first two arms hid.
- Rook v king is the hard rung (clean 0.12-0.35 at level 0): a rook needs the king's
  help for every mate and the cap is generous, so wins are rare and the window is
  slow; leave it, it is the honest measure of long-plan technique.
- Held-out level 3 moved only for queen-and-rook (0.12 -> 0.20); the single pieces are
  still at levels 0-1 after 120 updates, so level-3 transfer is not yet expected.
- Whole-game side effect: 21 decisive of 40, the highest of the project (POOL_LONG 22
  with 800 updates). Puzzle dials held. No SIL lock-in (entropy 0.35).
- Owner's suspicion from the morning holds: 120 updates show the direction, not the
  plateau. A longer run from CURSIL3 is the natural next step.

# Arm CURSIL_LONG: 360 more updates from CURSIL3 (2026-09-18, 14:18-16:32)

Owner: "continue." Named exception (480 technique updates in total from POOL_LONG).
2.2 h. Note: `init_from` loads weights only, so the curriculum levels restarted at 0 and
re-climbed (Q back to 1 by update 40, QR 1 -> 2 at 120 -> 3 at 280, RR 1 -> 2 at 240,
R never left 0). Level state should persist with the checkpoint (follow-up).

Held-out level 3 (50 games per set):

| update | Q | R | RR | QR |
|---|---|---|---|---|
| CURSIL3 end | 0.06 | 0.04 | 0.04 | 0.20 |
| +100 | 0.08 | 0.00 | 0.10 | 0.22 |
| +200 | 0.06 | 0.04 | 0.16 | 0.20 |
| +300 | 0.10 | 0.02 | 0.14 | 0.28 |
| +360 | 0.06 | 0.06 | 0.18 | 0.16 |

Clean success per 40-update window (network alone, levels 0..current mixed):
Q 0.27-0.44 at level 1 for 320 updates (never reached the 0.7 threshold);
R 0.24-0.58 at level 0 (never reached 0.7); RR 0.68 -> level 2, then 0.12-0.32;
QR climbed to level 3 and sits at 0.31-0.53 there.
Other dials at +360: m1 0.776, m2 0.688, m3 0.649, m4 0.549, endgame-mate 0.721 (all
slightly up over POOL_LONG), own-game 0.33, finishing 0.40 / 0.18 / 0.16, prior_score
0.35-0.59 (noisy), game entropy 0.32-0.34, sil_positive_share 0.38 -> 0.36.
Games: 19 decisive of 40, mate 19 of 85.

## Reading

- The curriculum is honest and the learning is slow. Two rooks and queen-and-rook,
  the pairs, climbed to levels 2 and 3 and their held-out rates rose (RR 0.04 -> 0.18,
  QR 0.04 -> 0.28 peak); the single pieces stalled: queen at level 1 with a third of
  clean games won for 300 updates, rook at level 0 at half.
- Why slow: about 64 plies per set per update, roughly two games, half of them clean.
  Over 480 updates the queen saw about 1,000 games, perhaps 500 unguided. A 0.7
  threshold on that trickle is hard to meet; the plateau at 0.3-0.4 may be the skill
  ceiling at this sample rate rather than a true ceiling.
- Whole-game dials did not suffer: puzzles up a point or two, decisive games 19-21 of
  40, no entropy collapse from SIL over 480 updates.
- The finishing conversion target (depth 10 / 20) is unchanged at 0.18 / 0.16. Technique
  on bare kings has not yet transferred to converting human middlegames.

## Follow-ups (owner decides)

1. Persist curriculum levels with the checkpoint (bug-class fix).
2. Advance threshold 0.5 over 40 (the queen sat at 0.3-0.4 for 300 updates; 0.7 assumes a
   faster learner than we have) or a Q-only technique layout to measure the ceiling.
3. Accept technique as a slow background rung, and turn to the conversion target
   directly: SIL is on all boards already, so the next lever there is the demonstrator
   quality on finishing boards (own past wins as the line) or the puzzle mix.

# Arm QONLY: all four technique boards on queen v king (2026-09-18, 16:39-17:24)

Owner: "i would test 2. Im wondering if we didnt reach the limit of current architecture
size." 120 updates from CURSIL_LONG, technique_sets [Q], level start 1, threshold 0.7
over 40 unchanged, so only the queen's sample rate changed (about 4x: 165-281 attempts
per 20 updates, 116-177 of them clean, vs 12-40 clean before).

Held-out Q (level 3, 50 games): 0.06 (start) -> 0.08 (50) -> **0.22** (100) -> 0.12 (120).
Level: 1 -> 2 at about update 60. Clean success: 0.51 / 0.48 at level 1 (was 0.27-0.44
at the same level with a quarter of the games), then 0.31 / 0.31 / 0.33 / 0.29 with
level-2 starts in the mix. Other dials: m1 0.777-0.786, m3 0.647-0.654, own-game
0.28-0.32, finishing d10 / d20 0.17 / 0.10 (0.20 / 0.19 at 100), prior_score 0.43-0.54,
game entropy 0.30 (down from 0.33; watch). Games: 19 decisive of 40, mate 0.28.

## Reading

- Sample rate was a limit. With four times the queen games, the queen cleared the level
  it had sat on for 300 updates, moved to level 2, and the held-out rate left the noise
  floor (0.06 -> 0.12-0.22). The architecture is not yet the binding constraint for
  the queen; the network learns the technique when it sees enough of it.
- The learning is still slow in absolute terms (about 0.3 clean at level 2 after 60
  updates there), which is what the good-starts curriculum and lambda 1 are for.
- Decision: the GSL arm starts from CURSIL_LONG with the same queen-only layout and
  budget as QONLY, so QONLY is the exact control for (good starts + lambda 1.0) vs
  (levels + lambda 0.95).

# Arm GSL: good-starts buckets + gae_lambda 1.0 (2026-09-18, 17:25-18:10)

Owner: "okay so 1 and 2 is next arm." Same start (CURSIL_LONG), budget (120) and
queen-only layout as QONLY, which is the control (levels + lambda 0.95).

| dial | QONLY (control) | GSL |
|---|---|---|
| held-out Q at 50 / 100 / 120 | 0.08 / 0.22 / 0.12 | 0.12 / 0.10 / 0.18 |
| clean training success, windows | 0.51, 0.48, 0.31, 0.31, 0.33, 0.29 | 0.03, 0.05, 0.18, 0.12, 0.12, 0.09 |
| own-game mate-in-one (selfplay_top1) | 0.29 / 0.32 / 0.28 | **0.36 / 0.35 / 0.32** |
| mate found in 40 games | 0.28 | **0.40** (21 of 52) |
| decisive games of 40 | 19 | 21 |
| finishing d10 / d20 at 120 | 0.17 / 0.10 | 0.24 / 0.12 |
| game entropy | 0.30 | 0.29-0.31 |
| clip fraction / approx KL | (not logged here) | 0.19 / 0.033 |

Buckets: 33 of 36 known by update 80; 6-10 in the band at any time; none graduated
(no bucket above 0.9). m1 0.783-0.789, m3 0.65, prior_score 0.48-0.54.

## Reading

- **Held-out queen: a tie** with the control (0.18 vs 0.12 at the end, 0.10 vs 0.22 at
  100; 50 games each, all inside noise). Neither arm is clearly ahead on the target.
- **The good-starts curriculum began at the hard end.** Unknown buckets all weigh 1, so
  the first 40 updates were effectively uniform random placement, and clean success sat
  near 0.05 while the bucket rates were being learned; the level ladder starts cornered
  and had 0.5 clean success from update 1. Florensa starts from the goal state itself and
  expands outward; our uniform prior over unknown buckets dropped that. Then no bucket
  ever exceeded 0.9, so the replay share never engaged, and the band held 6-10 buckets
  at a time: the curriculum was doing its job but from a cold start.
- **Whole-game dials moved**: own-game mate-in-one 0.36 (best of the project, from
  0.30-0.33), mate found in own games 0.40 (from 0.22-0.28), decisive 21 of 40. The
  candidate cause is lambda 1.0 on every board: the win's full credit reaches the moves
  that set up the mate. Both levers changed at once, so this is a hypothesis, and 40
  games is a small sample.
- PPO stayed healthy at lambda 1: clip fraction 0.19, KL 0.033, entropy unchanged.

## Follow-ups (owner decides)

1. **Lambda ablation**: levels curriculum + lambda 1.0, queen-only, same start and budget.
   If own-game mate and in-game mate rate repeat their rise, lambda 1 becomes the
   default for every board (a real gain on the owner's original complaint).
2. **Warm start for good-starts**: weight unknown buckets by an easiness prior (edge
   distance 0 -> 1, 1 -> 0.5, 2 -> 0.25, 3 -> 0.1) until they are known, so the
   curriculum starts near the goal as the paper does and expands outward.

# Arms LAMBDA and GSL2 (2026-09-18, 18:18-19:54)

Owner: "okay agreed on both." Queen-only, 120 updates from CURSIL_LONG. LAMBDA = level
ladder + lambda 1.0 (control QONLY); GSL2 = good starts with the warm-start prior +
lambda 1.0 (control GSL).

| arm | held-out Q 50/100/120 | own-game m1 | decisive of 40 | in-game mate | m1 | entropy |
|---|---|---|---|---|---|---|
| QONLY (levels, 0.95) | 0.08 / 0.22 / 0.12 | 0.29 / 0.32 / 0.28 | 19 | 0.28 | 0.777 | 0.30 |
| LAMBDA (levels, 1.0) | 0.12 / 0.18 / 0.16 | 0.32 / 0.30 / 0.33 | **23** | 0.32 | **0.791** | 0.31 |
| GSL (good starts flat, 1.0) | 0.12 / 0.10 / 0.18 | 0.36 / 0.35 / 0.32 | 21 | 0.40 | 0.783 | 0.29 |
| GSL2 (good starts warm, 1.0) | 0.12 / 0.22 / 0.18 | 0.33 / 0.31 / 0.35 | **23** | 0.31 | 0.784 | 0.29 |

Clip fraction 0.19-0.20 and KL 0.033-0.040 in all four: lambda 1 did not destabilise PPO.
Finishing d10 / d20 wandered 0.13-0.24 in all arms (noise band).

## Reading

- **Lambda 1.0: adopt.** Every lambda-1 arm beat the lambda-0.95 control on decisive
  games (21-23 vs 19), in-game mate rate (0.31-0.40 vs 0.28) and own-game mate-in-one
  (0.30-0.36 vs 0.28-0.32), with the same puzzle dials and a healthy clip band. Held-out
  queen: no separation (all 0.16-0.18 at the end). Small effects each, consistent
  direction across three arms; it becomes the default for the long run.
- **Warm start did not lift the good-starts curriculum**: clean success stayed 0.06-0.13
  (GSL 0.03-0.18). The per-bucket table from GSL2's curriculum.json explains why and is
  the most honest picture of the technique so far: every one of the 33 feasible buckets,
  including all nine with the weak king on the edge, has a clean success of 0.00-0.25
  (median 0.05). The level ladder's 0.4-0.5 "clean success at level 1" was an average
  over the mix of level-0 corner starts (mate in two or three, won at about 0.9) and
  level-1 edge starts (won at about 0.1). The network has learned to finish a cornered
  king; it has not learned to drive a king to the edge. Held-out level 3 at 0.15-0.2 is
  the same story.
- **Curriculum choice for the long run: levels.** Held-out tie, simpler, and it produces
  many more wins per update (level-0 starts) for self-imitation to replay. Good starts
  stays available as `technique_curriculum: good_starts` and its bucket table is the
  diagnostic to re-run at the end of the long run.
- The three infeasible buckets are the expected ones (central king with a piece five or
  more squares away).

## Next: LONG2 (owner: "some kind of long run after this ... increase LR a bit if the
run is moving slowly only"; "option 1")

600 updates from LAMBDA, default layout (3 puzzle / 4 technique on all four sets / 5
mid-game / 4 opening), pool, SIL, guidance 0.25, levels curriculum from level 0 for
every set (levels persist from now on), gae_lambda 1.0, min_lr 1e-4 (no decay). Then
games, finishes, probe, win matrix vs SL / M34 / POOL_LONG / CURSIL_LONG.

# LONG2: consolidation run (2026-09-18 19:57 to 23:37)

600 updates from LAMBDA on the real layout (3 puzzle / 4 technique on Q, R, RR, QR /
5 mid-game / 4 opening), pool, SIL, guidance 0.25, level ladder from 0 (persisted),
gae_lambda 1.0, min_lr 1e-4 (no decay; owner "option 1"). 3.6 h training, 4 min
evaluations. Owner's prediction: none. Claude's (implicit, from the arms): puzzles hold
or creep, own-game up, prior score par, technique pairs up, finishing unchanged.

## Dials at 600 (best previous in brackets)

| dial | LONG2 | best before |
|---|---|---|
| mate-in-1 | **0.802** | 0.796 (LAMBDA 0.791; POOL_LONG 0.785) |
| mate-in-2 | **0.712** | 0.697 |
| mate-in-3 | **0.665** | 0.663 |
| mate-in-4 | **0.574** (0.587 at 500) | 0.566 |
| endgame-mate | **0.742** | 0.737 |
| own-game mate-in-one | 0.333 (0.343 at 400 / 500) | 0.357 (GSL, one eval) |
| technique held-out Q / R / RR / QR | **0.26** / 0.02 / 0.18 / **0.28** (QR 0.34 at 550) | 0.22 / 0.06 / 0.18 / 0.28 |
| levels reached | Q 2, R 1, RR 2, QR 2 | |
| finishing d2 / d10 / d20 | 0.40 / 0.13 / 0.12 | unchanged band |
| prior_score (20 games per colour) | 0.45 (0.44-0.59 across the run) | |
| game entropy / clip / KL | 0.30 / 0.19 / 0.030 | healthy throughout at LR 1e-4 |
| 40 sampled games | 17 decisive, 22 draws, mate 17 of 54 (0.31), 129 plies | |

## Win matrix (30 games per colour, decisive W-L, rest drawn or capped)

|  | vs SL | vs M34 | vs POOL_LONG | vs CURSIL_LONG | mean score |
|---|---|---|---|---|---|
| SL (prior) | - | 19-9 | 8-11 | 6-7 | 0.510 |
| M34 | 9-19 | - | 6-17 | 7-28 | 0.346 |
| POOL_LONG | 11-8 | 17-6 | - | 8-20 | 0.444 |
| CURSIL_LONG | 7-6 | 28-7 | 20-8 | - | 0.542 |
| LONG2 | **8-7** | **36-4** | **33-4** | **24-10** | **0.658** |

## Reading

- **LONG2 is the strongest checkpoint of the project on every axis but one.** It beats
  M34 36-4, POOL_LONG 33-4 and CURSIL_LONG 24-10 in decisive games, is at par with the
  supervised prior (8-7, score 0.51; the drift the win matrix exposed on 2026-09-17 is
  closed), and holds the best value on all five puzzle sets. The RL family now orders
  cleanly by training time (M34 < POOL_LONG < CURSIL_LONG < LONG2), the opposite of the
  flat plateau seen in the first matrix.
- **Lambda 1 and the LR floor held up over 600 updates**: clip fraction 0.18-0.21, KL
  0.030-0.046, entropy 0.29-0.32, puzzle dials still rising at the end.
- **Technique**: queen 0.16 -> 0.26 held-out, the pairs at 0.18-0.34; all sets except the
  rook at level 2. The rook reached level 1 at update 500. The same "cornered king only"
  ceiling is visible: clean success 0.3-0.5 at level 2 with the level-0 share inside it.
- **Finishing conversion at depth 10 / 20 is unchanged for the fourteenth arm** (0.13 /
  0.12, band 0.10-0.24). Every other dial moved; this one did not. It is the human-
  middlegame conversion problem and neither technique nor lambda reached it.
- Sampled games got drawier (22 of 40) while head-to-head decisive counts against the
  other checkpoints rose sharply: LONG2 wins when there is a weaker side, and draws
  itself.

## Defaults after this run

`gae_lambda` 1.0 becomes the PPO default (three arms plus LONG2). `min_lr` stays 3e-5 in
the config; long runs set 1e-4 explicitly (owner's option 1) until a run at the default
shows a difference.

# LONG3: LONG2 + 600 updates, mixed finishing slot (2026-09-19, 01:39-05:19)

Finishing boards draw technique and human starts 0.5 / 0.5; everything else as LONG2.

| dial | LONG2 | LONG3 |
|---|---|---|
| mate-in-1 / 2 / 3 / 4 | 0.802 / 0.712 / 0.665 / 0.574 | **0.808 / 0.723 / 0.691 / 0.590** |
| endgame-mate | 0.742 | **0.747** |
| own-game mate-in-one | 0.333 | **0.387** |
| technique Q / R / RR / QR | 0.26 / 0.02 / 0.18 / 0.28 | 0.16 / 0.06 / 0.10 / 0.38 |
| finishing d2 / d10 / d20 (sampled) | 0.40 / 0.13 / 0.12 | 0.43 / 0.13 / 0.11 |
| calibration human-line d2 / d10 / d20 | 0.75 / 0.28 / 0.22 | 0.77 / 0.33 / 0.14 |
| vs LONG2 (decisive) | - | **19-10** |
| vs SL prior (decisive) | 8-7 (yesterday), 8-13 (today) | 6-12 |
| game entropy | 0.30 | 0.27-0.29 (lowest so far) |
| 40 games | 17 decisive | 19 decisive, mate 0.23 |

Finishing depth curriculum: stayed at depth 8 for all 600 updates (advance needs 0.6
over 100 episodes, all episodes counted; training success at d6 / d8 0.05-0.36).

## Reading

- **Puzzle and own-game dials keep rising** (own-game mate-in-one 0.39, best of the
  project) and LONG3 beats LONG2 19-10, so 1,200 updates of the consolidated recipe are
  still paying on everything that has a training signal.
- **Finishing d10 / d20 unchanged for the fifteenth arm, and now we know one more reason:
  the finishing depth curriculum never passed depth 8**, so no training board ever started
  10 or 20 plies before the mate. The evaluation asks for a depth the training never
  visited. The threshold (0.6 over 100, teacher-made wins counted) is the same rule that
  misled the technique ladder before the clean-episode fix.
- **Against the prior, back to slightly negative** (6-12; the matrix score against SL
  swung 0.45-0.51 across three sessions, so par is the honest description). Entropy is
  drifting down slowly; watch it.
- Technique held-out is noisy (Q 0.08-0.26 within one run at 50 games); the pairs hold.

## Next (owner decides)

1. **Train at the evaluated depths**: sample the finishing depth uniformly over [0, 20]
   (curriculum off, or start at 20), so d10 / d20 boards exist in training. Cheapest
   possible fix for a dial that has had no matching training signal.
2. **Self-demonstrations** on finishing boards (own winning line replaces the human's
   once one exists for the record), which the calibration says targets the right ply.
3. Entropy guard: if it slides under 0.27, raise entropy_coef or shorten the run.

## DEPTH20 (2026-09-19): LONG3 + 120 updates, human finishing depth uniform over [0, 20]

Owner: "lets go with option 1. The model has to know the idea if it can mate in 10 so
expanding it further should work." Config change only: `finish_depth_start = finish_depth_max
= 20`, so the curriculum cannot move and depths 10 and 20 appear in training for the first time
(LONG3's curriculum sat at 8 all run). Everything else as LONG3.

| dial | LONG3 | DEPTH20 |
|---|---|---|
| finishing d2 / d10 / d20 (held-out) | 0.43 / 0.13 / 0.11 | 0.40 / **0.20** / 0.13 |
| mate-in-1 / 2 / 3 / 4 / endgame (held-out top-1) | 0.808 / 0.723 / 0.691 / 0.590 / 0.747 | 0.804 / 0.723 / 0.683 / 0.580 / 0.751 |
| own-game mate-in-one | 0.387 | 0.352 |
| prior_score (20 games) | 0.44 | 0.41 |
| head to head, decisive (DEPTH20 view) | - | 12 to 9 for DEPTH20 |
| game entropy | 0.283 | 0.288 |
| training finishing attempts per depth (last update) | 0-8 only | 5-9 per depth up to 20 |

Reading: the first movement on depth 10 in the project's history (0.13 -> 0.20 after 120
updates, 100 held-out games, so roughly +-0.04 noise: real but modest), depth 20 within noise.
Puzzles held within noise; own-game mate-in-one down 0.03 (noise band). Training success at the
new depths after 120 updates: 0.25-0.6 on 5-9 attempts each, i.e. the boards are being won, so
the signal exists. Verdict: option 1 works as a mechanism and costs nothing; the depth-20 dial
needs more than 120 updates or the punisher. DEPTH20 is the start checkpoint for PUNISH.
