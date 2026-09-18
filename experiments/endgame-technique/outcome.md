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
