## Why

The SL + PPO 600 model (Lichess m1 0.57, m2 0.51) plays a recognisable game but
does not finish: in its own games it found 7 of 100 available mates-in-one and
75% of games ended by the ply cap. Two causes were established on 2026-09-17:
only 23% of the Elo-1500+ human games it was pretrained on end in checkmate
(683 of 3,000; the rest resign or run out of time), and mate recognition learned
on curated Lichess puzzles does not transfer to positions the model creates
itself (pretrained model 5/139, from-scratch model 3/31, near random). The owner
chose to fix this before NFSP because the weakness is in the reward signal, not
in the opponent.

## What Changes

- New data: the last 40 plies of human games that end in checkmate (both Elo
  >= 1500), split by game into train / held-out files.
- New board type: **finishing boards** start k plies before a human mate with
  the winner to move and run as full games under a per-board ply cap of
  k + margin; not mating by the cap is a draw (-0.5), so the cap is the
  pressure to finish. The defender is the same network.
- Reverse curriculum on k (Florensa et al. 2017): depth is sampled uniformly
  from {0, 2, ..., k_current}; k_current starts at 2 and rises by 2 whenever the
  rolling success rate over the last window of finishing episodes exceeds a
  threshold, up to a ceiling.
- Board layout becomes puzzle boards, finishing boards, mid-game boards,
  opening boards. Default 4 / 4 / 2 / 2 (owner: 4 puzzle, 4 finishing, 4 game).
- New training metrics: finishing attempts, success rate, success by depth,
  current curriculum depth. Finishing results stay out of the game win/draw rates.
- New held-out evaluation: conversion rate from rewound positions at fixed
  depths (2, 10, 20), reported every `eval_interval` next to the puzzle sets.
- The self-play mate-in-one evaluation set is re-harvested from the current
  checkpoint (the existing file came from an early model and holds opening
  blunders), so the own-game mate rate is tracked during training.
- Finishing boards are supported by the sync vector env; the async env rejects
  them (curriculum state would be split across workers).

## Capabilities

### New Capabilities
- `finishing-curriculum`: rewound-mate start positions, depth curriculum,
  per-board cap, success accounting, configuration and data preparation.

### Modified Capabilities
- `puzzle-evaluation`: adds the held-out finishing conversion metrics.

## Alternatives considered

- **Resignation positions as starts.** Plentiful (1,251 of 3,000 sampled games)
  but the distance to mate is unknown and often long at this level; no
  curriculum control. Salimans & Chen (2018) argue exploration cost scales with
  the distance between the start state and the reward, so an uncontrolled
  distance defeats the purpose.
- **Synthetic endgames (KQ vs K, KR vs K).** Guaranteed won, but a narrow skill;
  the observed failure is converting middlegames the model reached itself.
  Florensa et al. (2017) motivate starting from real goal states of the task
  ("obtaining a single state in which the task is achieved") and expanding
  backwards, which the rewound human mates provide directly.
- **Own-game positions as training starts.** Best distribution match but they
  depend on the checkpoint and drift; the 2026-09-16 A/B that mixed harvested
  positions into puzzle training raised accuracy on them without moving the
  in-game mate rate. Kept as the evaluation distribution instead.
- **NFSP first.** Heinrich & Silver (2016) change the opponent distribution,
  not the reward; would leave the mate rate near 7%. Deferred, still agreed as
  the following step.

## How we will know it worked

Arm: 120 updates (standard budget) from the SL + PPO 600 checkpoint with the
new default layout. Metrics to watch:

- `finish_rate_d2 / d10 / d20` on held-out rewound positions (baseline:
  the starting checkpoint evaluated before training).
- `selfplay_top1` on the re-harvested own-game mate-in-one set, and the
  own-game mate-in-one rate from 40 sampled games afterwards (baseline 0.07).
- Curriculum depth reached by update 120 (start 2, ceiling 40).
- Lichess m1 / m2 must not drop more than a few points (forgetting check).

Owner's prediction: skipped at the owner's request ("agreed, all seems good");
Claude's stated expectation: own-game mate-in-one rate above 0.2 at update 120,
depth between 8 and 16, d2 above 0.8. If depth stays at 2 by update 120 the
standard budget missed the knee and a longer run is the named exception.

## What to understand

- Reverse curriculum: why starting near the goal turns an exponential search
  into a short one, and what the adaptive threshold does.
- Why the ply cap is the reward for finishing here (draw -0.5 vs win +2), and
  why the material potential does not undo it (terminal potential is zero at
  the cap too).
- Distribution shift: why puzzle accuracy did not transfer to own games, and
  why the evaluation set is harvested from the model's own play.
- What the defender learns on the same board and why that is wanted.
