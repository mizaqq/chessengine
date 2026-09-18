## Context

`TechniqueSampler` draws a set and calls `generate_position` with random squares;
`FinishStart` carries fen, cap, winner, line, config; the vector env reports labelled
results through `finish_config` and calls `finish_sampler.report(depth, success)`
(a no-op today). The rollout keeps `StepRecord`s per step across all boards; episodes
cross rollout boundaries (a game lasts 140 plies, a rollout 64). PPO runs 4 epochs x 4
minibatches per update on the learner's plies.

## Goals / Non-Goals

**Goals:** per-set level curriculum in the generator; SIL buffer, loss and schedule;
diagnostics; one arm against TECH2.

**Non-Goals:** tablebases; changing the demonstrator; SIL on puzzle boards being
special-cased (they are included like every board); prioritised-replay bias correction
beyond the paper's simple exponent.

## Decisions

- **Levels in the generator.** `generate_position(material, rng, level)` picks the weak
  king's square from the level's set (corners / edge / one-from-edge / any), then the
  strong king from squares at the required distance, then the pieces from squares within
  the required distance, rejecting as today (legality, check, mate in one). Level 3 is
  the current generator.
- **Curriculum state in the sampler.** `TechniqueSampler.report(depth, success)` gets
  the label through `report_label(label, success)` (the env passes the label when the
  start has one); per set a deque of the last `technique_window` outcomes; advance when
  mean >= `technique_advance_rate`; `current_level(label)`; `levels()` dict for the
  summary. Draw level uniformly in [0, current] as the finishing depth does.
- **SIL buffer** (`src/training/sil.py`): `EpisodeAccumulator` keeps per-board lists of
  (obs, legal, action, mover) for learner plies; on `done` it computes R backwards from
  the terminal rewards (mover's side, gamma per ply) and pushes to `ReplayBuffer`
  (tensors in a ring of `sil_buffer` plies with a priority array). Terminal rewards are
  the training ones (win 2, loss -2, draw -0.5, puzzle miss 0), so R is on V's scale.
- **SIL update** after the PPO update: `sil_updates` minibatches of `sil_batch`, sampled
  proportional to priority^1.0 (paper: best of {0.6, 1.0}), bias-correction weights
  (N p)^-0.1 normalised by their max (paper 0.1); loss as the spec; one optimizer step
  per minibatch with the same grad clip; priorities refreshed for the drawn rows.
  Paper's PPO table: M 10, batch 512, loss weight 0.1, value weight 0.01, buffer 50k,
  for 20k on-policy samples per iteration; ours has 1,024 per update, so M 4 x 256.
- **BatchNorm**: SIL passes run in eval mode like the PPO update; `refresh_norm_stats`
  stays on the on-policy batch only.
- **Config**: `technique_curriculum: true`, `technique_level_start: 0`,
  `technique_advance_rate: 0.6`, `technique_window: 30`; `sil_updates: 4`,
  `sil_batch: 256`, `sil_loss_weight: 0.1`, `sil_value_weight: 0.01`, `sil_buffer: 50000`.

## Hyperparameters

- `technique_advance_rate` / `technique_window`: how sure we are before the level
  rises. Low rate or short window = levels race ahead of skill (the finishing curriculum
  at 0.6/100 never advanced past depth 8; here the window is per set, so 30).
- `technique_level_start`: 0 = start with cornered kings; 3 = no curriculum.
- `sil_updates` x `sil_batch`: how much off-policy replay per update. 0 = off; large =
  the paper's lock-in ("gets stuck at a sub-optimal policy", §5.4), watched through game
  entropy and prior_score.
- `sil_loss_weight`: 0.1 per the PPO table; 1.0 (A2C table) doubles the pull toward
  past wins.
- `sil_value_weight`: 0.01; larger drags V up toward the best past returns and
  shrinks (R - V)+, switching SIL off by itself.
- `sil_buffer`: 50,000 plies is about 50 updates of learner plies; smaller forgets old
  wins, larger replays stale behaviour.

## Risks / Trade-offs

- SIL replays returns from games against pool opponents and the current self alike;
  the return is what happened, not what would happen now. Accepted by the paper's
  lower-bound argument.
- Level-0 positions are easy enough that the demonstrator's mate-in-one fires almost at
  once; the curriculum must move past level 0 for the network's own approach to be
  tested, hence the held-out stays at level 3.
- Two levers in one arm: if the arm works we will not know which lever did it. Owner
  chose both; an ablation (curriculum only) is a follow-up if needed.
- Compute: 4 extra minibatches of 256 per update, about +15% update time.
