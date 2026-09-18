## Why

Arms TECH and TECH2 (2026-09-18): generated bare-king technique boards produced wins
in training (up to 0.47 per window for queen and rook) but the held-out network alone
still cannot finish (queen v king 0.00, rook v king 0.02 after 240 combined technique
updates). The wins were made by the rules demonstrator in the last one or two plies and
were seen once each; the thirty plies of approach before them got a faint, discounted
signal. The owner ruled out tablebases and search ("perfect environment") and chose
the two remaining model-free levers together: "lets continue with self imitation and
start-state curriculum".

## What Changes

- **Start-state curriculum on the technique generator.** Each material set has a
  difficulty level: 0 = weak king in a corner with the strong king and pieces close,
  1 = weak king on an edge, 2 = weak king one step from an edge, 3 = fully random
  (today). A set's level rises when the windowed training success reaches a threshold;
  starts are drawn uniformly over levels 0..current so easy positions stay in the mix
  (as the finishing depth curriculum does). Held-out evaluation stays at level 3, the
  real target. Reported: `technique_level_<set>`.
- **Self-imitation (Oh et al. 2018).** A replay buffer of the learner's own plies with
  the episode's Monte Carlo return R. After each PPO update, M extra minibatches are
  sampled from the buffer with priority (R - V)+ and trained with the SIL loss:
  policy -log pi(a|s) (R - V)+ and value 1/2 ||(R - V)+||^2. Only plies whose past
  return beat the current value estimate get gradient, so a rare win is learned from
  many times. No importance ratio (the paper's argument: the objective is a lower bound
  on the optimal soft Q-value, valid for any behaviour that did better than the learner).
  Applies to all learner plies on all boards; pool-opponent plies are excluded.

## Capabilities

### New Capabilities
- `self-imitation`: the replay buffer, the SIL loss and update schedule, its diagnostics.

### Modified Capabilities
- `endgame-technique`: technique starts carry a difficulty level with an adaptive per-set
  curriculum.

## Alternatives considered

- **Tablebase demonstrator** (Syzygy via python-chess): perfect move and distance in
  every technique position. Rejected by the owner: a perfect environment model.
- **Longer run on the current recipe**: TECH + TECH2 curves flat over 240 technique
  updates; wins came from the teacher.
- **Count-based / curiosity exploration** (Bellemare et al. 2016; Ostrovski et al. 2017,
  cited in Oh et al. 2018 §5.3 as complementary): not fetched; SIL's paper reports it
  competitive with these on hard-exploration Atari without a bonus. Deferred.
- **Reverse curriculum with a ruler** (Florensa et al. 2017 uses Brownian rollouts from
  the goal): our generator has no reverse model of chess; the level heuristic (king on
  edge/corner) is the ruler-free stand-in.

## How we will know it worked

Arm CURSIL: 120 updates from POOL_LONG, default layout (3 puzzle / 4 technique / 5
mid-game / 4 opening), pool, guidance 0.25. Same start and budget as TECH2, so TECH2 is
the control.
- `technique_rate_Q / R / RR / QR` at level 3 (held-out): TECH2 ended 0.00 / 0.02 / 0.06 / 0.12.
- `technique_level_<set>` in training: does any set leave level 0?
- `finish_rate_d10 / d20` (human finishing): 0.11 / 0.10 at TECH2.
- SIL diagnostics: `sil_valid_share` (plies with R > V among sampled), `sil_buffer_size`.
- Guard dials: m1-m4, endgame-mate, `prior_score`, game entropy (SIL §5.4 lock-in risk).

Owner's prediction: _____ (not given).
Claude's prediction: level 0 queen positions are won early (training success > 0.6 by
update 40) and the set reaches level 1-2 by 120; held-out level-3 queen rate 0.05-0.20;
rook slower; SIL raises own-game decisive rate a little; finishing d10 unchanged.

## What to understand

- Why a start-state curriculum changes what the network's own moves are responsible for
  (the reward is a few plies away, so the approach is learned from the end backwards).
- SIL's clip: why only (R - V)+ > 0 samples train, what "R" is here (Monte Carlo return
  of the finished episode from the mover's side), and why no importance ratio is needed.
- The lock-in risk (SIL §5.4) and the dials that would show it (entropy, prior score).

## Impact

`src/envs/technique.py` (levels, curriculum), `src/training/sil.py` (buffer, loss),
`src/model/training.py` (episode accumulation per board, SIL updates after PPO),
`src/training/metrics.py` (SIL and level keys), `src/entrypoints/train.py` and
`src/configs/train_default.yaml` (keys `technique_curriculum`, `technique_advance_rate`,
`technique_window`, `sil_updates`, `sil_batch`, `sil_loss_weight`, `sil_value_weight`,
`sil_buffer`), tests for each.
