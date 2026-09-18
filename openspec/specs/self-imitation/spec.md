# self-imitation Specification

## Purpose
Learn from the learner's own past good episodes more than once: a replay buffer with
Monte Carlo returns and an off-policy loss that trains only on plies whose past return
beat the current value estimate (Oh, Guo, Singh & Lee, ICML 2018).

## Requirements

### Requirement: Replay buffer of finished episodes
Every learner ply (observation as the network saw it, legal mask, action) SHALL be
held per board until that board's episode ends; then each ply SHALL be stored with its
discounted return R from the mover's side (the episode's terminal reward for the mover,
discounted by gamma once per own ply to the end, the game-ending ply included, as the
PPO value targets are; a ply of the side that lost has a negative R).
Plies chosen by a pool opponent SHALL NOT be stored. The buffer SHALL hold at most
`sil_buffer` plies and SHALL drop the oldest beyond that.

#### Scenario: three-ply win
- **WHEN** a board's episode is white, black, white and ends in white's mate, terminal
  reward +2 for the winner and -2 for the loser, gamma 0.99
- **THEN** the stored returns are 2 * 0.99^2 for the first white ply, -2 * 0.99 for the
  black ply, and 2 * 0.99 for the mating ply (gamma once per own ply including the
  game-ending one, the convention of the PPO value targets, so R is on V's scale)

#### Scenario: opponent plies excluded
- **WHEN** a pool board's episode has plies by the frozen opponent
- **THEN** only the learner's plies of that episode enter the buffer

### Requirement: Self-imitation update
After each policy update, `sil_updates` minibatches of `sil_batch` plies SHALL be
drawn from the buffer with probability proportional to the stored priority
max(R - V, 0) (a ply with priority 0 SHALL not be drawn while any positive priority
exists), and the network SHALL be trained on
`sil_loss_weight * ( -log pi(a|s) * (R - V)+ + sil_value_weight * 1/2 (R - V)+^2 )`
with V the current value estimate and `(x)+ = max(x, 0)`, so plies with R <= V give
no gradient. The priority of a drawn ply SHALL be refreshed to its new (R - V)+.
`sil_updates` of 0 SHALL disable the buffer and leave training unchanged.

#### Scenario: only better-than-expected plies train
- **WHEN** a minibatch holds one ply with R = 1.5 and V = 0.5 and one with R = -1 and V = 0
- **THEN** the policy term is -log pi(a|s) * 1.0 for the first ply and 0 for the second

#### Scenario: disabled
- **WHEN** `sil_updates` is 0
- **THEN** no buffer is kept and the update sequence is identical to today's

### Requirement: Diagnostics
The summary SHALL report `sil_buffer_size`, `sil_valid_share` (share of drawn plies
with R > V), `sil_positive_share` (share of all stored plies with positive priority),
`sil_policy_loss` and `sil_value_loss` when the buffer is active.

#### Scenario: keys present
- **WHEN** `sil_updates` is 4 and one update has run
- **THEN** the summary contains the five keys with finite values
