## MODIFIED Requirements

### Requirement: Pool boards and learner colour
The first `opponent_pool_boards` game boards (after the puzzle and finishing boards)
SHALL be pool boards. When a pool board starts a game it SHALL draw one pool member as the
opponent according to `opponent_pool_sampling` and SHALL draw the learner's colour uniformly at
random. With `uniform` (default) every member is equally likely. With `pfsp` member `m` is drawn
with probability proportional to `(1 - w_m)^pfsp_power + pfsp_min_weight`, where `w_m` is the
learner's smoothed win rate against `m` (exponential moving average with weight 0.1 of the
learner's score in finished pool games against `m`, initial value 0.5; defaults `pfsp_power` 2,
`pfsp_min_weight` 0.05). The assignment SHALL hold until that game ends. `opponent_pool_boards`
of 0 SHALL disable the pool and leave training unchanged.

#### Scenario: uniform draw
- **WHEN** `opponent_pool_sampling` is `uniform` and the pool holds the prior and two snapshots
- **THEN** each of the three is drawn with probability 1/3

#### Scenario: prioritised draw
- **WHEN** `opponent_pool_sampling` is `pfsp`, power 2, floor 0.05, and the smoothed win rates are prior 0.3, snap_a 0.5, snap_b 1.0
- **THEN** the unnormalised weights are 0.49 + 0.05, 0.25 + 0.05, 0 + 0.05, so the prior is drawn with probability 0.54/0.89 = 0.61, snap_a 0.34, snap_b 0.06

#### Scenario: win rate update
- **WHEN** the learner's smoothed win rate against the prior is 0.5 and it loses a pool game against the prior
- **THEN** the smoothed win rate becomes 0.9 x 0.5 + 0.1 x 0 = 0.45
