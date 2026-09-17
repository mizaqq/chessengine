## Purpose
Game boards can pit the learner against frozen past policies (the supervised prior
and recent snapshots of itself) so that drifting away from strong play costs games
instead of going unnoticed in self-play against an identical copy.

## ADDED Requirements

### Requirement: Pool membership
The opponent pool SHALL contain the supervised prior network and the most recent
`opponent_pool_size` snapshots of the learner, one snapshot taken every
`opponent_snapshot_every` updates. Snapshots SHALL be frozen copies (no gradient,
evaluation mode). When the pool is full the oldest snapshot SHALL be dropped.

#### Scenario: pool fills and rolls
- **WHEN** `opponent_pool_size` is 4, `opponent_snapshot_every` is 50, and training
  reaches update 250
- **THEN** the pool holds the prior plus snapshots from updates 100, 150, 200 and 250,
  and the update-50 snapshot has been dropped

#### Scenario: before the first snapshot
- **WHEN** training is at update 30 with the same settings
- **THEN** the pool holds only the prior, and every pool board plays the prior

### Requirement: Pool boards and learner colour
The first `opponent_pool_boards` game boards (after the puzzle and finishing boards)
SHALL be pool boards. When a pool board starts a game it SHALL draw one pool member
uniformly at random as the opponent and SHALL draw the learner's colour uniformly at
random. The assignment SHALL hold until that game ends. `opponent_pool_boards` of 0
SHALL disable the pool and leave training unchanged.

#### Scenario: assignment on reset
- **WHEN** a pool board's game ends
- **THEN** a new opponent and a new learner colour are drawn for the next game on that
  board, and other boards are unaffected

#### Scenario: pool off
- **WHEN** `opponent_pool_boards` is 0
- **THEN** no snapshot is taken, no prior is loaded, and all plies enter the batch as
  today

### Requirement: Opponent plies leave the batch
On a pool board, the ply SHALL be chosen by the opponent network whenever the side to
move is not the learner's colour, and by the learner otherwise. Plies chosen by an
opponent SHALL be excluded from the policy and value update and from the entropy and
clean-sample diagnostics. The learner's advantages, returns and bootstrap values on
that board SHALL be computed as today from the learner's own network.

#### Scenario: learner is black on a pool board
- **WHEN** a 4-ply rollout on a pool board has the learner playing black and the
  opponent white, plies alternating white, black, white, black
- **THEN** the batch receives the two black plies with the learner's log-probabilities
  and values, and neither white ply

#### Scenario: bootstrap with opponent to move
- **WHEN** the rollout ends on a pool board with the opponent to move
- **THEN** the learner's bootstrap value is the negated value the learner's own network
  assigns to that position seen from the mover

### Requirement: Pool results are reported
Every finished pool-board game SHALL be recorded under the opponent's name with the
learner's score (win 1, draw 0.5, loss 0). The periodic summary SHALL report
`pool_games`, `pool_score`, and `pool_score_<name>` for each member seen. These games
SHALL NOT count in the self-play win, loss and draw rates.

#### Scenario: two games against the prior
- **WHEN** the learner wins one game and draws one against the member named `prior`
  in the same logging window
- **THEN** the summary shows `pool_games` 2, `pool_score` 0.75, `pool_score_prior` 0.75

### Requirement: Score against the prior in evaluation
When `opponent_prior` is set, the periodic evaluation SHALL play `prior_eval_games`
games per colour between the learner and the prior, both sampling, and report
`prior_score` (learner's score, draws and capped games 0.5), `prior_wins` and
`prior_losses`.

#### Scenario: even match
- **WHEN** 20 games per colour give the learner 6 wins, 6 losses and 28 draws
- **THEN** `prior_score` is 0.5, `prior_wins` 6, `prior_losses` 6
