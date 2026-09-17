## MODIFIED Requirements

### Requirement: Rollout stores what re-evaluation needs
For every step the rollout SHALL store the observation as the acting model saw it,
the legal mask, the action, the log-probability of the action under the
distribution it was drawn from (the network's policy, or the behaviour mixture on
boards with exploration noise), and the value at collection, without an autograd
graph. The update SHALL re-run the network on the stored observations.

#### Scenario: noisy board
- **WHEN** a move on a noisy board is drawn from `b = 0.75 * pi + 0.25 * noise` and pi(a) = 0.01, noise(a) = 0.20
- **THEN** the stored log-probability is log(0.0575), not log(0.01)

## ADDED Requirements

### Requirement: Exploration noise on curriculum boards
When `explore_epsilon` > 0, moves on the selected boards (`explore_boards`:
`puzzle` = puzzle boards, `curriculum` = puzzle and finishing boards, `all` = every board) SHALL be drawn
from `(1 - explore_epsilon) * pi + explore_epsilon * Dir(explore_alpha)`, where
the Dirichlet sample is over the legal moves only. The network's probabilities,
logged entropy and evaluation SHALL be unaffected. `explore_epsilon` SHALL be in
[0, 1), `explore_alpha` > 0. The update summary SHALL report the share of
noisy-board moves whose network probability was below 0.05.

#### Scenario: off by default
- **WHEN** `explore_epsilon` is absent or 0
- **THEN** moves are drawn from pi and stored log-probabilities equal log pi(a)

#### Scenario: only legal moves receive noise
- **WHEN** a position has 3 legal moves out of 4674 actions
- **THEN** the behaviour distribution is zero on the other 4671 actions and sums to 1

#### Scenario: opening boards untouched
- **WHEN** `explore_boards` is `curriculum` and a board starts from the opening
- **THEN** that board's moves are drawn from pi
