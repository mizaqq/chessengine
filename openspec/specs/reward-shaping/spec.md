# reward-shaping

## Purpose

Defines how each model's per-move reward is formed from material potentials and
terminal outcomes in a self-play rollout, and how that shaping is configured.

## Requirements

### Requirement: Environment reports material balance
Each environment step SHALL report the material balance of the returned position
from white's perspective using piece values queen 9, rook 5, bishop 3, knight 3,
pawn 1, king 0, and SHALL NOT report a per-step material difference.

#### Scenario: initial position
- **WHEN** an environment is reset
- **THEN** the reported material is 0

#### Scenario: after a capture
- **WHEN** white captures a black knight and no other material changes
- **THEN** the reported material is 3

### Requirement: Shaped reward per own move
For a model, the reward credited to its move at position `s` SHALL be
`shaping_scale * (gamma * Phi(s') - Phi(s))` where `Phi` is the reported material
converted to the model's perspective (negated for black) and `s'` is the next
position in the rollout where the model is to move. Discounting SHALL be applied
once per own move as before.

#### Scenario: capture and recapture nets to zero for the first mover
- **WHEN** white to move at material 0 captures (material becomes 3), black to move
  recaptures (material becomes 0), white is to move again, gamma 1.0 and scale 1.0
- **THEN** white's shaped reward for the capture is 0 - 0 = 0 and black's shaped
  reward for the recapture is (0) - (-3) = 3

#### Scenario: discounted potential
- **WHEN** white to move at material 2, next white decision at material 5, gamma 0.9,
  scale 1.0
- **THEN** white's shaped reward is 0.9 * 5 - 2 = 2.5

#### Scenario: scale applies to the shaping only
- **WHEN** the same transition is used with scale 0.5
- **THEN** the shaped reward is 1.25 and any terminal reward is unchanged

### Requirement: Terminal position has zero potential
When a step is `done`, the potential of the position after it SHALL be 0 for both
models, so the last own move before a game end receives `-shaping_scale * Phi(s)`
plus the model's terminal reward. Potentials from a later game SHALL NOT be used.

#### Scenario: mating while ahead
- **WHEN** white to move at material 9 delivers checkmate, white terminal reward 2,
  gamma 1.0, scale 1.0, bootstrap 5
- **THEN** white's return at that move is (0 - 9) + 2 = -7 and the bootstrap has no
  effect

#### Scenario: opponent mates
- **WHEN** white moves at material 0, black then captures a rook and mates (`done`),
  white terminal reward -2, gamma 1.0, followed by a new game
- **THEN** white's return at its move is (0 - 0) + (-2) = -2 regardless of the new
  game's material

#### Scenario: shaping sums to zero over a complete game
- **WHEN** a complete game is played from material 0 to a terminal position with
  any sequence of captures, gamma 1.0
- **THEN** the sum of a model's shaped rewards over its moves is 0 and its return
  from the first move equals its terminal reward

### Requirement: Window truncation uses the last observed position
When the rollout ends before a model's next own move and no game end occurred, the
potential of the final observed position SHALL be used as `Phi(s')` for that model's
last own move, and the value bootstrap SHALL apply as before.

#### Scenario: window ends on the opponent's turn
- **WHEN** white's last move in the window is from material 1, the window ends with
  black to move at material -2, bootstrap for white is 4, gamma 1.0
- **THEN** white's return at its last move is (-2 - 1) + 4 = 1

### Requirement: Shaping is configurable
The system SHALL read `shaping` (`potential` default, or `none`) and `shaping_scale`
(default 0.2, must be >= 0) from the training config and SHALL reject other
`shaping` values with an error listing the accepted ones.

#### Scenario: outcome only
- **WHEN** `shaping: none`
- **THEN** every shaped reward is 0 and returns consist of discounted terminal
  rewards and bootstraps only

#### Scenario: invalid shaping value
- **WHEN** `shaping: material_diff`
- **THEN** training refuses to start and the error lists `potential` and `none`
