# advantage-estimation Specification

## Purpose
Defines how per-step advantages and value targets are computed from a self-play
rollout so that either player's policy can be improved from the steps it acted on.

## Requirements

### Requirement: GAE advantage for a model's own steps
For a chosen model, the system SHALL compute an advantage for every rollout step
where that model acted, as the GAE(gamma, lambda) sum of temporal-difference
residuals over that model's subsequent own steps. The residual at an own step is
`reward + gamma * next_value - value`, where `reward` is the potential-based
shaped reward of that move (see the `reward-shaping` spec), where `next_value` is the value the model
assigns to the next state it acts in, or the bootstrap value at the end of the
rollout, and where opponent steps between two own steps contribute no discount and
no residual of their own.

#### Scenario: lambda one equals return minus value
- **WHEN** a model acted on three consecutive steps with rewards 1, 2, 3, its value
  estimates are 0.5 at each step, the bootstrap value is 10, gamma is 0.99 and
  lambda is 1.0
- **THEN** advantages plus values equal the discounted returns 15.62329, 14.771,
  12.9, so the advantages are 15.12329, 14.271, 12.4

#### Scenario: lambda zero is the one-step residual
- **WHEN** the same rollout is used with lambda 0.0
- **THEN** the advantages are exactly the residuals 0.995, 1.995, 12.4

#### Scenario: intermediate lambda blends
- **WHEN** the same rollout is used with lambda 0.95
- **THEN** the last advantage is 12.4, the middle one is 1.995 + 0.9405 * 12.4 =
  13.6572, and the first is 0.995 + 0.9405 * 13.6572 = 13.8396 (within 1e-3)

#### Scenario: opponent steps are skipped without discount
- **WHEN** steps alternate white, black, white with white rewards 1, -2, 3, white
  values 0 at both own steps, bootstrap 0, gamma 1.0, lambda 1.0
- **THEN** white's advantages are 4 at the first step and 3 at the last, and no
  advantage is produced for the black step

### Requirement: Episode boundaries cut the chain
When a step's `done` flag is set, the system SHALL replace the future value for
both players with that player's terminal reward and SHALL reset the GAE
accumulator, regardless of which player acted on the terminating step.

#### Scenario: game ends on the opponent's move
- **WHEN** white acts at step 0 with reward 0 and value 0.2, black acts at step 1
  and that step is `done` with white terminal reward 2.0, gamma 0.99, lambda 0.95,
  bootstrap 5.0
- **THEN** white's advantage at step 0 is 0 + 0.99 * 2.0 - 0.2 = 1.78 and the
  bootstrap value has no effect

#### Scenario: new game after done is not linked to the old one
- **WHEN** an environment finishes a game at step 2 and starts a new one at step 3
- **THEN** advantages for steps 3 onward depend only on rewards and values from
  step 3 onward and the bootstrap

### Requirement: Value targets accompany advantages
The system SHALL return a value target for every own step equal to advantage plus
the value estimate used at that step, so that lambda 1.0 yields the discounted
return.

#### Scenario: targets equal returns at lambda one
- **WHEN** GAE is computed with lambda 1.0 on any rollout
- **THEN** the value targets equal the output of the existing discounted-returns
  computation for the same rollout and bootstrap

### Requirement: Lambda is configurable with a validated range
The system SHALL read `gae_lambda` from the training config, default 1.0 under the
`a2c` preset, and SHALL reject values outside [0, 1] with an error naming the key.

#### Scenario: invalid lambda
- **WHEN** the config sets `gae_lambda: 1.5`
- **THEN** training refuses to start with an error mentioning `gae_lambda`
