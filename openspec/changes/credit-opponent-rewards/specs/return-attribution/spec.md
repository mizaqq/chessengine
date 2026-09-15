## Purpose

Defines which per-step rewards are credited to which own step when a model's
discounted returns are computed from a shared self-play rollout.

## ADDED Requirements

### Requirement: Opponent-step rewards are credited to the preceding own step
For a chosen model, the reward credited to one of its own steps SHALL be the reward
of that step plus the rewards of all following steps up to but excluding the
model's next own step, each converted to the model's perspective. Discounting by
`gamma` SHALL be applied once per own step, as before.

#### Scenario: capture and recapture
- **WHEN** steps are white, black, white with white-perspective rewards 1, -2, 3,
  bootstrap 0 and gamma 1.0
- **THEN** white's return at the first step is (1 + -2) + 3 = 2 and at the last step
  is 3; black's return at its step is 2 + (-3) = -1

#### Scenario: several opponent steps in a row
- **WHEN** steps are white, black, black, white with white-perspective rewards
  0, -1, -3, 0, bootstrap 0 and gamma 1.0 (as after a desynchronised environment)
- **THEN** white's return at the first step is 0 + (-1) + (-3) + 0 = -4

#### Scenario: opponent reward at the end of the rollout is still credited
- **WHEN** the last step of the rollout is an opponent step with white-perspective
  reward -5, preceded by a white step with reward 0, bootstrap 4 and gamma 0.5
- **THEN** white's return at its step is (0 + -5) + 0.5 * 4 = -3

### Requirement: Game end bounds attribution
Rewards from steps after a `done` step SHALL NOT be credited to any step before it.
The reward of the `done` step itself SHALL be credited to the preceding own step of
the non-acting model, and the terminal reward SHALL replace the future value as
before.

#### Scenario: opponent captures and mates
- **WHEN** white acts with reward 0, then black acts with white-perspective reward
  -3 and `done`, white terminal reward -2, followed by a new-game white step with
  reward 0, bootstrap 5, gamma 1.0
- **THEN** white's return at the first step is (0 + -3) + (-2) = -5 and the new-game
  step's return is 5

#### Scenario: own mating move is not polluted by the new game
- **WHEN** white mates with reward 5 and `done` with terminal reward 2, and the new
  game continues with a black step of white-perspective reward -9
- **THEN** white's return at the mating step is 5 + 2 = 7 and the -9 is not credited
  to it

### Requirement: Existing behaviour is otherwise unchanged
Returns for rollouts in which a model acts on every step SHALL be identical to the
previous implementation.

#### Scenario: single player, no done
- **WHEN** all steps belong to the same model with rewards 1, 2, 3, bootstrap 10 and
  gamma 0.99
- **THEN** returns are 15.62329, 14.771, 12.9 as before
