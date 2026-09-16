# puzzle-evaluation Specification

## Purpose
Measures whether a policy finds mates in one on a fixed held-out set of positions,
independent of which positions self-play happens to reach.

## Requirements

### Requirement: Held-out mate-in-one metrics
Given the two models and a puzzle file, the evaluation SHALL, for every position,
query the model of the side to move with the legal-move mask and report: top-1
accuracy (share of positions where the most probable move is a mating move) and
mean mating-move probability (mean over positions of the total probability on
mating moves). Mating moves SHALL be determined from the position itself, not from
the file. Evaluation SHALL not change model parameters or RNG state used by training.

#### Scenario: perfect policy
- **WHEN** a policy puts probability 1 on a mating move in every position
- **THEN** top-1 accuracy is 1.0 and mean mating-move probability is 1.0

#### Scenario: uniform policy on a position with 20 legal moves and 1 mate
- **WHEN** the policy is uniform over legal moves in a single such position
- **THEN** mean mating-move probability is 0.05 and top-1 accuracy is 0 or 1
  depending only on tie-breaking of the argmax

### Requirement: Periodic evaluation during training
When a held-out puzzle file is configured, training SHALL run the evaluation every
`eval_interval` updates and at the end, and SHALL include the two metrics in the
logged summary for that update. `eval_interval` SHALL default to 50 and SHALL be a
positive integer.

#### Scenario: interval 50 over 120 updates
- **WHEN** training runs 120 updates with `eval_interval: 50`
- **THEN** puzzle metrics appear in the logs at updates 50, 100 and 120

#### Scenario: no held-out file
- **WHEN** `puzzle_eval_file` is not set
- **THEN** no evaluation runs and logs are as before
