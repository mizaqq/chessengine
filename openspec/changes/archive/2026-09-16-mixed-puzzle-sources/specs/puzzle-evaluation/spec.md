## ADDED Requirements

### Requirement: Several named held-out sets
Evaluation SHALL accept several named held-out files and SHALL report the
metrics of each under keys prefixed with the set name (`<name>_top1`,
`<name>_mate_prob`, `<name>_count`). A single unnamed file SHALL keep the
existing unprefixed keys.

#### Scenario: two sets
- **WHEN** sets `lichess` and `selfplay` are configured
- **THEN** each log entry with evaluation carries `lichess_top1` and
  `selfplay_top1`

#### Scenario: single file unchanged
- **WHEN** only `puzzle_eval_file` is configured
- **THEN** the keys are `puzzle_top1`, `puzzle_mate_prob`, `puzzle_count` as before
