## ADDED Requirements

### Requirement: Held-out finishing conversion metrics
When `finish_eval_file` is configured, evaluation SHALL play, for each depth in
`finish_eval_depths` (default 2, 10, 20), up to `finish_eval_games` (default
100) held-out records rewound by that depth with the same network on both
sides, moves sampled from the policy, under the finishing cap. It SHALL report
`finish_rate_d<k>` as the share of games the record's winner won and
`finish_games_d<k>` as the games played. Records shorter than a depth SHALL be
skipped for that depth.

#### Scenario: perfect finisher
- **WHEN** the network always plays the recorded moves for the winner and the defender replies as recorded
- **THEN** finish_rate_d2 is 1.0

#### Scenario: short record skipped
- **WHEN** a held-out record has 5 moves and depth 10 is evaluated
- **THEN** that record is not counted in finish_games_d10
