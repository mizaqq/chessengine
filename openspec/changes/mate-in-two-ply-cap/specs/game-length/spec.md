## ADDED Requirements

### Requirement: Training games end as draws at a ply cap
When `max_plies` is set, a game board SHALL end after `max_plies` plies with the
result `draw` if the game has not ended by the rules before. Puzzle boards SHALL
NOT be subject to the cap. `max_plies` SHALL be null (no cap) by default and
SHALL be rejected when set below 2.

#### Scenario: capped game
- **WHEN** a game board reaches `max_plies` plies without a rules ending
- **THEN** the step reports done with `game_results` `draw` and the board resets

#### Scenario: no cap
- **WHEN** `max_plies` is null
- **THEN** games end only by the rules
