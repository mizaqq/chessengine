## MODIFIED Requirements

### Requirement: Puzzle fraction configuration
`puzzle_fraction` SHALL be a number in [0, 1], default 0.0 so existing configs are
unchanged, and out-of-range values SHALL be rejected with an error naming the
accepted range. The number of puzzle boards SHALL be `round(puzzle_fraction *
num_envs)` unless `puzzle_boards` is given, which SHALL take precedence and SHALL
be an integer in [0, num_envs]. When the count is above 0 at least one training
puzzle file SHALL be required. `puzzle_miss_reward` SHALL default to 0.

#### Scenario: out of range
- **WHEN** a config sets `puzzle_fraction: 1.5`
- **THEN** training refuses to start with an error naming the range [0, 1]

#### Scenario: fraction without a file
- **WHEN** a config sets `puzzle_fraction: 0.5` and no puzzle file
- **THEN** training refuses to start with an error naming the missing key

#### Scenario: explicit board count wins
- **WHEN** a config sets `puzzle_fraction: 0.5`, `puzzle_boards: 4` and 12 envs
- **THEN** 4 boards are puzzle boards

#### Scenario: board count out of range
- **WHEN** a config sets `puzzle_boards: 13` with 12 envs
- **THEN** training refuses to start with an error naming the range

## ADDED Requirements

### Requirement: Weighted mix of puzzle sources
Puzzle boards SHALL be able to draw from several puzzle files with non-negative
weights that are normalised to one. Each draw SHALL pick a file with probability
equal to its weight and then a uniform position from that file. The mix SHALL be
deterministic for a given seed.

#### Scenario: equal weights
- **WHEN** two files have weights 0.5 and 0.5 and 1,000 draws are made
- **THEN** between 400 and 600 draws come from each file

#### Scenario: zero weight
- **WHEN** a file has weight 0
- **THEN** no draw comes from it

### Requirement: Self-play position harvest
A harvest SHALL play sampled games between the loaded checkpoint models, collect
every position in which the side to move has at least one mating move, remove
duplicate FENs, shuffle with a seed, and write a training file and a held-out
file in the puzzle file format with disjoint positions. Mating moves SHALL be
verified by the rules engine.

#### Scenario: every harvested row is a mate in one
- **WHEN** any harvested row's FEN is loaded and each listed move played
- **THEN** the resulting position is checkmate

#### Scenario: disjoint splits
- **WHEN** the harvested training and held-out files are loaded
- **THEN** no FEN appears in both
