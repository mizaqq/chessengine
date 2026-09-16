## Purpose

Lets training games begin from supplied positions instead of the opening, so that
episodes can start close to a terminal reward (reverse curriculum by start state).

## ADDED Requirements

### Requirement: Reset from a given position
An environment SHALL accept an optional FEN on reset and start the game from that
position, with the side to move, legal moves, observation and material all taken
from that position. Without a FEN the game SHALL start from the standard opening.

#### Scenario: rook mate position
- **WHEN** an environment is reset with FEN `6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1`
- **THEN** the side to move is white, the reported material is 5, and among the
  legal moves exactly one (Ra8) ends the game as a white win

#### Scenario: no FEN
- **WHEN** an environment is reset without a FEN
- **THEN** the position is the standard opening with material 0 and white to move

### Requirement: Mixed start positions during training
A vectorized environment SHALL take a start-position sampler. On every reset,
including automatic reset after a game ends, each env SHALL start from the sampler's
position when it returns one and from the opening otherwise. The sampler SHALL
return a training puzzle position with probability `puzzle_fraction` and nothing
otherwise, and SHALL be deterministic for a given seed.

#### Scenario: fraction 1
- **WHEN** `puzzle_fraction` is 1.0 and 4 envs are reset
- **THEN** all 4 envs start from puzzle positions (none has material 0 with the
  opening layout)

#### Scenario: fraction 0
- **WHEN** `puzzle_fraction` is 0.0
- **THEN** every reset starts from the opening, identical to today's behaviour

#### Scenario: auto-reset uses the sampler
- **WHEN** `puzzle_fraction` is 1.0 and a game in env 0 ends
- **THEN** the position returned for env 0 is a puzzle position, not the opening

#### Scenario: same seed, same starts
- **WHEN** two samplers are built with the same seed and puzzle set
- **THEN** they return the same sequence of positions

### Requirement: Puzzle fraction configuration
`puzzle_fraction` SHALL be a number in [0, 1], default 0.0 so existing configs are
unchanged, and out-of-range values SHALL be rejected with an error naming the
accepted range. When the fraction is above 0 a training puzzle file SHALL be
required.

#### Scenario: out of range
- **WHEN** a config sets `puzzle_fraction: 1.5`
- **THEN** training refuses to start with an error naming the range [0, 1]

#### Scenario: fraction without a file
- **WHEN** a config sets `puzzle_fraction: 0.5` and no `puzzle_train_file`
- **THEN** training refuses to start with an error naming the missing key

### Requirement: Puzzle position files
A puzzle file SHALL be a CSV with one position per row: puzzle id, FEN of the
position where the side to move has a mate in one, the mating moves in UCI
separated by spaces, and the Lichess rating. The training and held-out files SHALL
share no puzzle id. Every stored FEN SHALL have at least one legal mating move.

#### Scenario: disjoint sets
- **WHEN** the training and held-out files are loaded
- **THEN** the intersection of their puzzle ids is empty

#### Scenario: every row is a mate in one
- **WHEN** any row's FEN is loaded and each listed move is played
- **THEN** the resulting position is checkmate
