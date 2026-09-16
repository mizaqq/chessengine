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

### Requirement: Puzzle boards during training
A vectorized environment SHALL take a number of puzzle boards. Those boards SHALL
start every episode, including after automatic reset, from a training puzzle drawn
by a seeded sampler; the remaining boards SHALL start from the opening. The sampler
SHALL be deterministic for a given seed. (Revised 2026-09-16: mixing by reset was
replaced by assignment by board because a one-ply puzzle episode next to a 200-ply
game left about one percent of plies on puzzles.)

#### Scenario: two of four boards
- **WHEN** 4 boards are created with 2 puzzle boards
- **THEN** boards 0 and 1 start from puzzle positions and boards 2 and 3 from the
  opening (material 0)

#### Scenario: no puzzle boards
- **WHEN** 0 puzzle boards are requested
- **THEN** every board starts from the opening, identical to the behaviour before
  this change

#### Scenario: same seed, same starts
- **WHEN** two samplers are built with the same seed and puzzle set
- **THEN** they return the same sequence of positions

### Requirement: One-move puzzle episodes
A puzzle episode SHALL end after the mover's first move. If that move ends the game
the normal result applies. Otherwise the episode SHALL end with result
`puzzle_miss`; the mover SHALL receive `puzzle_miss_reward` (default 0) as terminal
reward and the other side 0. The board SHALL then reset to a new puzzle.

#### Scenario: mate found
- **WHEN** white plays Ra8# on a puzzle board
- **THEN** the board reports done with result `white_win` and white's terminal
  reward is the win reward

#### Scenario: mate missed
- **WHEN** white plays a non-mating move on a puzzle board
- **THEN** the board reports done with result `puzzle_miss`, white's terminal
  reward is `puzzle_miss_reward`, black's is 0, and the next position is a new
  puzzle

#### Scenario: opening board is unaffected
- **WHEN** a non-puzzle board plays a non-mating move
- **THEN** the game continues as before

### Requirement: Training metrics split by source
Puzzle episodes SHALL be counted as puzzle attempts and puzzle solves, and SHALL
NOT be counted in the win, draw and loss rates of the windowed summary, which cover
opening games only.

#### Scenario: entropy split by board type
- **WHEN** a window has steps on puzzle boards and on opening boards
- **THEN** the summary reports normalised entropy separately for each, and the
  puzzle figure is absent when there are no puzzle boards

#### Scenario: window with puzzle and opening games
- **WHEN** a window contains 30 puzzle attempts of which 3 were mates, and 2
  opening games that were drawn
- **THEN** the summary reports puzzle_attempts 30, puzzle_solved_rate 0.1,
  total_games 2 and draw_rate 1.0

### Requirement: Puzzle fraction configuration
`puzzle_fraction` SHALL be a number in [0, 1], default 0.0 so existing configs are
unchanged, and out-of-range values SHALL be rejected with an error naming the
accepted range. The number of puzzle boards SHALL be `round(puzzle_fraction *
num_envs)`. When the fraction is above 0 a training puzzle file SHALL be required.
`puzzle_miss_reward` SHALL default to 0.

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
