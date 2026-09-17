# finishing-curriculum Specification

## Purpose
Teaches the network to convert won positions: finishing boards start k plies before
a human checkmate with the winner to move, under a per-board ply cap, with an
adaptive depth curriculum (reverse curriculum, Florensa et al. 2017) and a
scheduled board layout; results are counted apart from ordinary games.

## Requirements

### Requirement: Finishing records from human checkmates
The preparation step SHALL keep a game only if it ends in checkmate and both
players are rated at least the configured minimum. For each kept game it SHALL
store an anchor position at most 40 plies before the end, the remaining moves
including the mating move, and the winner. Games SHALL be split into train and
held-out files by game, never by position.

#### Scenario: game shorter than the anchor window
- **WHEN** a kept game has 30 plies in total
- **THEN** the anchor is the initial position and all 30 moves are stored

#### Scenario: resignation rejected
- **WHEN** a game's final position is not checkmate
- **THEN** the game is not written

### Requirement: Finishing board start and cap
A finishing board SHALL start from a record rewound `k` plies before the
mating move, where `k` is even, so the eventual winner is to move. The board
SHALL play a full game under a cap of `k + cap_margin` plies from that
position; reaching the cap without a rules ending SHALL end the game as a draw.
The board SHALL report a success when the record's winner wins the game and a
failure otherwise, together with `k`.

#### Scenario: depth zero
- **WHEN** a record is rewound with k = 0 and the winner plays the recorded mating move
- **THEN** the game ends after one ply as the winner's win and a success at depth 0 is reported

#### Scenario: cap reached
- **WHEN** k = 10, cap_margin = 20 and 30 plies pass without a rules ending
- **THEN** the game ends as a draw and a failure at depth 10 is reported

#### Scenario: depth exceeds the record
- **WHEN** a record holds 7 moves and the sampled depth is 10
- **THEN** the depth used is the largest even value not exceeding 6

### Requirement: Depth curriculum
The finishing depth SHALL be drawn uniformly from the even values in
`[0, current_depth]`. `current_depth` SHALL start at `depth_start`, and SHALL
rise by 2, not beyond `depth_max`, whenever at least `window` finishing
episodes have been reported since the last rise and the success rate over the
last `window` episodes is at least `advance_rate`. It SHALL never decrease.

#### Scenario: advance
- **WHEN** depth_start = 2, window = 100, advance_rate = 0.6 and 100 episodes finish with 61 successes
- **THEN** current_depth becomes 4 and depths are drawn from {0, 2, 4}

#### Scenario: no advance below the threshold
- **WHEN** 100 episodes finish with 59 successes
- **THEN** current_depth stays at 2

#### Scenario: ceiling
- **WHEN** current_depth equals depth_max and the threshold is met again
- **THEN** current_depth stays at depth_max

### Requirement: Board layout and configuration
Boards SHALL be laid out as puzzle boards, then finishing boards, then mid-game
boards, then opening boards. `finish_boards` (default 0) SHALL be an integer such
that puzzle, finishing and mid-game boards fit within `num_envs`; a positive
value SHALL require `finish_train_file`. `finish_depth_start` (default 2),
`finish_depth_max` (default 40), `finish_advance_rate` (default 0.6, in (0, 1]),
`finish_window` (default 100, >= 1) and `finish_cap_margin` (default 20, >= 2)
SHALL be validated. The async environment SHALL reject `finish_boards > 0`
with a clear error.

#### Scenario: layout
- **WHEN** num_envs = 12, puzzle_boards = 4, finish_boards = 4, midgame_boards = 2
- **THEN** boards 0-3 are puzzles, 4-7 finishing, 8-9 mid-game, 10-11 opening

#### Scenario: too many boards
- **WHEN** num_envs = 12, puzzle_boards = 4, finish_boards = 6, midgame_boards = 4
- **THEN** configuration fails with an error naming the limit

### Requirement: Finishing metrics
Finishing episodes SHALL be counted separately from game results: they SHALL
NOT enter the win/draw/loss rates or the terminal-return mean. The update
summary SHALL report finishing attempts, overall success rate, success rate per
depth, and the current curriculum depth.

#### Scenario: summary
- **WHEN** an update saw 3 finishing episodes (successes at depths 0 and 2, failure at 2) and 1 capped opening game
- **THEN** the summary reports finish_attempts 3, finish_success_rate 0.667, finish_success_rate_d0 1.0, finish_success_rate_d2 0.5, and the draw rate counts only the opening game

### Requirement: Scheduled board layout
`layout_schedule` SHALL be a list of entries `{from_update, finish_boards,
midgame_boards}` (default empty). From the given update on, the boards after the
puzzle boards SHALL take the new roles when they next reset; a board still
playing keeps its game and its result is accounted under the role it started
with. Entries SHALL be validated like the initial layout.

#### Scenario: hand finishing boards back to the opening
- **WHEN** the layout is 4 puzzle / 6 finishing / 2 mid-game and the schedule says `{from_update: 200, finish_boards: 4, midgame_boards: 2}`
- **THEN** from update 200 boards 8-9 start from the opening once their current finishing game ends, and that last finishing game still counts as a finishing result
