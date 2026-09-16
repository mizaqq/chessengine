## ADDED Requirements

### Requirement: Mid-game start positions for game boards
When `midgame_boards` > 0 and `game_start_file` is set, the boards after the
puzzle boards, `[num_puzzle_envs, num_puzzle_envs + midgame_boards)`, SHALL start
every episode from a FEN sampled from the file and play a full game under the
normal rules and the ply cap. They SHALL NOT be flagged as puzzle boards.

#### Scenario: mid-game board plays a full game
- **WHEN** a mid-game board reaches a rules ending or the ply cap
- **THEN** the result is reported as a game result with `puzzle_boards` false and the board resets to a new sampled FEN
