## MODIFIED Requirements

### Requirement: Puzzle episodes end after the mover's decision
A puzzle board SHALL start from the sampled puzzle position and end after the
side to move at the start (the mover) has made `mate_in` moves, or earlier when
the game ends. For `mate_in > 1`, the mover's first move SHALL be one of the
puzzle's key moves; any other first move SHALL end the episode immediately with
the result `puzzle_miss`. A non-terminal end SHALL report `puzzle_miss`; a mate by
the mover SHALL report the normal win result.

#### Scenario: mate in two solved
- **WHEN** the mover plays the key move, the opponent replies, and the mover mates
- **THEN** the episode ends after three plies with the mover's win

#### Scenario: wrong first move
- **WHEN** the mover's first move on a mate-in-two board is not a key move
- **THEN** the episode ends after one ply with `puzzle_miss`

#### Scenario: right first move, mate missed
- **WHEN** the mover plays the key move, the opponent replies, and the mover does not mate
- **THEN** the episode ends after three plies with `puzzle_miss`

### Requirement: Puzzle records carry depth and key moves
A puzzle SHALL carry `mate_in` (1 or 2) and `key_moves` (UCI). Files with the
older `mating_moves` column SHALL load with `mate_in` 1.

#### Scenario: legacy file
- **WHEN** a CSV with columns `puzzle_id,fen,mating_moves,rating` is loaded
- **THEN** every puzzle has `mate_in` 1 and `key_moves` equal to the mating moves

### Requirement: Mate-in-two puzzles are verified at preparation
The preparation script SHALL keep a `mateIn2` puzzle only if its key move leaves
the opponent no reply that avoids mate on the next move (checked with a rules
engine).

#### Scenario: unforced line rejected
- **WHEN** some reply to the key move leaves no mating move
- **THEN** the puzzle is not written
