## ADDED Requirements

### Requirement: Human-move dataset from Lichess games
The system SHALL build a dataset of (oriented observation, legal mask, human move
action id, outcome from the mover) from Lichess PGN games, keeping only games
where both players' ratings are at least the configured floor, sampling at most
a fixed number of positions per game, and splitting train and held-out by game.

#### Scenario: rating filter and per-game sampling
- **WHEN** a game has one player below the floor
- **THEN** none of its positions are used
- **WHEN** a kept game has 60 plies and the per-game limit is 8
- **THEN** at most 8 of its positions are stored, and they all land in the same split

### Requirement: Supervised pretraining checkpoint
The system SHALL train the policy head with cross-entropy on the human move over
legal moves and the value head with a small-weight squared error on the outcome,
report held-out top-1 accuracy, and save a checkpoint that `init_from` accepts.

#### Scenario: checkpoint continues into RL
- **WHEN** training runs with `init_from` set to the pretraining checkpoint directory
- **THEN** the network starts from the pretrained weights (shared layout)
