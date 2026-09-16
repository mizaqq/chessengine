## Purpose

A single network plays both colours by seeing every position from the mover's
point of view, so patterns learned as white apply as black and vice versa.

## ADDED Requirements

### Requirement: Observation oriented to the mover
Before an observation reaches the network it SHALL be expressed from the side to
move's perspective. For white to move the observation SHALL be unchanged. For
black to move: ranks SHALL be reversed, each white piece plane SHALL be swapped
with the black plane of the same piece type, the white castling planes SHALL be
swapped with the black castling planes, and the empty-square, repetition,
irreversible-counter and side-to-move planes SHALL keep their meaning (rank
reversal applied to the per-square planes). The transform SHALL be its own
inverse when applied twice to a black-to-move observation.

#### Scenario: mirrored start positions look identical
- **WHEN** the initial position with white to move and the initial position with
  black to move are both oriented
- **THEN** the piece, empty-square and castling planes are identical, and only
  the side-to-move plane differs

#### Scenario: black pawn on a7 becomes "my pawn on rank 2"
- **WHEN** black is to move with a black pawn on a7 and no other pawns
- **THEN** the oriented observation has a one in the mover's pawn plane at the
  square a2 and zeros in the opponent's pawn plane

#### Scenario: castling rights follow the swap
- **WHEN** black is to move and only black may castle kingside
- **THEN** the oriented observation shows the mover's right-side castling plane
  set and the other three castling planes clear

### Requirement: Move indices need no mirroring
The move index sampled from the network SHALL be passed to the environment
unchanged for either colour, because the environment's move encoding is already
relative to the mover.

#### Scenario: legal-move masks of mirrored positions coincide
- **WHEN** a position and its colour-mirrored counterpart (board reflected across
  the middle rank, colours swapped, castling rights swapped) are compared
- **THEN** their legal-move masks are identical

### Requirement: One network for both colours
Training and evaluation SHALL use a single set of parameters for white and black.
Each update SHALL apply both sides' samples in one backward pass. Per-side
returns, bootstrap values and terminal rewards SHALL keep their existing
definitions from the mover's perspective.

#### Scenario: one parameter set
- **WHEN** training runs for one update
- **THEN** exactly one optimizer step is taken and the parameters used for
  white's and black's moves are the same object

#### Scenario: bootstrap sign unchanged
- **WHEN** a rollout ends with black to move and white's return is bootstrapped
- **THEN** the bootstrap is the negated value the shared network assigns to the
  oriented black-to-move position

### Requirement: Checkpoint compatibility
A run SHALL save one model file. Loading utilities SHALL accept a single-file
checkpoint and SHALL still load the legacy white/black pair (as two models) so
earlier experiments remain evaluable.

#### Scenario: legacy pair still loads
- **WHEN** a directory contains only `white_model_*.pth` and `black_model_*.pth`
- **THEN** evaluation runs with the white model for white-to-move positions and
  the black model for black-to-move positions, unoriented, as before

#### Scenario: single file loads for both sides
- **WHEN** a directory contains `model_*.pth`
- **THEN** evaluation uses it for both colours with oriented observations
