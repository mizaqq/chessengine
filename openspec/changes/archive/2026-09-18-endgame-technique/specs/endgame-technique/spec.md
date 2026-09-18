## Purpose
Teaches finishing technique with forced wins: generated bare-king positions with a
material list and a ply cap, plus a Lichess puzzle source restricted to endgames that
end in mate, so the conversion signal does not depend on a human defender's blunder.

## ADDED Requirements

### Requirement: Endgame-mate puzzle source
The puzzle preparation SHALL accept a theme filter and SHALL keep a puzzle only if it
carries every required theme and a `mateIn<N>` theme with N from 1 to 5. The stored
mate depth SHALL be N. Depth 1 SHALL be verified as a mate in one, depth 2 by the
forcing-move check, and depth 3 and above by replaying the solution to a checkmate,
as for the existing sets. The output SHALL be written under a name derived from the
filter and SHALL carry the solution line.

#### Scenario: endgame mate-in-two kept
- **WHEN** a puzzle has themes `endgame mateIn2 short` and its forcing move checks out
- **THEN** it is written with `mate_in` 2 and the two-move solution

#### Scenario: middlegame mate rejected by the filter
- **WHEN** the filter requires `endgame` and a puzzle has themes `middlegame mateIn1`
- **THEN** it is not written

### Requirement: Generated technique starts
A technique sampler SHALL produce start positions for a configured list of material
sets, each a string of the strong side's pieces beyond the king (for example `Q`,
`R`, `RR`, `QR`), with the strong side to move. Positions SHALL be legal, SHALL NOT
have the side to move in a position with a mate in one available, SHALL NOT be a
check on the strong king, and SHALL be drawn uniformly over the configured sets. The
strong side's colour SHALL be drawn at random. Each start SHALL carry the set's cap
in plies and the set label.

#### Scenario: queen start
- **WHEN** the sampler draws set `Q` with cap 40
- **THEN** the position has exactly a king and queen against a lone king, the queen's
  side is to move, no legal move mates at once, and the start carries cap 40 and label `Q`

#### Scenario: reproducible
- **WHEN** two samplers are built with the same seed and configuration
- **THEN** they produce the same sequence of starts

### Requirement: Technique board outcome
A technique board SHALL play from the start under its cap; reaching the cap without a
rules ending SHALL end the game as a draw. The board SHALL report success when the
strong side wins and failure otherwise, together with the set label. Successes and
failures SHALL be counted per label as `technique_attempts_<label>` and
`technique_success_rate_<label>` and SHALL NOT enter the self-play or finishing rates.

#### Scenario: mate inside the cap
- **WHEN** a `R` board with cap 60 ends in checkmate by the rook's side after 30 plies
- **THEN** a success for label `R` is reported and no finishing result is reported

#### Scenario: cap reached
- **WHEN** a `Q` board with cap 40 reaches 40 plies without a rules ending
- **THEN** the game ends as a draw and a failure for label `Q` is reported

### Requirement: Technique evaluation
When technique sets are configured, the periodic evaluation SHALL play a fixed
generated held-out set per label (same positions every evaluation, from a fixed seed),
both sides sampling from the learner, and SHALL report `technique_rate_<label>` and
`technique_games_<label>`.

#### Scenario: fixed positions
- **WHEN** the evaluation runs at update 50 and at update 100
- **THEN** the start positions for each label are identical
