## MODIFIED Requirements

### Requirement: Generated technique starts
A technique sampler SHALL produce start positions for a configured list of material
sets, each a string of the strong side's pieces beyond the king (for example `Q`,
`R`, `RR`, `QR`), with the strong side to move. Positions SHALL be legal, SHALL NOT
have the side to move in a position with a mate in one available, SHALL NOT be a
check on the strong king, and SHALL be drawn uniformly over the configured sets. The
strong side's colour SHALL be drawn at random. Each start SHALL carry the set's cap
in plies and the set label. Each start SHALL also carry a difficulty level: 0 = weak
king on a corner square with the strong king at distance 2 and every strong piece
within distance 4 of the weak king (within 2, nearly every queen position is a mate in
one or an illegal check and is rejected); 1 = weak king on an edge square with the strong
king within distance 3; 2 = weak king one square from an edge; 3 = unconstrained.
When the curriculum is on, a set's level SHALL be drawn uniformly from 0 to the set's
current level; when off, every start SHALL be level 3.

#### Scenario: queen start
- **WHEN** the sampler draws set `Q` with cap 40
- **THEN** the position has exactly a king and queen against a lone king, the queen's
  side is to move, no legal move mates at once, and the start carries cap 40 and label `Q`

#### Scenario: reproducible
- **WHEN** two samplers are built with the same seed and configuration
- **THEN** they produce the same sequence of starts

#### Scenario: level 0 start
- **WHEN** a level-0 `Q` start is generated
- **THEN** the weak king stands on a1, a8, h1 or h8, the strong king is exactly two
  squares away, and the queen is within four squares of the weak king

## ADDED Requirements

### Requirement: Technique level curriculum
Each material set SHALL keep a current level starting at `technique_level_start`.
When the success rate over the last `technique_window` finished boards of that set is
at least `technique_advance_rate`, the level SHALL rise by one up to 3 and the window
SHALL reset. Levels SHALL never fall. The summary SHALL report `technique_level_<set>`.

#### Scenario: advance
- **WHEN** set `Q` is at level 0, the window is 30 and 20 of the last 30 boards were won
- **THEN** `Q` moves to level 1 and the next window starts empty

#### Scenario: held-out unaffected
- **WHEN** the periodic technique evaluation runs while `Q` is at level 1
- **THEN** its positions are level-3 positions, as before
