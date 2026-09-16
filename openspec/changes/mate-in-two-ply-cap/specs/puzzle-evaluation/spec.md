## MODIFIED Requirements

### Requirement: Held-out puzzle metrics
For every held-out puzzle the system SHALL query the side to move and score the
most probable legal move against the target set: the rules-engine mating moves
for `mate_in` 1, the puzzle's key moves for `mate_in` 2. It SHALL report top-1
accuracy and the mean probability mass on the target set.

#### Scenario: mate-in-two set
- **WHEN** a held-out file of mate-in-two puzzles is evaluated
- **THEN** a policy that always picks the key move scores top-1 1.0
