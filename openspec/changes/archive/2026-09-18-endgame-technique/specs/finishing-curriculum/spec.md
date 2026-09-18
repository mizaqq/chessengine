## MODIFIED Requirements

### Requirement: Finishing board start and cap
A finishing board SHALL start from a record rewound `k` plies before the
mating move, where `k` is even, so the eventual winner is to move, OR, when the
finishing slot is fed by a technique sampler, from a generated technique start with
its own cap and set label. The board SHALL play a full game under the start's cap;
reaching the cap without a rules ending SHALL end the game as a draw. For a human
record the board SHALL report a success when the record's winner wins the game and a
failure otherwise, together with `k`. For a labelled start the board SHALL report the
outcome together with the label instead of `k`, and the result SHALL be counted under
the technique dials, not the finishing dials.

#### Scenario: depth zero
- **WHEN** a record is rewound with k = 0 and the winner plays the recorded mating move
- **THEN** the game ends after one ply as the winner's win and a success at depth 0 is reported

#### Scenario: cap reached
- **WHEN** k = 10, cap_margin = 20 and 30 plies pass without a rules ending
- **THEN** the game ends as a draw and a failure at depth 10 is reported

#### Scenario: depth exceeds the record
- **WHEN** a record holds 7 moves and the sampled depth is 10
- **THEN** the depth used is the largest even value not exceeding 6

#### Scenario: labelled start counted apart
- **WHEN** the slot is fed by a technique sampler and a `Q` start ends in mate for the queen's side
- **THEN** a technique success for `Q` is reported and the finishing success counters are unchanged
