## ADDED Requirements

### Requirement: Punishing one-ply gifts
When `punish_epsilon` > 0, on the boards selected by `punish_boards` (`puzzle`,
`curriculum`, `all`; default `all`) a learner ply whose position has a punishing
move and no label demonstration SHALL be drawn from
`(1 - punish_epsilon) * pi + punish_epsilon * uniform(punishing moves)`, sampled by
component, with the mixture's log-probability stored. The punishing moves are the rules
mates in one when any exist; otherwise the legal captures whose captured value minus the value of the
capturing piece when the destination square can be recaptured (pawn 1, knight 3,
bishop 3, rook 5, queen 9) is at least `punish_threshold` (default 3). Boards with a
label demonstration (puzzle key move, human line) SHALL keep `guide_epsilon`.
Pool-opponent plies SHALL be unaffected. `punish_epsilon` SHALL be in [0, 1);
`punish_threshold` SHALL be a positive integer. A ply played by the punisher SHALL
mark the episode as teacher-touched for the technique and finishing curricula. The
update summary SHALL report the number of learner plies with a punishing move on a
punish board and how many were played by the punisher.

#### Scenario: hung queen taken with probability punish_epsilon
- **WHEN** `punish_epsilon` = 0.5, the opponent has just left a queen attackable by a knight with no recapture, the network gives the capture 0.02 and there is no other punishing move
- **THEN** the behaviour probability of the capture is 0.51 and the stored log-probability is log(0.51)

#### Scenario: recapture cancels the gift
- **WHEN** a bishop can take a knight but a rook recaptures on that square
- **THEN** the net gain is 0 and the capture is not a punishing move

#### Scenario: label demonstration wins
- **WHEN** a finishing board still follows the human line and the human's next move is not a capture
- **THEN** the demonstration is the human's move under `guide_epsilon`, not a punishing move under `punish_epsilon`

#### Scenario: mate before material
- **WHEN** the side to move has both a mate in one and a free rook to take
- **THEN** the punishing moves are the mating moves only

#### Scenario: off by default
- **WHEN** `punish_epsilon` is absent or 0
- **THEN** behaviour is that of label-guided exploration alone and the summary reports no punish counts
