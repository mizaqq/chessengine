## Purpose
Estimate a checkpoint's playing strength on an engine-anchored Elo scale by playing a ladder
of rating-limited Stockfish settings and fitting a logistic model.

## ADDED Requirements

### Requirement: Ladder matches against rating-limited Stockfish
The evaluation SHALL play the checkpoint against Stockfish with `UCI_LimitStrength` true and
`UCI_Elo` set to each configured level, the same number of games as white and as black per
level, with the network moving by sampling (default) or argmax, and a ply cap that ends the
game as a draw. It SHALL report wins, draws and losses per level from the network's side.

#### Scenario: both colours
- **WHEN** games per colour is 20 and the ladder has 5 levels
- **THEN** 200 games are played and each level reports 40 results

### Requirement: Logistic Elo estimate
From the per-level results the evaluation SHALL report the rating `R` maximising the
likelihood of the observed scores under `p(win) = 1 / (1 + 10^((E - R) / 400))` with draws
counted as half a win, together with a bootstrap interval over games. When every level is
lost or every level is won, the estimate SHALL be reported as a bound, not a number.

#### Scenario: fit by hand
- **WHEN** the network scores 0.5 against 1500 and 0.24 against 1700 (40 games each)
- **THEN** the estimate is about 1500 (within the interval)

#### Scenario: below the floor
- **WHEN** the network scores 0.1 against every level
- **THEN** the estimate is reported as below 1320 with the fitted value shown for reference
