## Context
Stockfish 18 at `/opt/homebrew/bin/stockfish`; python-chess 1.11 drives it over UCI. Our
network needs OpenSpiel observations, so each game keeps an OpenSpielEnv (observation, legal
mask, result) and a python-chess board (engine input) in step; engine moves are mapped to
OpenSpiel actions through `actions_for_uci`.

## Decisions
- **Ladder 1320-1900 by default**: the current networks are expected near or below the
  floor; levels above 1900 would all be losses and add nothing to the fit.
- **Move time 50 ms**: at skill level L Stockfish picks at depth 1+L, reached in milliseconds
  at low levels; the CCRL calibration assumed blitz, so the number is an estimate either way.
- **Draws as half a win** in the likelihood (standard for Elo from scores).
- **Grid-search MLE** over 400-3400 in 1-Elo steps plus 200 bootstrap resamples: trivial cost,
  no dependency.
- **Sampled moves by default** to match every other evaluation; `--greedy` for the
  deterministic policy.

## Risks / Trade-offs
- Our 300-ply cap and Stockfish's adjudication-free play inflate draws; reported separately.
- CCRL blitz Elo is not Lichess Elo (roughly, Lichess ratings run higher).
