## Why

Every strength number so far is relative: matches between our own checkpoints and priors.
Owner (2026-09-22): "add an elo evaluation, probably best would be against the stockfish
models to estimate how good our does". Stockfish 18 is installed and exposes a rating-limited
mode (`UCI_LimitStrength`, `UCI_Elo` 1320-3190) anchored to CCRL blitz Elo.

## What Changes

- `scripts/eval_elo.py`: plays a checkpoint against Stockfish at a ladder of `UCI_Elo`
  settings (default 1320, 1400, 1500, 1700, 1900), both colours, sampled or greedy moves,
  300-ply cap as a draw; reports W/D/L per level and a maximum-likelihood Elo under the
  logistic model `p = 1 / (1 + 10^((E_opp - R) / 400))` (draws count half), with a bootstrap
  interval. Output JSON under `experiments/elo/`.
- Dashboard results table gets an `elo` column when `elo.json` exists in an arm's folder.

## Capabilities

### New Capabilities
- `elo-evaluation`: engine-anchored strength estimate for a checkpoint.

### Modified Capabilities
- (none)

## Alternatives considered

- **Lichess bot account** — true human-pool Elo, but slow (days) and needs an API token and
  a running bot; later.
- **Skill Level 0-20 without UCI_Elo** — same mechanism, no calibrated scale; UCI_Elo is the
  documented mapping (Stockfish `search.h`).
- **Fixed-depth or node-limited Stockfish** — arbitrary strength, no Elo meaning.
- **Only win rates between our checkpoints** — what we have; relative only (Balduzzi 2018).

## How we will know it worked

Ladder results for slhi2_256, LONG256X, the old SL prior and LONG3: the fit should order
them as the win matrices do (LONG256X > slhi2_256 > LONG3 > SL) and the interval should be
tight enough (< +-100) at 40 games per level to tell them apart. Owner's prediction: (not given).

## What to understand

- Why the estimate is a fit, not a reading (floor at 1320; logistic model; draws as half).
- What "CCRL blitz Elo" means and why it is not a Lichess or FIDE rating.
- How the interval width scales with games.
