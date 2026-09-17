## Why

Mate-in-one accuracy sat at 0.57 for 900 PPO updates and moved to 0.64 only after
shaping was removed, while the same network reaches 0.90 when told the mating
move. The measured cause is exploration collapse: on 35% of puzzles the policy
puts under 0.05 on the mate, so the mate is almost never played and the outcome
never reaches the update. The owner chose behaviour noise over a larger entropy
bonus ("we can add the noise in training to explore more moves").

## What Changes

- On curriculum boards (puzzle and finishing) the training move is drawn from a
  behaviour policy `b = (1 - eps) * pi + eps * Dir(alpha)` over legal moves;
  the network's own probabilities are unchanged and evaluation stays noise-free.
- The rollout stores `log b(a)` as the old log-probability, so the PPO ratio
  compares the new policy with the distribution the move was actually drawn
  from (importance-corrected, then clipped as before).
- New diagnostic: share of noisy-board moves whose network probability was
  below 0.05 (the "tried against the prior" rate).
- New config keys `explore_epsilon` (default 0 = off), `explore_alpha` (0.3),
  `explore_boards` (`curriculum` | `all`).
- Probe script gains a mate-mass histogram so the confident-wrong share is a
  tracked number.

## Capabilities

### New Capabilities
- none

### Modified Capabilities
- `policy-update`: rollout stores the behaviour log-probability; new requirement
  for exploration noise on curriculum boards.

## Alternatives considered

- **Larger entropy bonus.** Changes the network itself and flattens the 52% of
  positions it gets right; 0.01 did not prevent the collapse. Schulman et al.
  2017 use the entropy term as a regulariser, not as targeted exploration.
- **Root Dirichlet noise as in AlphaZero** (Silver et al. 2017, p.14: "Dirichlet
  noise Dir(alpha) was added to the prior probabilities in the root node; this
  was scaled in inverse proportion to the approximate number of legal moves ...
  alpha = 0.3 for chess"; AlphaGo Zero: weight eps = 0.25; evaluation greedy,
  p.15). Chosen: same device without the search, applied to the sampling
  distribution; behaviour log-prob in the ratio keeps the PPO objective honest.
- **Label-guided behaviour** (play the known mating move with probability eps;
  Salimans & Chen 2018 demonstration-guided starts). Faster but needs labels, so
  nothing for finishing boards beyond depth 0. Kept as the comparison arm if
  noise is too slow.
- **Search as teacher** (AlphaZero). Full cure; the agreed later step.

## How we will know it worked

Two arms of 120 updates from the FIN3 checkpoint with FIN3's layout (4 puzzle /
6 finishing / 2 mid-game, threshold 0.4): NOISE (eps 0.25, alpha 0.3) and
CONTROL (no noise). Dials: mate-mass histogram on 1,000 Lichess puzzles (share
below 0.05, today 35%); Lichess m1 / m2 (0.638 / 0.545); own-game mate set
(0.12); finishing conversion; the tried-against-the-prior rate.

Owner's prediction: not given (owner approved the design without predicting).
Claude's expectation: below-0.05 share 35% -> 25-30%, m1 0.64 -> 0.68-0.72 in
NOISE, CONTROL m1 within 0.02 of 0.64.

## What to understand

- Behaviour policy vs target policy: why the ratio must use the distribution the
  move came from, and what the clip does to a rare well-rewarded move.
- Why noise in the data differs from entropy in the loss (owner articulated
  this on 2026-09-17).
- Why the histogram, not the mean, is the dial for this failure.
