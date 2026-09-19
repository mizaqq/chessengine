## Why

The blunder audit (`scripts/blunder_audit.py`, 2026-09-19, LONG3 as mover) shows that
our self-play defender takes a hung piece only about half the time (0.54) and delivers
an allowed mate in one about one time in eight (0.13); the SL prior is no better (0.49 /
0.09). So in training a hung piece costs half a piece and an allowed mate costs almost
nothing: the gradient that should teach "do not do this" is diluted by exactly those
factors. Owner (2026-09-19): "the model we are playing against is not punishing for very
stupid moves so model is not learning to avoid them at all" → "go ahead with it".

## What Changes

- **A third demonstrator rule: take a free piece.** Besides the puzzle key move, the
  human line and the rules mate, the demonstrator now also offers every legal capture
  that wins at least `punish_threshold` (default 3, a minor piece) net of the cheapest
  immediate recapture (one-ply rules check, no search).
- **Punishing guidance on every board.** A new probability `punish_epsilon` (default 0,
  off; run at 0.5) applies on the boards selected by `punish_boards` (default `all`): when
  the side to move has a rules mate or a free capture and the board has no other
  demonstration, the behaviour policy plays one of them with probability
  `punish_epsilon`, sampled by component, log-probability stored as the mixture's, like
  label-guided exploration. Boards with a label demonstration (puzzle key move, human
  line) keep `guide_epsilon`. Learner plies only; pool-opponent plies are untouched.
- **Diagnostics.** `punish_count` (learner plies with a punishing move available on a
  punish board) and `punish_picks` (how many were played by the demonstrator). A move
  played by the punisher marks the episode as teacher-touched for the technique and
  finishing curricula, exactly as the label demonstrator does.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `policy-update`: extends the pending "Label-guided exploration" requirement (change
  `exploration-noise`, not yet archived) with the free-capture demonstration and the
  separate punishing probability on all boards.

## Alternatives considered

- **Referee wrapper on the pool opponents only** (prior and snapshots always punish).
  Certain punishment, learner plies untouched, but only on the five pool boards; the
  finishing and technique boards, where the defender is also us, stay unpunishing, and
  the learner is not taught to take pieces itself. Bansal et al. 2018 (opponent pool)
  motivates the pool but does not change what the pool members can see.
- **Stronger pool members.** The audit shows the SL prior punishes no better than we do
  and every snapshot is us; Balduzzi et al. 2019: self-play only climbs when the game is
  approximately transitive, and beating the current self is not getting better.
- **Material penalty in the reward** (potential-based shaping, Ng et al. 1999; archived
  change `potential-based-shaping`): teaches the price of a lost piece without teaching
  the refutation, and shaped the value head in earlier runs. Rejected for this problem.
- **Colder opponent (argmax defender).** Audit row "ourselves, greedy": mate punishment
  0.13 → 0.31, material unchanged (0.49). Too small.
- **Exploiter agents trained to beat the main agent** (AlphaStar league, Vinyals et al.
  2019). The right shape, but a learned exploiter is a second training run; the one-ply
  rules punisher is the hand-made exploiter for the blunder class we measured.

## How we will know it worked

Arm PUNISH: 120 updates from the best checkpoint (DEPTH20 if it holds LONG3's numbers,
else LONG3), `punish_epsilon 0.5`, `punish_boards all`, everything else as LONG3.
- Blunder audit of the result against itself: free-piece gifts per 100 plies (LONG3 9.6),
  allowed mates per 100 plies (0.9), punish rates (0.54 / 0.13; expected ≥ 0.75 with the
  punisher firing, but the audit runs the bare network, so the interesting number is the
  gift rate).
- Own-game mate-in-one (LONG3 0.387), puzzles (must not fall), prior score, win matrix
  vs LONG3.
- Entropy (LONG3 0.27-0.29): a sharper opponent usually sharpens the policy further.
- Owner's prediction: (not given; owner skipped the prediction.)

## What to understand

- Why self-play stalls when the opponent does not punish (Balduzzi: transitivity;
  AlphaStar: exploiters), and how the audit measures it.
- Why the punishing move is sampled by component with the mixture's log-probability
  (PPO stays on-policy in expectation; teacher-touched episodes do not count for
  curricula).
- What `punish_epsilon` at 0, 0.5 and 1 does to the punish rate and to the gradient of
  the punishing ply (at 1 the ply is fully off-policy and clipped away).
