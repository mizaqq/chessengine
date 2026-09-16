# RL glossary

Terms the owner has used correctly in this project. One or two sentences each.
Add a term only once it has been understood, not when first met.

**Return**: Discounted sum of rewards that followed a move, with the value head's guess standing in for whatever lies past the rollout window.

**Value (V)**: The model's estimate, at a position, of the return it expects from there.

**Advantage**: Return minus value. How much better a move turned out than expected; the only quantity the policy gradient reacts to.

**Bootstrap**: The value estimate used in place of the unknown future when a rollout is cut off before the game ends.

**Rollout**: A fixed number of steps of self-play collected before one learning update. Here 15 steps on 12 boards.

**Policy gradient**: Push up the probability of moves with positive advantage, down for negative, in proportion to the advantage.

**Reward shaping**: Extra hand-made reward added to speed up learning. Risky: it can change which policy is optimal.

**Potential-based shaping**: Shaping of the form gamma × Φ(next) − Φ(now) for some state potential Φ. The only form guaranteed not to change the optimal policy. Works like a loan repaid at the terminal state.

**Potential (Φ)**: A number assigned to a state; here material balance from the mover's view. Zero at the terminal state by construction.

**Terminal reward**: The outcome payment at game end: win, loss or draw. Under potential shaping it is the only reward that does not net to zero.

**Zero-sum two-player framing**: From one player's seat the opponent is part of the environment; a transition runs from my move to my next move and includes their reply.

**Discount (gamma)**: Weight on future reward per own decision. Adds bias regardless of value accuracy.

**Lambda (GAE)**: Weight on longer-horizon surprises when estimating the advantage. Zero means one-step TD error, one means full return minus value. Adds bias only when the value head is wrong.

**TD residual / surprise**: r + gamma × V(next) − V(now). The one-step advantage estimate.

**Probability ratio (PPO)**: New policy probability of the taken move divided by the old one. Equals one at the first step after collection.

**Clipping (PPO)**: Ignoring the objective's gradient once the ratio leaves [1 − eps, 1 + eps] in the helpful direction, so several passes over one rollout are safe.

**A2C**: The one-pass, no-clip special case of PPO.

**Greedy play**: Always the most likely move. Deterministic; loops into repetition with a weak policy. Not a faithful summary of a policy trained by sampling.

**Sampled play**: Moves drawn from the policy's probabilities. What the policy was trained for; what evaluation should use.

**Mate-in-one rate**: Share of positions with a mate available where the mover played one. Project's first evaluation metric; baseline 3.7%.

**Entropy (normalised)**: Spread of the move distribution divided by log of the number of legal moves, so 1 is uniform and 0 is a single move. Kept in the loss to prevent premature certainty.

**Reverse curriculum**: Start episodes near the goal state and move the starts outward as the agent succeeds (Florensa et al. 2017). Needs failed attempts to end quickly.

**Start state**: The position an episode begins from. Changing it changes exploration cost without changing what is optimal.

**One-move puzzle episode**: A training episode from a mate-in-one position that ends after the mover's move: mate gives the win, anything else ends with `puzzle_miss` and reward 0.

**Attempts per update**: How many tries at the sparse reward one gradient update sees. The real currency of a start-state curriculum.

**Held-out set**: Positions never used in training, kept to measure whether a skill transfers. Here 2,000 Lichess mate-in-one puzzles.

**Train/test gap**: Training solve rate minus held-out accuracy. Near zero means learned skill; large means memorised positions.

**Distribution shift**: Training positions (tactical puzzles) differ from deployment positions (self-play games), so puzzle accuracy moves before the in-game rate.

**Metrics by source**: Counting puzzle episodes and opening games separately, because short episodes flood any shared counter.
