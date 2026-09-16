# RL Resources

Curated, high-trust sources for this project. Every RL design decision should cite
an entry here. Add a one-line annotation to each link: what it covers and when to
reach for it. Prune anything that turns out shallow or wrong.

## Knowledge

<!-- Papers, official docs, reference implementations. Annotate every entry. -->

- [Paper: "Proximal Policy Optimization Algorithms" — Schulman, Wolski, Dhariwal, Radford, Klimov (2017)](https://arxiv.org/abs/1707.06347)
  Defines the clipped surrogate objective (eq. 7), the combined loss with value and
  entropy terms (eq. 9, c1=0.5, c2=0.01), Algorithm 1, and the truncated GAE used in
  practice (eq. 11-12). Hyperparameters: Table 3 (Atari: T=128, 3 epochs, minibatch
  32, eps=0.1, gamma=0.99, lambda=0.95) and Table 5 (MuJoCo: T=2048, 10 epochs,
  minibatch 64, eps=0.2). Use for: why clipping, what epsilon/epochs/minibatches do.
- [Paper: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — Schulman, Moritz, Levine, Jordan, Abbeel (2016)](https://arxiv.org/abs/1506.02438)
  Section 3: TD residual delta_t = r_t + gamma V(s_{t+1}) - V(s_t) (eq. 10), GAE as
  sum of (gamma*lambda)^l delta_{t+l} (eq. 16), lambda=0 is one-step TD (eq. 17),
  lambda=1 is Monte Carlo returns minus baseline (eq. 18). Key sentence: gamma
  introduces bias regardless of value accuracy; lambda introduces bias only when V is
  inaccurate, so best lambda is usually lower than best gamma. Use for: choosing and
  explaining lambda, bias/variance intuition.
- [Blog: "The 37 Implementation Details of Proximal Policy Optimization" — Huang, Dossa, Raffin, Kanervisto, Wang (ICLR Blog Track 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  The 13 core details: vectorized rollouts, GAE with bootstrapping and
  returns = advantages + values, shuffled minibatches, per-minibatch advantage
  normalization, clipped value loss (optional), global grad-norm clip 0.5, Adam
  eps 1e-5, linear LR annealing, debug variables (clip fraction, approx KL). Also
  invalid-action masking via -inf logits. Use for: implementation checklist and
  what to log.

- [Book: *Reinforcement Learning: An Introduction* — Sutton & Barto, Section 1.5 tic-tac-toe example](http://incompleteideas.net/book/the-book-2nd.html)
  The learning player has "no model of its opponent of any kind"; the opponent's
  reply is simply part of how the board changes between the player's own moves, and
  the TD backup V(s) <- V(s) + a[V(s') - V(s)] runs between states where the player
  is to move. Use for: why opponent moves are environment from one player's view,
  and why rewards between two own moves belong to the earlier one. (Verified via the
  archive.org full text; the incompleteideas.net site had a certificate error.)

- [Paper: "Policy invariance under reward transformations: Theory and application to reward shaping" — Ng, Harada & Russell (ICML 1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
  Theorem 1: F(s,a,s') = gamma*Phi(s') - Phi(s) is necessary and sufficient for the
  shaped MDP to keep the original optimal policies. Corollary 2: Q' = Q - Phi,
  V' = V - Phi. Remark 1: near-optimal policies preserved. Remark 2: a purely
  potential-based reward makes every policy optimal. Use for: material shaping
  design, why shaping nets to zero over a game, reading the shaped value head.

- [Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" — Silver et al. (2017)](https://arxiv.org/abs/1712.01815)
  Abstract: superhuman play "given no domain knowledge except the game rules",
  tabula rasa self-play, no handcrafted evaluation. Grounds `shaping: none` as the
  outcome-only reference point. The exact reward formulation (game outcome only) is
  in the paper body; verify there before quoting numbers.

- [Paper: "Reverse Curriculum Generation for Reinforcement Learning" — Florensa et al. (CoRL 2017)](https://arxiv.org/abs/1707.05300)
  Abstract: for goal-oriented sparse-reward tasks, start the agent near the goal and
  widen the start-state distribution as it succeeds; needs only one goal state, no
  demonstrations or shaped reward; the curriculum adapts to performance. Use for:
  why puzzle start positions help, and the later adaptive `puzzle_fraction`.

- [Paper: "Learning Montezuma's Revenge from a Single Demonstration" — Salimans & Chen (2018)](https://arxiv.org/abs/1812.03381)
  Abstract: resetting episodes to states along a demonstration turns exploration cost
  that scales exponentially in the number of steps between rewards into roughly
  quadratic. Use for: the argument that a mate hundreds of moves from the opening
  is unreachable by random play, one move away it is not.

- [Data: Lichess puzzle database (CC0)](https://database.lichess.org/#puzzles)
  6.1M rated puzzles as CSV with FEN, UCI moves, rating and themes. FEN is the
  position *before* the opponent's move; the solution starts at move two. Theme
  `mateIn1` gave 898,211 puzzles on 2026-09-16; we use 20,000 train / 2,000
  held-out, converted by `scripts/prepare_puzzles.py`.

## Wisdom (Communities)

<!-- Places to test understanding against practitioners. Optional. -->

## Gaps

Sources still needed, ordered by roadmap priority:

- Engine-based evaluation of learned policies (roadmap/06_evaluation.md §1)
- Reward decay / shaping schedule in sparse two-player games (roadmap/04_rewards_curriculum.md §1)
- Supervised pre-training on expert moves before RL (AlphaGo, Silver et al. 2016, Nature 529) — fetch the paper body before running the comparison arm
- Fictitious self-play and opponent sampling (roadmap/05_opponent_sampling.md §1)
- MCTS combined with a learned policy/value network (roadmap/03_algorithm.md §2)
