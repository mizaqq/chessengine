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

- [Paper: "Mastering Chess and Shogi by Self-Play..." — Silver et al. (2017), methods p.13](https://arxiv.org/abs/1712.01815)
  "The board is oriented to the perspective of the current player. The M feature
  planes are composed of binary feature planes indicating the presence of the
  player's pieces ... and a second set of planes indicating the presence of the
  opponent's pieces", plus constant planes for colour, move count, castling,
  repetitions, no-progress count. Use for: one network for both colours;
  `src/model/orientation.py`. OpenSpiel's move ids are already mover-relative
  (verified 2026-09-16), its observation planes are absolute.

- [Paper: "Mastering the game of Go without human knowledge" — Silver et al. (Nature 550, 2017), unformatted manuscript](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)
  Read 2026-09-16 (pp. 3-6, 24-27). Search: each edge stores {N, W, Q, P};
  select a = argmax Q + U with U = c_puct * P * sqrt(sum_b N(s,b)) / (1 + N(s,a));
  leaf expanded and evaluated by the network once, edges initialised N=W=Q=0,
  P=p_a; backup N += 1, W += v, Q = W/N; play pi(a) proportional to N^(1/tau),
  tau = 1 for the first 30 moves then -> 0; root prior noise P = (1-eps) p +
  eps eta, eta ~ Dir(0.03), eps = 0.25; subtree reused after the played move.
  Training: loss l = (z - v)^2 - pi^T log p + c||theta||^2 (eq. 1), c = 1e-4,
  cross-entropy and MSE weighted equally because rewards are unit scaled;
  1,600 simulations per move; resignation when root and best child value below
  a threshold tuned to <5% false positives. MCTS is "a powerful policy
  improvement operator" and self-play with search "a powerful policy evaluation
  operator" (p. 3). Raw network without search reached 3,055 Elo vs 5,185 with
  search (p. 12). Use for: the search-in-the-loop change; AlphaZero (above)
  inherits these parameters except where it says otherwise (800 simulations,
  Dir alpha 0.3 for chess, lr 0.2 dropped 3x, max game length -> draw).

- [Paper: "Mastering the game of Go with deep neural networks and tree search" — Silver et al. (Nature 529, 2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  Read 2026-09-16 (pp. 484-485, Methods "Policy network: classification",
  "Policy network: reinforcement learning", "Value network: regression"). SL
  policy: 13 layers, trained on 30M positions from 160k KGS games (6-9 dan) to
  predict the human move by stochastic gradient ascent on log p(a|s); 57.0%
  held-out accuracy (44.4% prior state of the art); minibatch 16, step 0.003
  halved every 80M steps, 340M steps. RL stage: "weights rho are initialized to
  the same values" as the SL network, REINFORCE with a value baseline against a
  pool of earlier policies ("randomizing from a pool of opponents ... stabilizes
  training by preventing overfitting to the current policy"); the RL policy won
  >80% of games against the SL policy and 85% against Pachi with no search.
  Value network: regressing on all positions of KGS games overfit ("successive
  positions are strongly correlated ... regression target is shared for the
  entire game"; test MSE 0.37 vs 0.19 train), fixed by one position per
  self-play game. Caution (p. 486): the SL policy worked better *inside search*
  than the stronger RL policy, "presumably because humans select a diverse beam
  of promising moves". Use for: supervised pretraining on Lichess games before
  PPO (change `human-pretraining`), sampling few positions per game, and the
  opponent-pool idea for later.
- [Data: Lichess standard rated games database (CC0)](https://database.lichess.org/)
  Monthly PGN .zst files; counts at `standard/counts.txt`. 2013-06: 293,459
  games, 33 MB; 2013-12: 578,262 games, 92 MB; 2014-06: 961,868 games, 182 MB.
  Headers carry WhiteElo/BlackElo/Result. Use for: human-move pretraining and
  mid-game start positions.

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

- [Paper: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero) — Schrittwieser et al. (2020)](https://arxiv.org/abs/1911.08265)
  Abstract read 2026-09-17: learns a model that predicts only "the reward, the
  action-selection policy, and the value function" and plans with tree search
  inside that model, "without any knowledge of their underlying dynamics";
  matched AlphaZero on chess/shogi/Go. Use for: the search-with-learned-model
  option when a perfect simulator is not available.

- [Paper: "Muesli: Combining Improvements in Policy Optimization" — Hessel et al. (2021)](https://arxiv.org/abs/2104.06159)
  Abstract read 2026-09-17: regularised policy optimisation with model learning
  as an auxiliary objective; matches MuZero on Atari "without using deep search";
  "acts directly with a policy network". Use for: the model-free-at-play middle
  road between PPO and MuZero.

- [Paper: "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games" (NFSP) — Heinrich & Silver (2016)](https://arxiv.org/abs/1603.01121)
  Abstract read 2026-09-17: plain self-play RL "diverged" in Leduc poker while
  NFSP (best-response network + average-policy network) approached a Nash
  equilibrium; no game model needed. Use for: roadmap 05 opponent sampling.

- [Paper: "Mastering Diverse Domains through World Models" (DreamerV3) — Hafner et al. (2023)](https://arxiv.org/abs/2301.04104)
  Abstract read 2026-09-17: "learns a model of the environment and improves its
  behavior by imagining future scenarios"; one configuration across 150+ tasks.
  Use for: the learned-world-model family, contrast with MuZero (imagined
  rollouts for policy gradient vs tree search).

- [Paper: "Reverse Curriculum Generation for Reinforcement Learning" — Florensa et al. (2017)](https://arxiv.org/abs/1707.05300)
  Abstract read 2026-09-17: the agent "is trained in reverse, gradually learning
  to reach the goal from a set of start states increasingly far from the goal";
  the start distribution "adapts to the agent's performance"; needs only "a
  single state in which the task is achieved". Use for: the finishing boards
  (start k plies before a human checkmate, depth rises with the success rate);
  the puzzle boards are the k = 0 special case.

- [Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero preprint) — Silver et al. (2017)](https://arxiv.org/abs/1712.01815)
  Read 2026-09-17 (pp. 3-4, 13-15): loss (z - v)^2 - pi^T log p + c||theta||^2
  (eq. 1); outcome -1/0/+1 only; "the noise that is added to the prior policy to
  ensure exploration ... is scaled in proportion to the typical number of legal
  moves" (p.4); Methods p.14: "Dirichlet noise Dir(alpha) was added to the prior
  probabilities in the root node ... alpha = {0.3, 0.15, 0.03} for chess, shogi
  and Go"; evaluation "selects moves greedily" (p.15); 800 simulations per move;
  chess input 119 planes (8-step history), policy 8x8x73. Use for: exploration
  noise (alpha 0.3 for chess) and the later search change.

- [Paper: "Self-Imitation Learning" — Oh, Guo, Singh & Lee (ICML 2018)](https://arxiv.org/abs/1806.05635)
  Read 2026-09-17 (pp. 1-8): replay buffer of (s, a, R) with Monte-Carlo return R;
  loss L_policy = -log pi(a|s) (R - V(s))_+ and L_value = 1/2 ||(R - V(s))_+||^2
  (eq. 2-3): only state-actions whose past return beat the current value estimate
  get gradient; sampled with priority (R - V)_+; no importance ratio, so it works
  from any past behaviour. M = 4 SIL updates per A2C update (Atari), 10 per PPO
  iteration (MuJoCo). Helped most on sparse/delayed-reward tasks; caveat (5.4):
  "sometimes gets stuck at a sub-optimal policy" when exploitation is excessive,
  fixed by fewer SIL updates or a smaller weight later. Appendix A read 2026-09-18:
  PPO+SIL (MuJoCo) M = 10 SIL updates per batch, SIL batch 512, loss weight 0.1,
  value weight 0.01 or 0.05, buffer 50,000, prioritisation exponent 0.6 or 1.0, bias
  correction 0.1; A2C+SIL (Atari) M = 4, batch 512, loss weight 1, value weight 0.01,
  buffer 1e5. Use for: replaying our own won games on finishing boards (the
  demonstrator becomes the past self); change `self-imitation-curriculum`.

- [Paper: "Emergent Complexity via Multi-Agent Competition" — Bansal et al. (ICLR 2018)](https://arxiv.org/abs/1710.03748)
  Read 2026-09-17 (pp. 4-8): "training agents against the most recent opponent
  leads to imbalance ... the other agent is unable to recover" (4.2); sample the
  opponent from Uniform(delta*v, v) over checkpoint history; delta = 0 (whole
  history) best for Ant, delta = 0.5 for Humanoid, delta = 1 (latest only) worst
  in both (Table 1). Evaluation: win-rate matrix between agents trained with each
  delta. Also 5.5.2: long self-play "over-fitting to the behavior of the opponent",
  cured by an ensemble pool. Use for: opponent pool from saved checkpoints, and
  the checkpoint-vs-checkpoint win matrix as the "cycling" probe.

## Wisdom (Communities)

<!-- Places to test understanding against practitioners. Optional. -->

## Gaps

Sources still needed, ordered by roadmap priority:

- Engine-based evaluation of learned policies (roadmap/06_evaluation.md §1)
- Reward decay / shaping schedule in sparse two-player games (roadmap/04_rewards_curriculum.md §1)
