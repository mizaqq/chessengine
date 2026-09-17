## Context

`_collect_rollout` samples `Categorical(probs=p)` per side and stores
`dist.log_prob(a)` as `old_log_prob`; `update_model` forms the PPO ratio from it.
The vector env exposes `is_puzzle_env(i)` / `is_finish_env(i)`; the layout can
change during a run (`set_layout`), so the noisy-board mask is recomputed every
update.

## Decisions

- **Mixture, not temperature.** `b = (1-eps) p + eps d`, `d ~ Dir(alpha)` over
  legal moves (AlphaZero root noise). A temperature would also flatten the
  confident-right positions; the mixture adds a floor that is independent of how
  wrong the prior is.
- **Behaviour log-prob in the ratio.** `old_log_prob = log b(a)`. The PPO
  objective then is the importance-weighted surrogate with respect to the data
  distribution; the clip bounds `pi_new(a)/b(a)`, so a move drawn mainly from the
  noise (`pi(a) << b(a)`) gets a small ratio and is raised by at most the clip
  per update. This is the honest version; storing `log pi(a)` instead would
  overstate the ratio for noise-driven picks. `approx_kl` and `clip_fraction`
  are relative to `b` on noisy boards (documented in the log).
- **Noise per step per board.** A fresh Dirichlet sample per move (AlphaZero: per
  root). Cheap: one `torch.distributions.Dirichlet` sample over the legal count.
- **Diagnostic** `explore_offprior_share`: fraction of noisy-board moves with
  `pi(a) < 0.05`, accumulated by the metrics aggregator per window. With eps 0.25
  and ~35 legal moves the expected value is a few percent; it shows the device is
  doing something and how much.
- **Entropy logging** stays the network's entropy, so entropy curves remain
  comparable across arms.

## Hyperparameters

- `explore_epsilon` (0 default, 0.25 in the arm): weight of the noise. 0 = today;
  near 1 = random play on those boards, the update then learns from near-uniform
  data and the ratio clip throttles every step.
- `explore_alpha` (0.3): Dirichlet concentration. Small alpha (0.03, Go) puts the
  noise mass on one or two random moves, so a given position gets a big push
  toward one random move occasionally; large alpha spreads it evenly, like
  eps-uniform. AlphaZero scales it inversely with the legal-move count; chess 0.3.
- `explore_boards` (`curriculum`): where the noise acts. `all` also perturbs the
  opening boards; kept off there because those games are already mostly draws and
  noise makes them noisier.

## Risks

- Off-policy data with clipping: a mate drawn from the noise with pi(a) = 0.01
  gets ratio ~0.17 and a large positive advantage; the unclipped side of the
  objective still passes the gradient (clip only bounds the gain side above
  1+eps), so the move is raised, but by a clipped step. Expect slow, steady
  gains, not a jump.
- Draw-heavy finishing boards at depth > 0 may get noisier without benefit.
