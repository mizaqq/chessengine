## Context

Label-guided exploration (change `exploration-noise`) already builds a behaviour
mixture from a per-board demonstration set, samples it by component so that a
demonstrator pick can be told from a network pick, stores the mixture's log-prob and
flags the episode as teacher-touched for the curricula. The demonstration today is the
puzzle key move, the human line, a rules mate, or a forcing mate. Guidance runs on
curriculum boards only (`guide_boards: curriculum`, epsilon 0.25). Pool boards route
the opponent's plies to a frozen model outside this path.

## Goals / Non-Goals

**Goals:** make a hung piece or an allowed mate cost the gifter on every board where the
learner defends; teach the learner to take them; keep PPO's stored log-probability
honest; keep the curricula clean of teacher-made wins.

**Non-Goals:** search deeper than one ply; changing pool-opponent plies; a material term
in the reward; removing existing guidance.

## Decisions

- **One rules helper, shared with the audit.** `capture_gain` and `free_capture_actions_of`
  move from `scripts/blunder_audit.py` into `src/envs/open_spiel_env.py` next to
  `mating_actions_of`; the audit imports them. The env gains `punish_actions()` =
  mates ∪ free captures (threshold from the env, default 3).
- **Two demonstration channels, one mixture.** The rollout asks each board for its
  label demonstration (as now) and, on punish boards, its punishing moves. Per row it
  picks (demo set, epsilon): label demo with `guide_epsilon` if non-empty, else
  punishing moves with `punish_epsilon`. `guided_probs` takes a per-row epsilon.
  Alternative: a second mixture layer; rejected because two mixtures make the
  log-probability harder to reason about and both sources are rules-clean.
- **`punish_epsilon`** — probability the punishing move is played when available. 0: off,
  today's behaviour, punish rate stays ~0.5 for material and ~0.13 for mates. 0.5:
  punish rate ~0.75 / ~0.56 and the punishing ply keeps a usable PPO ratio. 1: every gift
  punished, but the ply is drawn purely from the uniform demo, so its ratio is clipped
  and the network does not learn to punish from its own gradient, only the gifter learns.
- **`punish_threshold`** — minimum net material (after the cheapest recapture) for a
  capture to count. 1: every pawn grab is a demo, the behaviour policy becomes greedy and
  entropy drops. 3 (default): minor piece or better. 9: only queens.
- **`punish_boards`** default `all`: the gifts were measured in games, and the finishing
  and technique defenders are also us.
- **Teacher-touched flag.** A punisher pick sets the same per-board flag as the label
  demonstrator, so technique/finishing wins after a punisher capture do not advance the
  curricula (lesson from CURSIL/CURSIL2).
- **Metrics.** `punish_count`, `punish_picks` in the update summary; `add_punish` on
  `TrainingMetrics`.

## Risks / Trade-offs

- Exploration on game boards changes: with probability 0.5 the learner takes every free
  piece and mate. Entropy may fall further (LONG3 0.27). Watch it; lower epsilon if it
  falls below ~0.2.
- One-ply gains ignore poisoned pieces (a capture that loses to a follow-up). The
  punisher will sometimes take a poisoned piece; the outcome signal corrects that over
  time, and the audit's punish rate is a lower bound for the same reason.
- Pool boards: opponent plies stay unpunishing; only the learner's colour is fixed there.
