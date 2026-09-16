## Why

After `add-ppo-gae` the network learns mate-in-one puzzles (held-out 0.42-0.44)
but games still run 200-220 plies and end in repetition or fifty-move draws, with
a 5-9% in-game mate rate. Two things follow from the evidence and the sources:
(1) mate-in-one needs no plan; the next rung of the reverse curriculum (Florensa
et al. 2017) is mate-in-two, which needs one look-ahead and a reply from the
opponent; (2) AlphaZero terminated over-long games as draws: "Chess and shogi
games exceeding a maximum number of steps (determined by typical game length)
were terminated and assigned a drawn outcome" (Silver et al. 2017, Methods p.12).
Our boards spend most plies in dead positions; a cap ends them and resets sooner.

## Decisions taken in conversation (2026-09-16)

- Owner: "mate in 2 and ply cap". Draw reward stays -0.5: "the alphazero
  approach with win/lose wont stand in our scenarios with small network"
  (AlphaZero used 0 for a draw with win/loss +-1, p.3).
- Honest reward on mate-in-two (Claude's call after the question stayed open,
  consistent with the owner's earlier objection to learning "how to finish badly
  played games"): the win counts only when the mover's first move is a key move
  that forces mate; any other first move ends the episode at once as a miss.
- Ply cap 120 (Claude's guess; AlphaZero: "determined by typical game length",
  ours is pathological at 200+). Config `max_plies`, null = no cap.
- Puzzle boards stay at 4, drawn 50/50 from mate-in-one and mate-in-two.
- Budget: 120 updates x 64 plies per arm (owner's standard 7,500 plies/board).

Owner's prediction: not yet given (asked for the mate-in-two held-out top-1 after
120 updates).

## What Changes

- `Puzzle` gains `mate_in` and `key_moves`; `scripts/prepare_puzzles.py --depth 2`
  builds `mate_in_2_{train,eval}.csv` from the 807,425 `mateIn2` puzzles, keeping
  only puzzles whose key move is verified (python-chess) to force mate against
  every reply.
- `OpenSpielEnv.reset(fen, puzzle_moves, key_moves, max_plies)`: puzzle episodes
  end after the mover's `puzzle_moves`-th move (or at once when the first move is
  not a key move, `puzzle_moves > 1`); game boards end as a draw at `max_plies`.
- Vector envs pass the sampled `Puzzle` to reset and report `info["puzzle_depth"]`.
- Metrics: `puzzle_solved_rate_m1` / `_m2` beside the overall rate.
- Evaluation: `evaluate_puzzles` scores mate-in-one against rules-engine mates and
  mate-in-two against the key move; keys unchanged, new held-out set `lichess_m2`.
- Config: `max_plies: 120`, mixed puzzle sources, `lichess_m2` eval set.

## How we will know it worked

Arms from the current default (PPO, lr 1e-4, 64 plies), 120 updates, seed 42:

| Arm | puzzle sources | max_plies |
|-----|----------------|-----------|
| base | mate-in-one | none |
| M2 | mate-in-one + mate-in-two (50/50) | none |
| CAP | mate-in-one | 120 |
| BOTH | mate-in-one + mate-in-two | 120 |

Read: `lichess_top1`, `lichess_m2_top1`, `puzzle_solved_rate_m1/_m2`, games
finished per update, draw rate, opening entropy, and 40-game evaluation (mate
rate, mean plies). Expect CAP to raise finished games per update and shorten
evaluation games only if the policy changes; the cap itself does not apply in
evaluation games.

## What to understand

- A curriculum rung is "one more decision before the reward"; the defender's
  reply is environment (Sutton & Barto tic-tac-toe view).
- Why a lenient reward (any mate within two moves) would teach the wrong first
  move, and what the key-move check costs.
- Why a ply cap is a training device, not a rule change: AlphaZero's reason.
