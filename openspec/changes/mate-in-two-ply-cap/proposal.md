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

Owner (while the data built): "no need to run 4 arms". Only BOTH runs; the
baseline is `experiments/ppo-gae/C2_ppo_lr1e-4` (same config, mate-in-one only, no
cap) read at updates 120 and 150.

Then: "we just stay with mates in 2 ... run with mate-in-2 and ply limit with a
longer run that will go for more episodes". The cap-only arm was cancelled; LONG
continues from BOTH's checkpoint for 480 more updates (600 total, 38,400 plies
per board), read against C2 at 250 (16,000 plies) and BOTH at 120.

Read: `lichess_top1`, `lichess_m2_top1`, `puzzle_solved_rate_m1/_m2`, games
finished per update, draw rate, opening entropy, and 40-game evaluation (mate
rate, mean plies). Expect CAP to raise finished games per update and shorten
evaluation games only if the policy changes; the cap itself does not apply in
evaluation games.

### Outcome (2026-09-16, BOTH 120 updates + LONG 480 = 600 updates x 64 plies, seed 42)

| update | Lichess m1 top-1 | Lichess m2 top-1 | train solved m1 / m2 | opening entropy | puzzle entropy | games per window | clip / KL |
|---|---|---|---|---|---|---|---|
| 50 | 0.10 | 0.07 | 0.07 / 0.01 | 0.55 | 0.50 | 47 | 0.42 / 0.074 |
| 120 | 0.22 | 0.08 | 0.20 / 0.01 | 0.44 | 0.31 | 47 | 0.31 / 0.058 |
| 300 | - | - | 0.34 / 0.12 | 0.43 | 0.29 | 48 | 0.29 / 0.068 |
| 600 | **0.444** | **0.351** | 0.42 / 0.23 | 0.42 | 0.15 | 45 | 0.20 / 0.039 |

Reference: C2 (same config, mate-in-one only, no cap) reached 0.42 at 250 updates.
40 uncapped evaluation games: mean 186 plies (BOTH at 120: 201; C2: 222), draw
rate 0.85, decisive 7.5%, 3 mates in 31 chances.

Reading:
- **The second rung was learned, late.** Mate-in-two solved rate sat at ~1% for
  200 updates (the reward needs two right choices out of ~30 each), then climbed
  0.03 -> 0.12 -> 0.23 between updates 200 and 600; held-out first-move accuracy
  went 0.08 -> 0.35. Mate-in-one caught up with the mate-in-one-only baseline
  (0.444 vs 0.42) despite half the attempts.
- **The take-off needed the longer run.** At the owner's standard 120-update
  budget the rung looked dead. Sparse two-step rewards show a plateau then a
  knee; the budget rule needs an exception for new curriculum rungs, or partial
  credit for the verified forcing move so the first decision gets its own signal.
- **Ply cap**: games finished per window 45-48 vs 19-21 before; training draw
  rate rose to 0.98 (most games now end by the cap). Evaluation games got shorter
  (186 plies) and mate chances rarer (31 vs 96-120 opportunities): the policy
  plays fewer positions where a mate is on the board. In-game mate rate 3/31 is
  too small a count to read.
- Opening entropy stayed 0.42-0.55 all run (PPO at 1e-4 holding); puzzle-board
  entropy fell to 0.15 as the puzzles were learned.
- Cost 14.5-17.5 s/update.

## What to understand

- A curriculum rung is "one more decision before the reward"; the defender's
  reply is environment (Sutton & Barto tic-tac-toe view).
- Why a lenient reward (any mate within two moves) would teach the wrong first
  move, and what the key-move check costs.
- Why a ply cap is a training device, not a rule change: AlphaZero's reason.
