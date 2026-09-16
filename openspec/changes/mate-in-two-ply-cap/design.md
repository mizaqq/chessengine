## Context

- Puzzle boards `[0, num_puzzle_envs)` reset from `sampler.sample()` with
  `one_move=True`; `is_done = terminal or plies >= 1`; non-terminal done ->
  `"puzzle_miss"`. Mixed sampler exists (`MixedSampler`).
- OpenSpiel `state.action_to_string(player, a)` equals python-chess SAN for every
  legal move (checked on castling, promotion, ordinary moves), so UCI key moves
  map to action ids via `chess.Board(fen).parse_san(san).uci()`.
- Lichess `mateIn2` rows: `Moves` = opp, key, reply, mate (4 plies); FEN before
  the first. 807,425 rows in the 2026-09-16 dump.

## Decisions

### D1. Env episode control
`reset(fen=None, puzzle_moves=0, key_moves=None, max_plies=None)`. `mover` = side to
move at reset. On each step, if the side to move is the mover: `mover_moves += 1`;
if `mover_moves == 1 and key_actions is not None and action not in key_actions`:
`failed = True`. `is_done = terminal or (puzzle and (failed or mover_moves >=
puzzle_moves)) or (max_plies and plies >= max_plies)`. `game_result`: terminal ->
rules; puzzle non-terminal -> `"puzzle_miss"`; capped -> `"draw"`. Key actions
computed at reset from `key_moves` via SAN. For `puzzle_moves == 1` no key check
(any mate is right by definition).

### D2. Data
`convert_row(row, depth)`: depth 1 unchanged (key_moves = all mates). Depth 2:
`len(moves) == 4`; push moves[0]; key = moves[1]; verify: push key; not terminal;
for every reply there exists a mating move (python-chess); else reject. Columns:
`puzzle_id, fen, key_moves, mate_in, rating`; loader accepts the old
`mating_moves` header (mate_in defaults to 1). Files `mate_in_2_train.csv`
(50,000) and `mate_in_2_eval.csv` (2,000), seed 0.

### D3. Sampler returns `Puzzle`
`StartPositionSampler.sample()` and `MixedSampler.sample()` return the `Puzzle`;
vector envs call `env.reset(p.fen, puzzle_moves=p.mate_in, key_moves=p.key_moves)`
and add `info["puzzle_depth"][i] = p.mate_in` on episode end. Async workers get
`max_plies` as an argument.

### D4. Metrics and eval
`add_puzzle_result(solved, depth=1)` -> `puzzle_solved_rate` plus
`puzzle_solved_rate_m<depth>` for every depth seen in the window.
`evaluate_puzzles` (alias `evaluate_mate_in_one`): target actions = rules-engine
mates when `mate_in == 1`, else key actions; `puzzle_top1` = argmax in targets,
`puzzle_mate_prob` = mass on targets.

### D5. Config
`max_plies` (int or null, default null; yaml 120), validated >= 2 when set.
Default yaml: sources `[{mate_in_1, 0.5}, {mate_in_2, 0.5}]`, eval sets
`lichess`, `lichess_m2`, `selfplay`.

## Risks
- A wrong first move ends the episode without the reply: the network never sees
  the position after its error on puzzle boards. Accepted: game boards provide
  that; the puzzle board's job is the forcing move.
- Ply cap 120 may cut real endgames; it is applied to training boards only and
  the 40-game evaluation is uncapped (400-ply limit as before).
- Key-move mapping via SAN: tested on castling/promotion; if a SAN ever fails to
  parse, reset raises with the FEN and SAN in the message.
