## Context

Finishing boards already do everything a technique board needs: a start with a FEN,
a cap, a winner and an optional demonstration line (`FinishStart`), auto-reset with a
success report keyed by the start's depth, and an evaluation that plays fixed starts
in lockstep. The puzzle preparation script converts one Lichess row at a time by
theme and depth. The default layout is 4 puzzle / 2 finishing / 5 mid-game / 5 opening
with the opponent pool on the first five game boards.

## Goals / Non-Goals

**Goals:** an endgame-mate puzzle source; generated bare-king technique starts feeding
the finishing slot; per-set success dials in training and evaluation; default layout
with technique boards instead of human finishing boards.

**Non-Goals:** tablebase demonstrators; "crushing" puzzles played out; new board
roles beyond the finishing slot; changes to rewards or the update.

## Decisions

- **Reuse the finishing slot.** `FinishStart` gains `config: str = ""`. A
  `TechniqueSampler` returns `FinishStart(fen, depth=0, cap, winner, line=(), config=label)`
  and implements `report(depth, success)` as a no-op plus `current_depth = 0`, so the
  vector env and the layout schedule need no new role. The env adds
  `info["finish_config"]`; the rollout routes a labelled result to
  `metrics.add_technique_result(label, success)` instead of `add_finish_result`.
- **Generator.** Place the two kings on distinct non-adjacent squares, then the
  strong side's pieces on free squares; reject if the position is illegal, the weak
  king is in check when it is not to move, the strong side has a mate in one, or the
  position is stalemate/insufficient. Strong side colour random; FEN with the strong
  side to move, no castling, halfmove clock 0. Uniform over the configured sets.
- **Caps.** Perfect-play worst cases: king and queen 10 moves, king and rook 16,
  two rooks 7, queen and rook 6. Caps 40 / 60 / 40 / 40 plies, about twice the
  worst case in plies for the queen and rook, more for the rest.
- **Evaluation.** `src/eval/technique.py`: `evaluate_technique(white, black, sets, caps,
  games, oriented, seed)` builds the held-out starts once per call from a fixed seed
  (same positions every evaluation), plays them with `play_finishes`, reports
  `technique_rate_<label>` and `technique_games_<label>`.
- **Puzzle source.** `prepare_puzzles.py --themes endgame --depth any`: depth taken
  from the `mateIn<N>` theme (1-5), verified as today by depth; output
  `endgame_mate_{train,eval}.csv`. Mix: five sources at 0.2. Held-out `lichess_end`.
- **Config.** `finish_source: technique | human` (default `technique`),
  `technique_sets: [Q, R, RR, QR]`, `technique_caps: {Q: 40, R: 60, RR: 40, QR: 40}`,
  `technique_eval_games: 50`; the human finishing evaluation stays on (it is the
  target dial).

## Hyperparameters

- `technique_sets`: which material sets are drawn. One set = a single skill, fastest
  to learn, no transfer test; many sets = slower per set, broader technique.
- `technique_caps`: plies allowed. At the perfect-play distance the dial measures
  perfection and stays near zero; far above it, sloppy technique passes and the dial
  saturates without teaching efficiency. Twice the worst case is the middle.
- `finish_boards` (2): how much of the batch is technique. 0 = today's run without
  finishing boards; many = puzzles and games starve.
- `technique_eval_games` (50 per set): noise of the dial vs evaluation time
  (about 0.3 s a game).

## Risks / Trade-offs

- Generated positions are off the human distribution; the prior has rarely seen a
  bare king. A near-zero start is expected, not a bug.
- The defender is the learner: as it learns to run to the centre, the task gets
  harder. If the dial stalls after rising, the prior can defend instead.
- Repetition draws: a network that shuffles pieces will draw by threefold repetition
  or the cap; both count as failures, which is the intended signal.
- Fifty-move rule is far above the caps; not a factor.
- The endgame-mate puzzles overlap the existing m1-m4 sets (a puzzle can be in both);
  held-out ids are excluded from all train files by id where they collide.
