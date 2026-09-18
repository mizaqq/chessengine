## Why

Conversion of won positions from 10-20 plies out has not moved in nine arms
(finishing evaluation depth 10 / 20 stuck at 0.07-0.20) while every mate-in-N puzzle
dial rose. The finishing boards rewind human games, where the mate often existed only
because the defender blundered, and Lichess mate puzzles are short middlegame tactics.
The owner asked for "puzzles like mating with queen, mating with rook and other
tactical games that can be finished", then prioritised "the endgames which end in
mate" from Lichess, with "the bare king stuff generated if it doesn't exist". A scan of
the Lichess puzzle database (1.5M rows, 2026-09-18) found 226,603 puzzles themed
endgame + mate (mateIn1 86k, mateIn2 111k, mateIn3 25k) and zero bare-king technique
positions, so rung 1 comes from Lichess and rung 2 is generated.

## What Changes

- **Endgame-mate puzzle rung (Lichess).** `prepare_puzzles.py` gains a theme filter;
  puzzles themed `endgame` and `mateIn1..5` become `data/puzzles/endgame_mate_{train,eval}.csv`
  with the solution line stored. The default puzzle mix becomes five sources at 0.2
  (m1, m2, m3, m4, endgame_mate); evaluation reports `lichess_end_top1`.
- **Technique boards (generated).** The finishing-board slot can be fed by a
  `TechniqueSampler`: random legal positions for a material list (queen v king, rook v
  king, two rooks, queen and rook), strong side to move, no mate in one on the board,
  each with a ply cap. Success = the strong side mates within the cap; failure
  otherwise. Reported per material set. The default layout switches the two
  finishing boards to technique boards (finishing boards have not moved the deep dial).
- **Technique evaluation.** A fixed generated held-out set per material set, played
  every evaluation, reported as `technique_rate_<set>`.
- The demonstrator on technique boards is the existing rules fallback (mate in one,
  forcing mate in two); no tablebases in this change.

## Capabilities

### New Capabilities
- `endgame-technique`: generated technique starts, their caps, success accounting,
  and the technique evaluation; the endgame-mate puzzle source.

### Modified Capabilities
- `finishing-curriculum`: the finishing slot accepts a technique sampler; a start
  carries an optional material label; results with a label are counted per label
  instead of per depth.

## Alternatives considered

- **Keep human finishing records only** (Florensa et al. 2017 reverse curriculum on
  human checkmates, current). Nine arms, depth 10 / 20 flat; the human mate is not a
  forced win after the network's own reply (greedy-attacker check 2026-09-17).
- **Lichess "crushing" puzzles played to the end** (376k available): tactic guided by
  the solution line, then convert. Deferred by the owner in favour of mates first;
  its dial has a ceiling below 1 because "crushing" is an engine advantage, not a
  forced win.
- **Tablebase demonstrator** (Syzygy 3-4 piece files via python-chess): the perfect
  move in every technique position, the strongest possible teacher for rung 2.
  Deferred until the queen rung is seen to stall without it. No source fetched yet
  for tablebase-guided RL; Salimans & Chen 2018 (demonstration starts) is the
  nearest cited idea.
- **Search at play time** (AlphaZero MCTS, Silver et al. 2017): the standard fix for
  long conversions; ruled out by the owner for now (model-free only).

## How we will know it worked

120 updates from POOL_LONG (standard budget), default layout with the pool.
- `technique_rate_KQ`, `_KR`, `_KRR`, `_KQR` (held-out generated, 50 games each, cap
  40 / 60 / 40 / 40 plies): baseline measured on POOL_LONG before the arm.
- `finish_rate_d10`, `finish_rate_d20` on the human finishing set: the real target,
  flat at 0.07-0.20 so far.
- `lichess_end_top1` from its POOL_LONG baseline; m1-m4 and `prior_score` held.
- 40 sampled games: decisive rate (POOL_LONG 22 of 40), mate found (0.26).

Owner's prediction: not given (asked twice; the owner skipped it).
Claude's prediction: technique KQ from a low baseline (0.1-0.3) to above 0.6 within
120 updates; KR lower (0.3-0.5); finish depth 10 unchanged within noise, because the
human finishing positions are middlegames with many pieces where a bare-king plan
does not apply; endgame-mate top-1 rises 5 points.

## What to understand

- Why a forced win by rule (technique) gives a cleaner conversion signal than a human
  mate (which may not be forced after a different reply).
- What the cap does: turns "did not finish in time" into a draw, the same signal as a
  failed finishing board, and why it must exceed the perfect-play mate distance with
  margin (queen: 10 moves worst case; rook: 16).
- Why zero baseline is expected on generated positions the prior has rarely seen, and
  what a rising technique dial with a flat finishing dial would mean (skill learned,
  but the human positions are a different problem).

## Impact

`scripts/prepare_puzzles.py` (theme filter, depth from theme), `src/envs/technique.py`
(new sampler and generator), `src/envs/start_positions.py` (`FinishStart.config`),
`src/envs/open_spiel_vector_env.py` (info `finish_config`), `src/model/training.py`
(route labelled results), `src/training/metrics.py` (`technique_success_rate_<set>`),
`src/eval/technique.py` (held-out evaluation), `src/entrypoints/train.py` (config
keys `finish_source`, `technique_sets`, `technique_caps`, `technique_eval_games`),
`src/configs/train_default.yaml` (five-source mix, technique boards). Data:
`data/puzzles/endgame_mate_{train,eval}.csv` (committed like the other sets).
