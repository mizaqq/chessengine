#!/bin/zsh
# Arm FIN2 (owner, 2026-09-17). After FIN was flat at 120 updates: 4 puzzle / 6
# finishing / 2 mid-game boards (no boards from move one; the draw games carried no
# finishing signal), advance threshold 0.4, 300 updates as the named exception to
# the standard budget, and from update 200 two finishing boards hand back to
# opening play (owner: "at some time in the training we should also add opening
# positions, e.g. at update 200"). Then 40 games and the finishing evaluation.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/finishing-curriculum
echo "=== start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT/FIN2 --set init_from=experiments/human-pretraining/ppo_from_sl \
  max_updates=300 finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 \
  'layout_schedule=[{from_update: 200, finish_boards: 4, midgame_boards: 2}]' 2>&1 \
  | grep -E "layout|done in|Traceback|Error" || true
$PY -m scripts.eval_games $OUT/FIN2 --games 40 --seed 0 --out $OUT/FIN2/games.json 2>&1 | tail -1
$PY -m scripts.eval_finishes $OUT/FIN2 --out $OUT/FIN2/finishes.json 2>&1 | grep finish_rate
echo "=== all done $(date +%H:%M:%S)"
