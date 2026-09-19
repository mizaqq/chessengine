#!/bin/zsh
# PUNISH (owner, 2026-09-19: "go ahead with it"). START + 120 updates with the one-ply
# rules punisher on every board at punish_epsilon 0.5 (threshold 3); everything else as
# LONG3 / DEPTH20 (mixed finishing slot, human depth uniform over [0,20] if START is
# DEPTH20, lambda 1, LR floor 1e-4, pool, SIL, guidance 0.25). Then games, finishes,
# blunder audit (self / SL), win matrix vs START. Standard 120-update budget.
#   START=experiments/endgame-technique/DEPTH20 zsh experiments/punish-gifts/run_punish.sh
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=${START:-experiments/endgame-technique/LONG3}
OUT=experiments/punish-gifts/PUNISH
mkdir -p $OUT
echo "=== PUNISH start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START max_updates=120 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 punish_epsilon=0.5 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT SL=experiments/human-pretraining/sl --games 20 --seed 1 --out experiments/blunder-audit/punish.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix START=$START PUNISH=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_punish.json 2>&1 | grep -v OMP
echo "=== PUNISH all done $(date +%H:%M:%S)"
