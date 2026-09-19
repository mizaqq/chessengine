#!/bin/zsh
# DEPTH20 (owner, 2026-09-19: "lets go with option 1"). LONG3 + 120 updates with the human
# finishing depth drawn uniformly from the even values in [0, 20] (depth_start = depth_max = 20,
# so the curriculum cannot move): the evaluated depths 10 and 20 appear in training for the
# first time. Everything else as LONG3 (mixed finishing slot, lambda 1, LR floor 1e-4, pool,
# SIL, guidance 0.25). Then games and finishes. Standard 120-update budget.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/DEPTH20
mkdir -p $OUT
echo "=== DEPTH20 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/LONG3 max_updates=120 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.eval_matrix LONG3=experiments/endgame-technique/LONG3 DEPTH20=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_depth20.json 2>&1 | grep -v OMP
echo "=== DEPTH20 all done $(date +%H:%M:%S)"
