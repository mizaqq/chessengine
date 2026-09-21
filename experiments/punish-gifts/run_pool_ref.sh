#!/bin/zsh
# POOL_REF (owner, 2026-09-21: "okay go 1 now"). DEPTH20 + 120 updates with the referee on the
# five pool opponents only (pool_referee true, punisher off): the opponent always plays a rules
# mate or a free capture >= 3 when one exists; learner plies untouched. Everything else as
# DEPTH20. Then games, finishes, blunder audit, win matrix vs DEPTH20 / PUNISH.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/endgame-technique/DEPTH20
OUT=experiments/punish-gifts/POOL_REF
mkdir -p $OUT
echo "=== POOL_REF start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START max_updates=120 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 pool_referee=true 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT SL=experiments/human-pretraining/sl --games 20 --seed 1 --out experiments/blunder-audit/pool_ref.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix DEPTH20=$START PUNISH=experiments/punish-gifts/PUNISH POOL_REF=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_pool_ref.json 2>&1 | grep -v OMP
echo "=== POOL_REF all done $(date +%H:%M:%S)"
