#!/bin/zsh
# LONG192B (owner, 2026-09-21: "lets go with 1 and 2"). LONG192 + 600 updates, same recipe
# (curves still rising at 600). Then games, finishes, audit, matrix vs slbig192 / LONG192.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG192; OUT=experiments/width/LONG192B; mkdir -p $OUT
echo "=== LONG192B start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START opponent_prior=experiments/human-pretraining/slbig192 max_updates=600 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
  | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG192B.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLBIG192=experiments/human-pretraining/slbig192 LONG192=$START LONG192B=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long192b.json 2>&1 | grep -v OMP
echo "=== LONG192B all done $(date +%H:%M:%S)"
