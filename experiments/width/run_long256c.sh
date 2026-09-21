#!/bin/zsh
# LONG256C (owner, 2026-09-21: "go, install all three"). LONG256 + 600 updates with the three
# guards: learning_rate 5e-5 (arm LR5), target_kl 0.03 (early stop per update), prior alarm
# (stop_below_prior 0.5, patience 2, restores the last good weights). Everything else as
# LONG256. Then games, finishes, audit, matrix vs slhi256 / LONG256. GPU: outside the sandbox.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256; PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/LONG256C; mkdir -p $OUT
echo "=== LONG256C start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START opponent_prior=$PRIOR max_updates=600 learning_rate=0.00005 min_lr=0.00005 target_kl=0.03 stop_below_prior=0.5 stop_patience=2 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 2>&1 \
  | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256C.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256=$START LONG256C=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long256c.json 2>&1 | grep -v OMP
echo "=== LONG256C all done $(date +%H:%M:%S)"
