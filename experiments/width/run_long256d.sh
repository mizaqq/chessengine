#!/bin/zsh
# LONG256D (owner, 2026-09-21: "go with 0.02"). LONG256 + 600 updates: lr 5e-5, target_kl 0.02
# (whole-batch KL before each pass, stop at 1.5x), prior alarm on 40 games (threshold 0.5,
# patience 2). Everything else as LONG256. Then games, finishes, audit, matrix vs slhi256 / LONG256.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256; PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/LONG256D; mkdir -p $OUT
echo "=== LONG256D start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START opponent_prior=$PRIOR max_updates=600 learning_rate=0.00005 min_lr=0.00005 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 2>&1 \
  | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256D.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256=$START LONG256D=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long256d.json 2>&1 | grep -v OMP
echo "=== LONG256D all done $(date +%H:%M:%S)"
