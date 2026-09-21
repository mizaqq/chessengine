#!/bin/zsh
# LONG256H (owner, 2026-09-21 night: "okay agreed"). LONG256G + 600 updates, guarded: optimizer
# restored, lr 1e-4, target_kl 0.02, alarm on 40 games (restores the last good weights). Then
# games, finishes, audit, 200-game matrix vs slhi256 / LONG256G from the opening.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256G; PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/LONG256H; mkdir -p $OUT
echo "=== LONG256H start $(date +%H:%M:%S) from $START"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START opponent_prior=$PRIOR max_updates=600 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
  | grep -E "^device|optimizer state|done in|Traceback|Error|below" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256H.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256G=$START LONG256H=$OUT --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_long256h.json 2>&1 | grep -v OMP
echo "=== LONG256H all done $(date +%H:%M:%S)"
