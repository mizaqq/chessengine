#!/bin/zsh
# LONG256 (owner, 2026-09-21: "The next one we will test should be 256"). slhi256 (high-Elo,
# 256 filters) + 600 updates, today's recipe, own prior as pool opponent; counts unchanged.
# Then games, finishes, audit, matrix vs slhi256 / LONG192B / LONG192. GPU: launch outside the sandbox.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/LONG256; mkdir -p $OUT
echo "=== LONG256 start $(date +%H:%M:%S) from $PRIOR"
$PY -m scripts.run_experiment --out $OUT --set init_from=$PRIOR opponent_prior=$PRIOR max_updates=600 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
  | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG192=experiments/width/LONG192 LONG192B=experiments/width/LONG192B LONG256=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long256.json 2>&1 | grep -v OMP
echo "=== LONG256 all done $(date +%H:%M:%S)"
