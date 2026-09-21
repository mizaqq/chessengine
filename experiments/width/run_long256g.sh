#!/bin/zsh
# LONG256G (owner, 2026-09-21: "ok"). slhi256 + 600 updates with the guards on from update 1:
# lr 1e-4 (as LONG256), target_kl 0.02 (whole-batch KL before each pass, stop at 1.5x), prior
# alarm on 40 games (0.5, patience 2), optimizer saved. Then games, finishes, audit, 200-game
# matrix vs slhi256 / LONG256 from the opening.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/LONG256G; mkdir -p $OUT
echo "=== LONG256G start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$PRIOR opponent_prior=$PRIOR max_updates=600 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001 2>&1 \
  | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/LONG256G.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256=experiments/width/LONG256 LONG256G=$OUT --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_long256g.json 2>&1 | grep -v OMP
echo "=== LONG256G all done $(date +%H:%M:%S)"
