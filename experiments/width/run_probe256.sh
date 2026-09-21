#!/bin/zsh
# PROBE256: why do continuations of LONG256 lose to the prior from the opening within 100
# updates while puzzles rise? LONG256 + 100 updates (lr 5e-5, target_kl 0.02, alarm OFF so the
# checkpoint survives), then 200-game matches vs slhi256 from the opening and from mid-game
# starts, for both LONG256 and the probe.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256; PRIOR=experiments/human-pretraining/slhi256; OUT=experiments/width/PROBE256; mkdir -p $OUT
echo "=== PROBE256 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$START opponent_prior=$PRIOR max_updates=100 learning_rate=0.00005 min_lr=0.00005 target_kl=0.02 stop_below_prior=null prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 2>&1 \
  | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
$PY -m scripts.eval_matrix SLHI256=$PRIOR LONG256=$START PROBE256=$OUT --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_probe256_opening.json 2>&1 | grep -v OMP
echo "=== PROBE256 all done $(date +%H:%M:%S)"
