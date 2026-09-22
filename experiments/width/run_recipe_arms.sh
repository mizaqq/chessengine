#!/bin/zsh
# Recipe arms (owner, 2026-09-22: "Then we start 1 and 2"). Why does the policy start losing to
# the prior once game entropy falls to ~0.2? Two guarded runs from slhi2_256, 600 updates each
# with the alarm (the alarm's update, or its absence, is the result):
#   DECAY: learning rate halves every 100 updates down to 2.5e-5 (the 256 line ran flat at 1e-4).
#   NOPOOL: opponent pool holds the prior only (no snapshots of the learner's sharp self).
# Everything else as LONG256X. Then standard evals and a 200-game matrix vs slhi2_256 / LONG256X.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PRIOR=experiments/human-pretraining/slhi2_256
COMMON="init_from=$PRIOR opponent_prior=$PRIOR max_updates=600 target_kl=0.02 stop_below_prior=0.5 stop_patience=2 prior_eval_games=40 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20"
for ARM in DECAY NOPOOL; do
  if [[ $ARM == DECAY ]]; then EXTRA="lr_decay_interval=100 lr_decay_factor=0.5 min_lr=0.000025"; else EXTRA="min_lr=0.0001 opponent_pool_size=0"; fi
  OUT=experiments/width/$ARM; mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S) ($EXTRA)"
  $PY -m scripts.run_experiment --out $OUT --set ${=COMMON} ${=EXTRA} 2>&1 | grep -E "^device|done in|Traceback|Error|below" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/$ARM.json 2>&1 | grep -v OMP
  echo "=== $ARM done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix SLHI2_256=$PRIOR LONG256X=experiments/width/LONG256X DECAY=experiments/width/DECAY NOPOOL=experiments/width/NOPOOL --games 100 --seed 11 --workers 1 --out experiments/win-matrix/matrix_recipe_arms.json 2>&1 | grep -v OMP
echo "=== RECIPE ARMS all done $(date +%H:%M:%S)"
