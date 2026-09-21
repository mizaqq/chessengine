#!/bin/zsh
# WIDTH comparison (owner, 2026-09-21: "compare if 192x got better results"). Fair test: same
# recipe, same budget, same pipeline point. CTRL128 = 128-filter SL prior + 120 updates;
# RL192 = 192-filter SL prior (sl192) + 120 updates. Today's full recipe (mixed finishing
# slot, human depth uniform over [0,20], lambda 1, LR floor 1e-4, pool with each network's
# own prior, SIL, guidance 0.25). Then puzzles/games/finishes and a four-way matrix.
# Waits for SL192 and POOL_REF to finish so at most one training runs at a time.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "all done" experiments/human-pretraining/run_sl192.log && grep -q "all done" experiments/punish-gifts/run_pool_ref.log; do sleep 60; done
COMMON="max_updates=120 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001"
for ARM in ${ARMS:-CTRL128 RL192}; do
  if [[ $ARM == CTRL128 ]]; then PRIOR=experiments/human-pretraining/sl; else PRIOR=experiments/human-pretraining/sl192; fi
  OUT=experiments/width/$ARM; mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S) from $PRIOR"
  $PY -m scripts.run_experiment --out $OUT --set init_from=$PRIOR opponent_prior=$PRIOR ${=COMMON} 2>&1 \
    | grep -E "done in|Traceback|Error" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
  echo "=== $ARM done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl SL192=experiments/human-pretraining/sl192 CTRL128=experiments/width/CTRL128 RL192=experiments/width/RL192 --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_width.json 2>&1 | grep -v OMP
echo "=== WIDTH all done $(date +%H:%M:%S)"
