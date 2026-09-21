#!/bin/zsh
# WIDTH long runs (owner, 2026-09-21: "the actual test will be long run"). LONG128 = big-data
# 128-filter prior + 600 updates; LONG192 = big-data 192-filter prior + 600 updates. Today's full
# recipe, each network's own prior as pool opponent; counts kept at the agreed 600 so results
# line up with LONG2 / LONG3 / PUNISH_LONG. Then games, finishes, blunder audit, matrix vs the
# priors and LONG3. GPU: launch outside the sandbox.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
COMMON="max_updates=600 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20 min_lr=0.0001"
for ARM in ${ARMS:-LONG128 LONG192}; do
  if [[ $ARM == LONG128 ]]; then PRIOR=experiments/human-pretraining/slbig128; else PRIOR=experiments/human-pretraining/slbig192; fi
  OUT=experiments/width/$ARM; mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S) from $PRIOR"
  $PY -m scripts.run_experiment --out $OUT --set init_from=$PRIOR opponent_prior=$PRIOR ${=COMMON} 2>&1 \
    | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/$ARM.json 2>&1 | grep -v OMP
  echo "=== $ARM done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix SLBIG128=experiments/human-pretraining/slbig128 SLBIG192=experiments/human-pretraining/slbig192 LONG3=experiments/endgame-technique/LONG3 LONG128=experiments/width/LONG128 LONG192=experiments/width/LONG192 --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_width_long.json 2>&1 | grep -v OMP
echo "=== WIDTH LONG all done $(date +%H:%M:%S)"
