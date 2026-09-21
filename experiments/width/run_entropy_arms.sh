#!/bin/zsh
# Entropy guard arms (owner, 2026-09-21: "run both as two arms"). LONG256 game entropy hit 0.198
# (lowest ever) with KL/clip up and m1 flat from update 350. ENT03: entropy_coef 0.03 (from
# 0.01). LR5: learning_rate 5e-5 with min_lr 5e-5 (from 1e-4 floor). Each LONG256 + 120 updates,
# everything else as LONG256. Then games, finishes, audit, matrix vs LONG256. GPU: outside sandbox.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
START=experiments/width/LONG256; PRIOR=experiments/human-pretraining/slhi256
COMMON="init_from=$START opponent_prior=$PRIOR max_updates=120 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed finish_depth_start=20 finish_depth_max=20"
for ARM in ENT03 LR5; do
  if [[ $ARM == ENT03 ]]; then EXTRA="min_lr=0.0001 entropy_coef=0.03"; else EXTRA="learning_rate=0.00005 min_lr=0.00005"; fi
  OUT=experiments/width/$ARM; mkdir -p $OUT
  echo "=== $ARM start $(date +%H:%M:%S) ($EXTRA)"
  $PY -m scripts.run_experiment --out $OUT --set ${=COMMON} ${=EXTRA} 2>&1 | grep -E "^device|done in|Traceback|Error" | cut -c1-300 || true
  $PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.blunder_audit $OUT self=$OUT --games 20 --seed 1 --out experiments/blunder-audit/$ARM.json 2>&1 | grep -v OMP
  echo "=== $ARM done $(date +%H:%M:%S)"
done
$PY -m scripts.eval_matrix LONG256=$START ENT03=experiments/width/ENT03 LR5=experiments/width/LR5 --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_entropy_arms.json 2>&1 | grep -v OMP
echo "=== ENTROPY ARMS all done $(date +%H:%M:%S)"
