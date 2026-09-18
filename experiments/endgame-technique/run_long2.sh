#!/bin/zsh
# LONG2 (owner, 2026-09-18: "some kind of long run after this"; LR "option 1" = floor at
# 1e-4, no decay). Consolidation run on the real layout from LAMBDA: 600 updates, all four
# technique sets on the level ladder (levels persist via curriculum.json from now on),
# pool, SIL, guidance 0.25, gae_lambda 1.0, min_lr 1e-4. Named exception to the budget.
# Then games, finishes, probe, win matrix vs SL / M34 / POOL_LONG / CURSIL_LONG.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/LONG2
mkdir -p $OUT
echo "=== LONG2 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/LAMBDA max_updates=600 guide_epsilon=0.25 technique_curriculum=levels technique_level_start=0 gae_lambda=1.0 min_lr=0.0001 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl M34=experiments/exploration-noise/M34 POOL_LONG=experiments/opponent-pool/POOL_LONG \
  CURSIL_LONG=experiments/endgame-technique/CURSIL_LONG LONG2=$OUT --games 30 --seed 4 --workers 1 --out experiments/win-matrix/matrix_long2.json 2>&1 | grep -v OMP
echo "=== LONG2 all done $(date +%H:%M:%S)"
