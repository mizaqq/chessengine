#!/bin/zsh
# Arm M34 second seed (owner, 2026-09-17: "run for other seed"). Same as run_m34.sh
# (m1-m4 mix, guidance 0.25, 120 updates from CLEAN_GUIDE) with seed=1, then the win
# matrix against the SL prior and CLEAN_GUIDE. Tests whether M34's par with the prior
# repeats.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/M34_s1
echo "=== M34_s1 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/CLEAN_GUIDE max_updates=120 guide_epsilon=0.25 seed=1 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-600 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl CLEAN_GUIDE=experiments/exploration-noise/CLEAN_GUIDE M34_s1=$OUT \
  --games 30 --seed 2 --workers 1 --out experiments/win-matrix/matrix_m34_s1.json 2>&1 | grep -v OMP
echo "=== M34_s1 all done $(date +%H:%M:%S)"
