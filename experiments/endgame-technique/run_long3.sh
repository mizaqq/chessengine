#!/bin/zsh
# LONG3 (owner, 2026-09-19: "after this we can run a long run to be tested in the
# morning"). LONG2 + 600 updates with the finishing slot mixed: 4 boards drawing technique
# (all four sets, level ladder, levels persisted) and human finishing starts 0.5 / 0.5.
# Everything else as LONG2 (lambda 1, LR floor 1e-4, pool, SIL, guidance 0.25). Then games,
# finishes, probe, calibration, win matrix vs SL / LONG2.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/LONG3
mkdir -p $OUT
echo "=== LONG3 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/LONG2 max_updates=600 guide_epsilon=0.25 technique_curriculum=levels finish_source=mixed min_lr=0.0001 2>&1 \
  | grep -E "done in|Traceback|Error|curriculum" | cut -c1-300 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
$PY -m scripts.calibrate_finishes $OUT --games 100 --out experiments/finishing-calibration/long3.json 2>&1 | grep -v OMP
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl LONG2=experiments/endgame-technique/LONG2 LONG3=$OUT --games 30 --seed 5 --workers 1 --out experiments/win-matrix/matrix_long3.json 2>&1 | grep -v OMP
echo "=== LONG3 all done $(date +%H:%M:%S)"
