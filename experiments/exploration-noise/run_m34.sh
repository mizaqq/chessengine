#!/bin/zsh
# Arm M34 (owner, 2026-09-17: "add more puzzles than mate in 2"; "i like the
# recommendation"). Mate-in-3 and mate-in-4 puzzle rungs with the solution line as the
# demonstrator, default layout 4 puzzle / 4 finishing / 2 mid-game / 2 opening, guidance
# 0.25, 120 updates from CLEAN_GUIDE (m1 0.752, m2 0.638). Waits for the puzzle files.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/M34
until [ -f data/puzzles/mate_in_3_eval.csv ] && [ -f data/puzzles/mate_in_4_eval.csv ]; do sleep 20; done
sleep 10
echo "=== M34 start $(date +%H:%M:%S)"
$PY -m scripts.eval_checkpoint_puzzles experiments/exploration-noise/CLEAN_GUIDE --eval data/puzzles/mate_in_3_eval.csv --out $OUT/baseline_m3.json 2>&1 | grep -E "top1" || true
$PY -m scripts.eval_checkpoint_puzzles experiments/exploration-noise/CLEAN_GUIDE --eval data/puzzles/mate_in_4_eval.csv --out $OUT/baseline_m4.json 2>&1 | grep -E "top1" || true
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/CLEAN_GUIDE max_updates=120 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-600 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== M34 all done $(date +%H:%M:%S)"
