#!/bin/zsh
# Arm CURSIL_LONG (owner, 2026-09-18: "continue"): 360 more updates from CURSIL3 with the
# honest curriculum and SIL (named exception to the 120-update budget). Levels and clean
# success per set, held-out level 3, finishing, games. TECH2 / CURSIL3 are the controls.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/CURSIL_LONG
mkdir -p $OUT
echo "=== CURSIL_LONG start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/CURSIL3 max_updates=360 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== CURSIL_LONG all done $(date +%H:%M:%S)"
