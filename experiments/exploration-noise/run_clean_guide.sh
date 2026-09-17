#!/bin/zsh
# Arm CLEAN_GUIDE (owner, 2026-09-17: "okay run guide on the clean checkpoint").
# Label-guided exploration on the unpolluted base: from CLEAN (SL -> 600 outcome-only
# updates; m1 0.647, m2 0.565), 420 guided updates so the total (1,020) matches the
# GUIDE_LONG lineage (m1 0.734, m2 0.630) for a like-for-like reading. FIN3 layout,
# two finishing boards back to the opening at update 320 (same 100-update tail).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/CLEAN_GUIDE
echo "=== CLEAN_GUIDE start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/CLEAN max_updates=420 \
  finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 guide_epsilon=0.25 \
  'layout_schedule=[{from_update: 320, finish_boards: 4, midgame_boards: 2}]' 2>&1 \
  | grep -E "layout|done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== CLEAN_GUIDE all done $(date +%H:%M:%S)"
