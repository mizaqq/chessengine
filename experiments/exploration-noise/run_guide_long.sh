#!/bin/zsh
# Arm GUIDE_LONG (owner, 2026-09-17: "We can do longer run"). 300 more updates from the
# GUIDE checkpoint (m1 0.685, m2 0.605) with label-guided exploration on; named
# exception to the standard budget, to see where guidance saturates. FIN3 layout,
# switch two finishing boards back to the opening at update 200 as in FIN2/FIN3.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/GUIDE_LONG
echo "=== GUIDE_LONG start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/GUIDE max_updates=300 \
  finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 guide_epsilon=0.25 \
  'layout_schedule=[{from_update: 200, finish_boards: 4, midgame_boards: 2}]' 2>&1 \
  | grep -E "layout|done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== GUIDE_LONG all done $(date +%H:%M:%S)"
