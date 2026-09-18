#!/bin/zsh
# Arm CURSIL (owner, 2026-09-18: "lets continue with self imitation and start-state
# curriculum"). Same start (POOL_LONG), budget (120 updates) and layout as TECH2, plus the
# technique level curriculum (levels 0-3 per set, advance at 0.6 over 30) and self-imitation
# (4 x 256 per update, weight 0.1, value 0.01, buffer 50k). TECH2 is the control.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/CURSIL
BASE=experiments/opponent-pool/POOL_LONG
mkdir -p $OUT
echo "=== CURSIL start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$BASE max_updates=120 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== CURSIL all done $(date +%H:%M:%S)"
