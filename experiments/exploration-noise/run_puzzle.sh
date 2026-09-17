#!/bin/zsh
# Arm PUZNOISE (owner, 2026-09-17: "can we do test 2 in parallel?"). Mechanism test for
# the flattening seen in NOISE: noise on the one-move puzzle boards only (a random miss
# is unambiguously negative there), 120 updates from the CONTROL checkpoint, FIN3 layout.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/PUZNOISE
echo "=== PUZNOISE start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/CONTROL max_updates=120 \
  finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 explore_epsilon=0.25 explore_boards=puzzle 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== PUZNOISE all done $(date +%H:%M:%S)"
