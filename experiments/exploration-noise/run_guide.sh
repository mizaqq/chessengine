#!/bin/zsh
# Arm GUIDE (owner, 2026-09-17: "okay, agreed on it"). Label-guided exploration: on
# puzzle and finishing boards the behaviour policy plays the demonstrator's move
# (puzzle key move / human line / rules mate) with probability 0.25. 120 updates from
# the CONTROL checkpoint (m1 0.653, m2 0.571), FIN3 layout. Then games, finishes, probe.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/GUIDE
echo "=== GUIDE start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/CONTROL max_updates=120 \
  finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== GUIDE all done $(date +%H:%M:%S)"
