#!/bin/zsh
# Arm CLEAN (owner, 2026-09-17: "isnt the fin3 start already polluted with previous
# evaluation using material also?"). Outcome-only reward from the pure SL checkpoint
# (m1 0.17, m2 0.13), FIN3 layout, 600 updates, so it reads against the shaped
# ppo_from_sl run at 300 and 600. Then games, finishes, probe.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise/CLEAN
echo "=== CLEAN start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/human-pretraining/sl max_updates=600 \
  finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 2>&1 | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== CLEAN all done $(date +%H:%M:%S)"
