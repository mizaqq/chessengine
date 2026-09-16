#!/bin/zsh
# Owner (2026-09-16): "run with mate-in-2 and ply limit with a longer run". Continue
# from BOTH (120 updates) for 480 more -> 600 updates total, 38,400 plies per board.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q "all arms done" experiments/mate-in-two-ply-cap/run_arms.log; do sleep 30; done
echo "=== LONG start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out experiments/mate-in-two-ply-cap/LONG --set init_from=experiments/mate-in-two-ply-cap/BOTH max_updates=480 2>&1 | grep -E "done in|Traceback|Error" || true
$PY -m scripts.eval_games experiments/mate-in-two-ply-cap/LONG --games 40 --seed 0 --out experiments/mate-in-two-ply-cap/LONG/games.json 2>&1 | tail -1
echo "=== LONG end $(date +%H:%M:%S)"
