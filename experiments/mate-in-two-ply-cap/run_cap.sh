#!/bin/zsh
# Owner (2026-09-16): "i want the other improvement with ply cap" -> ply cap alone,
# mate-in-one puzzles only, otherwise the default config. Runs after BOTH.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M1='[{file: data/puzzles/mate_in_1_train.csv, weight: 1.0}]'
until grep -q "all arms done" experiments/mate-in-two-ply-cap/run_arms.log; do sleep 30; done
echo "=== CAP start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out experiments/mate-in-two-ply-cap/CAP --set "puzzle_train_files=$M1" max_plies=120 2>&1 | grep -E "done in|Traceback|Error" || true
$PY -m scripts.eval_games experiments/mate-in-two-ply-cap/CAP --games 40 --seed 0 --out experiments/mate-in-two-ply-cap/CAP/games.json 2>&1 | tail -1
echo "=== CAP end $(date +%H:%M:%S)"
