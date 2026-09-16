#!/bin/zsh
# Arms for change mate-in-two-ply-cap (2026-09-16): default config (PPO lr 1e-4,
# 64 plies, 120 updates, 4 puzzle boards), seed 42. Only the named keys differ.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M1='[{file: data/puzzles/mate_in_1_train.csv, weight: 1.0}]'
M12='[{file: data/puzzles/mate_in_1_train.csv, weight: 0.5}, {file: data/puzzles/mate_in_2_train.csv, weight: 0.5}]'
run() { name=$1; shift
  echo "=== $name start $(date +%H:%M:%S)"
  $PY -m scripts.run_experiment --out experiments/mate-in-two-ply-cap/$name --set "$@" 2>&1 | grep -E "done in|Traceback|Error" || true
  $PY -m scripts.eval_games experiments/mate-in-two-ply-cap/$name --games 40 --seed 0 --out experiments/mate-in-two-ply-cap/$name/games.json 2>&1 | tail -1
  echo "=== $name end $(date +%H:%M:%S)"
}
# Owner (2026-09-16): "no need to run 4 arms". Only the new default runs; the
# baseline is experiments/ppo-gae/C2_ppo_lr1e-4 read at updates 120/150.
run BOTH "puzzle_train_files=$M12" max_plies=120
echo "=== all arms done $(date +%H:%M:%S)"
