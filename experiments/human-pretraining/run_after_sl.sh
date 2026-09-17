#!/bin/zsh
# Owner (2026-09-16, 23:05): "after pretraining we can run longer PPO, lets make up to
# 600 updates". Waits for the SL checkpoint (run.json), evaluates it, then PPO 600
# updates from it with 4 mid-game boards, then 40 games.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/human-pretraining
until [ -f $OUT/sl/run.json ]; do sleep 30; done
echo "=== SL eval $(date +%H:%M:%S)"
$PY -m scripts.eval_checkpoint_puzzles $OUT/sl --out $OUT/sl/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT/sl --eval data/puzzles/mate_in_2_eval.csv --out $OUT/sl/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT/sl --games 40 --seed 0 --out $OUT/sl/games.json 2>&1 | tail -1
echo "=== PPO-from-SL start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT/ppo_from_sl --set init_from=$OUT/sl midgame_boards=4 game_start_file=data/games/midgame_starts.csv max_updates=600 2>&1 | grep -E "done in|Traceback|Error" || true
$PY -m scripts.eval_games $OUT/ppo_from_sl --games 40 --seed 0 --out $OUT/ppo_from_sl/games.json 2>&1 | tail -1
echo "=== all done $(date +%H:%M:%S)"
