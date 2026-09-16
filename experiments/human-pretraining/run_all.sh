#!/bin/zsh
# Change human-pretraining (2026-09-16 night). Chain: wait for the data build ->
# supervised pretraining (3 epochs) -> evaluate the SL checkpoint -> PPO 120 updates
# from it with 4 mid-game boards -> evaluate. Owner's decisions: Elo >= 1500, 300k
# positions, 4 of 8 game boards mid-game, no mate-in-three.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/human-pretraining
until grep -q "human_train.npz" $OUT/prepare_games.log; do sleep 20; done
echo "=== SL start $(date +%H:%M:%S)"
$PY -m scripts.pretrain_human --out $OUT/sl --epochs 3 --eval-every 200 --eval-limit 2048 2>&1 | grep -E "^\{|final|saved|Traceback|Error"
echo "=== SL eval $(date +%H:%M:%S)"
$PY -m scripts.eval_checkpoint_puzzles $OUT/sl --out $OUT/sl/puzzles_m1.json 2>&1 | tail -1
$PY -m scripts.eval_checkpoint_puzzles $OUT/sl --eval data/puzzles/mate_in_2_eval.csv --out $OUT/sl/puzzles_m2.json 2>&1 | tail -1
$PY -m scripts.eval_games $OUT/sl --games 40 --seed 0 --out $OUT/sl/games.json 2>&1 | tail -1
echo "=== PPO-from-SL start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT/ppo_from_sl --set init_from=$OUT/sl midgame_boards=4 game_start_file=data/games/midgame_starts.csv 2>&1 | grep -E "done in|Traceback|Error" || true
$PY -m scripts.eval_games $OUT/ppo_from_sl --games 40 --seed 0 --out $OUT/ppo_from_sl/games.json 2>&1 | tail -1
echo "=== all done $(date +%H:%M:%S)"
