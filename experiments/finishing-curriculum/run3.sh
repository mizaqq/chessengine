#!/bin/zsh
# Arm FIN3 (owner, 2026-09-17: "fallback on rewarding the model basing on material as
# we have such pretraining? The model thus can learn the real reward is the win").
# Same as FIN2 (4 puzzle / 6 finishing / 2 mid-game, threshold 0.4, 300 updates, switch
# to 4 finishing / 2 mid-game / 2 opening at 200) with material shaping OFF: the value
# head had become a material counter (-2.6 on own positions with a mate available),
# and the pretrained prior supplies the density shaping was there for.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/finishing-curriculum
echo "=== start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT/FIN3 --set init_from=experiments/human-pretraining/ppo_from_sl \
  shaping=none max_updates=300 finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 \
  'layout_schedule=[{from_update: 200, finish_boards: 4, midgame_boards: 2}]' 2>&1 \
  | grep -E "layout|done in|Traceback|Error" || true
$PY -m scripts.eval_games $OUT/FIN3 --games 40 --seed 0 --out $OUT/FIN3/games.json 2>&1 | tail -1
$PY -m scripts.eval_finishes $OUT/FIN3 --out $OUT/FIN3/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT/FIN3 2>&1 | grep -v OMP
echo "=== all done $(date +%H:%M:%S)"
