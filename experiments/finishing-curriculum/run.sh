#!/bin/zsh
# Finishing curriculum (owner, 2026-09-17: "finish curriculum first to address the
# weaknesses"). 120 updates (standard budget) from the SL+PPO 600 checkpoint with the
# default layout 4 puzzle / 4 finishing / 2 mid-game / 2 opening boards, then 40
# sampled games and the held-out finishing evaluation. Waits for the re-harvested
# own-game mate-in-one set, which the periodic `selfplay` evaluation reads.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/finishing-curriculum
until grep -q "wrote" $OUT/harvest.log 2>/dev/null; do sleep 20; done
echo "=== start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT/FIN --set init_from=experiments/human-pretraining/ppo_from_sl max_updates=120 2>&1 | grep -E "done in|Traceback|Error" || true
$PY -m scripts.eval_games $OUT/FIN --games 40 --seed 0 --out $OUT/FIN/games.json 2>&1 | tail -1
$PY -m scripts.eval_finishes $OUT/FIN --out $OUT/FIN/finishes.json 2>&1 | grep finish_rate
echo "=== all done $(date +%H:%M:%S)"
