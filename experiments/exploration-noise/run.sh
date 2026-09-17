#!/bin/zsh
# Exploration noise (owner, 2026-09-17: "we can add the noise in training to explore
# more moves"). Two arms of 120 updates from the FIN3 checkpoint (shaping off, m1 0.638),
# FIN3 layout 4 puzzle / 6 finishing / 2 mid-game, threshold 0.4:
#   NOISE   eps 0.25, alpha 0.3 on puzzle + finishing boards
#   CONTROL no noise (isolates the effect from plain continued training)
# then 40 games, held-out finishes and the probe with the mate-mass histogram.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/exploration-noise
INIT=experiments/finishing-curriculum/FIN3
echo "=== start $(date +%H:%M:%S)"
$PY -m scripts.probe_finish_values $INIT 2>&1 | grep -v OMP > $OUT/baseline_probe.txt
for ARM in NOISE CONTROL; do
  if [ $ARM = NOISE ]; then EPS=0.25; else EPS=0.0; fi
  echo "=== $ARM start $(date +%H:%M:%S)"
  $PY -m scripts.run_experiment --out $OUT/$ARM --set init_from=$INIT max_updates=120 \
    finish_boards=6 midgame_boards=2 finish_advance_rate=0.4 explore_epsilon=$EPS 2>&1 \
    | grep -E "done in|Traceback|Error" | cut -c1-400 || true
  $PY -m scripts.eval_games $OUT/$ARM --games 40 --seed 0 --out $OUT/$ARM/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
  $PY -m scripts.eval_finishes $OUT/$ARM --out $OUT/$ARM/finishes.json 2>&1 | grep finish_rate
  $PY -m scripts.probe_finish_values $OUT/$ARM 2>&1 | grep -v OMP | tee $OUT/$ARM/probe.txt
done
echo "=== all done $(date +%H:%M:%S)"
