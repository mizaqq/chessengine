#!/bin/zsh
# Arm TECH2 (owner, 2026-09-18: "make puzzle 3, technique 4, mid game 5, opening 4").
# Same recipe as TECH from the same POOL_LONG checkpoint (clean layout comparison), 120
# updates, default config now 3 puzzle / 4 technique / 5 mid-game / 4 opening, pool on
# the 5 mid-game boards, guidance 0.25; then games, finishes, probe. Baselines are TECH's.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/TECH2
BASE=experiments/opponent-pool/POOL_LONG
mkdir -p $OUT
echo "=== TECH2 start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=$BASE max_updates=120 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== TECH2 all done $(date +%H:%M:%S)"
