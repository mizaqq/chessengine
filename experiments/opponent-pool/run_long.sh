#!/bin/zsh
# Opponent-pool long run (owner, 2026-09-17: "the next run we test should be a long one
# so the evaluation happens in the morning"): named exception to the 120-update budget.
# 800 updates from M34 (seed 42), default config = 16 boards (4 puzzle / 2 finishing /
# 5 mid-game / 5 opening), pool = SL prior + last 4 snapshots every 50 updates on 5 game
# boards, guidance 0.25 on curriculum boards, prior_score every 50 updates. Then games,
# finishes, probe and the win matrix (prior, M34, CLEAN_GUIDE as the out-of-pool control).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/opponent-pool/POOL_LONG
echo "=== POOL_LONG start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/exploration-noise/M34 max_updates=800 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
echo "=== training done $(date +%H:%M:%S)"
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
$PY -m scripts.eval_matrix SL=experiments/human-pretraining/sl CLEAN_GUIDE=experiments/exploration-noise/CLEAN_GUIDE \
  M34=experiments/exploration-noise/M34 POOL_LONG=$OUT --games 30 --seed 3 --workers 1 --out experiments/win-matrix/matrix_pool_long.json 2>&1 | grep -v OMP
echo "=== POOL_LONG all done $(date +%H:%M:%S)"
