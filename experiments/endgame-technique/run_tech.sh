#!/bin/zsh
# Arm TECH (owner, 2026-09-18: endgame mates from Lichess first, bare-king technique
# generated). Waits for the endgame_mate puzzle files, measures POOL_LONG baselines
# (technique held-out per set, endgame-mate top-1), then 120 updates from POOL_LONG with
# the default config (16 boards, 2 technique boards, five-source puzzle mix, pool,
# guidance 0.25), then games, finishes, probe.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/TECH
BASE=experiments/opponent-pool/POOL_LONG
until [ -f data/puzzles/endgame_mate_eval.csv ] && [ -f data/puzzles/endgame_mate_train.csv ]; do sleep 20; done
sleep 5
mkdir -p $OUT
echo "=== TECH start $(date +%H:%M:%S)"
$PY -m scripts.eval_checkpoint_puzzles $BASE --eval data/puzzles/endgame_mate_eval.csv --out $OUT/baseline_end.json 2>&1 | grep -E "top1" || true
$PY - <<'PYEOF' 2>&1 | grep -v OMP | tee $OUT/baseline_technique.txt
import json
from src.model.checkpoints import load_models
from src.eval.technique import evaluate_technique
w, b, o, _ = load_models("experiments/opponent-pool/POOL_LONG")
out = evaluate_technique(w, b, ["Q", "R", "RR", "QR"], {"Q": 40, "R": 60, "RR": 40, "QR": 40}, games=50, oriented=o, seed=42 + 5000)
print("baseline technique", json.dumps(out))
PYEOF
$PY -m scripts.run_experiment --out $OUT --set init_from=$BASE max_updates=120 guide_epsilon=0.25 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
$PY -m scripts.eval_finishes $OUT --out $OUT/finishes.json 2>&1 | grep finish_rate
$PY -m scripts.probe_finish_values $OUT 2>&1 | grep -v OMP | tee $OUT/probe.txt
echo "=== TECH all done $(date +%H:%M:%S)"
