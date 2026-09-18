#!/bin/zsh
# Arm QONLY (owner, 2026-09-18: "i would test 2", the queen-only technique layout, while
# wondering whether the architecture size is the limit). All four technique boards play
# queen v king (4x the queen's sample rate), level start 1 (where CURSIL_LONG left the
# queen; levels do not yet persist with the checkpoint), threshold 0.7 over 40 unchanged
# so only the sample rate changes. 120 updates from CURSIL_LONG. If the queen still stalls
# near 0.3-0.4 clean at level 1, the sample rate is not the limit.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=experiments/endgame-technique/QONLY
mkdir -p $OUT
echo "=== QONLY start $(date +%H:%M:%S)"
$PY -m scripts.run_experiment --out $OUT --set init_from=experiments/endgame-technique/CURSIL_LONG max_updates=120 guide_epsilon=0.25 "technique_sets=[Q]" technique_level_start=1 2>&1 \
  | grep -E "done in|Traceback|Error" | cut -c1-400 || true
$PY -m scripts.eval_games $OUT --games 40 --seed 0 --out $OUT/games.json 2>&1 | grep -E "mate_in_one_rate|decisive"
echo "=== QONLY all done $(date +%H:%M:%S)"
