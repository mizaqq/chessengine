#!/bin/zsh
# Elo ladders (change elo-eval, 2026-09-22): LONG256X (best), its prior, the old 128 prior and
# LONG3 (old lineage's best), sampled moves, 20 games per colour per level, 50 ms per engine move.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for E in LONG256X=experiments/width/LONG256X SLHI2_256=experiments/human-pretraining/slhi2_256 SL=experiments/human-pretraining/sl LONG3=experiments/endgame-technique/LONG3; do
  NAME=${E%%=*}; DIR=${E#*=}
  echo "=== ELO $NAME start $(date +%H:%M:%S)"
  OMP_NUM_THREADS=4 $PY -m scripts.eval_elo $DIR --games 20 --movetime 0.05 --out $DIR/elo.json 2>&1 | grep -v OMP
  cp $DIR/elo.json experiments/elo/$NAME.json
done
echo "=== ELO all done $(date +%H:%M:%S)"
