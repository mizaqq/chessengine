#!/bin/zsh
# Step 0 of the model-free plan (owner, 2026-09-17: "go with 0 but limit it to 35
# minutes, slightly less games"). Checkpoint win matrix, training order, 30 games per
# colour per pairing, both sides sampling. M34 added later when it finishes.
cd "$(dirname "$0")/../.."
echo "=== matrix start $(date +%H:%M:%S)"
.venv/bin/python -m scripts.eval_matrix \
  SL=experiments/human-pretraining/sl \
  PPO_SL=experiments/human-pretraining/ppo_from_sl \
  FIN3=experiments/finishing-curriculum/FIN3 \
  CLEAN=experiments/exploration-noise/CLEAN \
  GUIDE=experiments/exploration-noise/GUIDE \
  CLEAN_GUIDE=experiments/exploration-noise/CLEAN_GUIDE \
  --games 30 --seed 0 --workers 1 --out experiments/win-matrix/matrix.json 2>&1 | grep -v OMP
echo "=== matrix done $(date +%H:%M:%S)"
