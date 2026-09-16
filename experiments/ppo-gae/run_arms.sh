#!/bin/zsh
# Arms for change add-ppo-gae (2026-09-16): fresh network, seed 42, 12 boards
# (4 Lichess puzzle boards), 64-ply rollouts, 250 updates each; only the named
# keys differ. Run from the repo root: zsh experiments/ppo-gae/run_arms.sh
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
COMMON=(steps_per_update=64 max_updates=250 log_interval=10)
run() { name=$1; shift
  echo "=== $name start $(date +%H:%M:%S)"
  $PY -m scripts.run_experiment --out experiments/ppo-gae/$name --set $COMMON "$@" 2>&1 | grep -E "done in|Traceback|Error" || true
  $PY -m scripts.eval_games experiments/ppo-gae/$name --games 40 --seed 0 --out experiments/ppo-gae/$name/games.json 2>&1 | tail -1
  echo "=== $name end $(date +%H:%M:%S)"
}
run A_a2c_lam1        algorithm=a2c gae_lambda=1.0
run B_a2c_lam095      algorithm=a2c gae_lambda=0.95
run C_ppo_lr3e-4      algorithm=ppo learning_rate=0.0003
run C2_ppo_lr1e-4     algorithm=ppo learning_rate=0.0001 min_lr=0.00003
echo "=== all arms done $(date +%H:%M:%S)"
