#!/usr/bin/env bash
# Best-config SAC campaign: net 256, lr 1e-3, batch 128, 1M steps, 3 seeds.
set -u
cd /home/andreipc/Inno-RL
OUT=artifacts/best_config
mkdir -p "$OUT"
PY=.venv/bin/python
run_one () {
  local seed=$1
  local tag="best${seed}"
  echo "[$tag] launch (seed $seed)"
  $PY -u run_experiment.py \
    --algo sac --seed "$seed" --steps 1000000 \
    --net 256,256 --lr 1e-3 --batch 128 \
    --eval-episodes 30 \
    --checkpoint-every 500000 --checkpoint-eval-episodes 10 \
    --threads 4 --tag "$tag" > "$OUT/${tag}.log" 2>&1
  echo "[$tag] exit $?"
}
run_one 50 &
run_one 51 &
run_one 52 &
wait
echo "best-config campaign done"
