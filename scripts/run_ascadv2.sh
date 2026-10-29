#!/usr/bin/env bash

output_root=${1:-outputs}

for seed in 0 1 2 3 4; do
    python experiments/train.py --config-file config/ascadv2.yaml --dest "$output_root/ascadv2/seed=$seed" --training.seed "$seed"
done