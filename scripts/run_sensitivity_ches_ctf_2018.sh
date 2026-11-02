#!/usr/bin/env bash

output_root=${1:-outputs/sensitivity}
seed=${2:-0}
sweep_root="$output_root/ches-ctf-2018"

run_trial() {
    local parameter=$1 value=$2 override=$3
    local dest="$sweep_root/$parameter/$value/seed=$seed"
    python experiments/train.py --config-file config/ches_ctf_2018.yaml --dest "$dest" --training.seed "$seed" "--$override" "$value"
    python experiments/attack.py --model-ckpt-path "$dest/best_val_rank.ckpt"
    python experiments/plot_performance.py --run-dir "$dest"
}

for value in 0.00225 0.003375 0.005625 0.00675; do
    run_trial base_lr "$value" training.base_lr
done

for value in 128 192 320 384; do
    run_trial embedding_dim "$value" model.embedding_dim
done

for value in 2 3 5 6; do
    run_trial trunk_blocks "$value" model.trunk_blocks
done

for value in 6500 10000 32500 65000; do
    run_trial patch_size "$value" model.patch_size
done
