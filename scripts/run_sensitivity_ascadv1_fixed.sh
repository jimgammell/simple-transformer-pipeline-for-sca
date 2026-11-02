#!/usr/bin/env bash

output_root=${1:-outputs/sensitivity}
seed=${2:-0}
sweep_root="$output_root/ascadv1-fixed"

run_trial() {
    local parameter=$1 value=$2 override=$3
    local dest="$sweep_root/$parameter/$value/seed=$seed"
    python experiments/train.py --config-file config/ascadv1_fixed.yaml --dest "$dest" --training.seed "$seed" "--$override" "$value"
    python experiments/attack.py --model-ckpt-path "$dest/best_val_rank.ckpt"
    python experiments/plot_performance.py --run-dir "$dest"
}

for value in 0.00230 0.003450 0.005750 0.00690; do
    run_trial base_lr "$value" training.base_lr
done

for value in 256 384 640 768; do
    run_trial embedding_dim "$value" model.embedding_dim
done

for value in 4 6 10 12; do
    run_trial trunk_blocks "$value" model.trunk_blocks
done

for value in 500 800 1250 2000; do
    run_trial patch_size "$value" model.patch_size
done
