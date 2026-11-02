#!/usr/bin/env bash

output_root=${1:-outputs/sensitivity}
seed=${2:-0}
sweep_root="$output_root/eshards-non-shuffled"

run_trial() {
    local parameter=$1 value=$2 override=$3
    local dest="$sweep_root/$parameter/$value/seed=$seed"
    python experiments/train.py --config-file config/eshards_non_shuffled.yaml --dest "$dest" --training.seed "$seed" "--$override" "$value"
    python experiments/attack.py --model-ckpt-path "$dest/best_val_rank.ckpt"
    python experiments/plot_performance.py --run-dir "$dest"
}

for value in 0.00024860141101564785 0.000372902116523471775 0.000621503527539119625 0.00074580423304694355; do
    run_trial base_lr "$value" training.base_lr
done

for value in 64 192 256 320; do
    run_trial embedding_dim "$value" model.embedding_dim
done

for value in 1 3 4 5; do
    run_trial trunk_blocks "$value" model.trunk_blocks
done

for value in 28 50 140 200; do
    run_trial patch_size "$value" model.patch_size
done
