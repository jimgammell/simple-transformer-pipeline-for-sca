#!/usr/bin/env bash


output_root=${1:-outputs}


for seed in 0 1 2 3 4; do
    dest="$output_root/ascadv1-fixed/seed=$seed"
    python experiments/train.py --config-file config/ascadv1_fixed.yaml --dest "$dest" --training.seed "$seed"
    python experiments/attack.py --model-ckpt-path "$dest/best_val_rank.ckpt"
    python experiments/attribute.py --ckpt-path "$dest/best_val_rank.ckpt" --attr-methods input-x-gradient
    python experiments/plot_performance.py --run-dir "$dest"
done


