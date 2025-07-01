#!/bin/bash

# This script runs the standalone training and evaluation process to establish a baseline.
# The parameters can be adjusted as needed.

python fbd_single_run.py \
    --experiment_name bloodmnist \
    --model_flag resnet18 \
    --epochs 10 \
    --lr 0.001 \
    --output_dir "fbd_single_run_results/bloodmnist_resnet18" \
    --cache_dir "cache"

echo "Standalone training run finished. Results are in fbd_single_run_results/bloodmnist_resnet18" 