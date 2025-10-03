#!/bin/bash
#SBATCH --partition=a5-batch           # a5‐batch GPU queue
#SBATCH --qos=a5-batch-qos
#SBATCH --gres=gpu:2                   # request 2 GPUs
#SBATCH --cpus-per-task=8              # up to 8 CPU cores
#SBATCH --mem=100G                     # 100 GB RAM
#SBATCH --time=04:00:00                # 4 hour wall clock
#SBATCH --output=%x_%j_full.out
#SBATCH --error=%x_%j_full.err              

OUTPUT_DIR=/home/c-mrohatgi/assignment5-alignment/cs336_alignment/ei_outputs
uv run python ei.py \
    --output_dir $OUTPUT_DIR
