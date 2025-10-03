#!/bin/bash
#SBATCH --partition=a5-batch           # a5‐batch GPU queue
#SBATCH --qos=a5-batch-qos
#SBATCH --gres=gpu:2                   # request 2 GPUs
#SBATCH --cpus-per-task=8              # up to 8 CPU cores
#SBATCH --mem=100G                     # 100 GB RAM
#SBATCH --time=04:00:00              
#SBATCH --output=%x_%j_full.out
#SBATCH --error=%x_%j_full.err              

NUM_EXAMPLES=1355

OUTPUT_DIR=/home/c-mrohatgi/assignment5-alignment/cs336_alignment/summary_of_sft
uv run python run_sft.py \
    --num_examples $NUM_EXAMPLES \
    --output_dir $OUTPUT_DIR
