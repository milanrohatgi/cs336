#!/bin/bash
#SBATCH --job-name=ddp_bucket_benchmark
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --gpus=2
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --output=ddp_bucket_benchmark_%j.out
#SBATCH --error=ddp_bucket_benchmark_%j.err

uv run --active -- python ddp_benchmark_overlap.py \
    --world_size 2 \
    --batch_size 8 \
    --context_length 512 \
    --d_model 1600 \
    --d_ff 6400 \
    --num_layers 48 \
    --num_heads 25 \
    --bucket_size_mb 1 \
    --use_bf16 