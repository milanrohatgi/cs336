#!/bin/bash
#SBATCH --job-name=ddp_benchmark
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --gpus=2
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --output=ddp_benchmark_%j.out
#SBATCH --error=ddp_benchmark_%j.err

nsys profile \
    --output=naive_bs8_ctx512 \
    --trace=cuda,nvtx \
    --stats=true \
    uv run python ddp_naive_benchmark.py \
        --d_model=1600 \
        --d_ff=6400 \
        --num_layers=48 \
        --num_heads=25 \
        --context_length=512 \
        --batch_size=8 \
        --use_bf16
