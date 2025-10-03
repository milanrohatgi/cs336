#!/bin/bash
#SBATCH --job-name=naive_ddp_training
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --gpus=2        
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --output=naive_ddp_%j.out
#SBATCH --error=naive_ddp_%j.err

uv run python ddp_naive.py \
    --world_size=2 \
    --batch_size=64 \
    --context_length=256 \
    --d_model=768 \
    --num_layers=4 \
    --num_heads=12 \
    --max_iters=10 \
    --verify \
    --seed=42
