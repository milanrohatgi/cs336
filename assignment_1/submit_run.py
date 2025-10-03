import os
from pathlib import Path

run_name = "owt_test_final"

job_dir = Path("sbatch_jobs")
job_dir.mkdir(exist_ok=True)

job_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output={run_name}%j.out
#SBATCH --error={run_name}%j.err

uv run python train.py \\
  --batch_size 256 \\
  --context_length 256 \\
  --num_layers 8 \\
  --d_model 768 \\
  --num_heads 12 \\
  --d_ff 2048 \\
  --rope_theta 10000.0 \\
  --learning_rate .002 \\
  --max_iters 75000 \\
  --vocab_size 32000 \\
  --train_path /data/c-mrohatgi/owt-train-tokenized.npy \\
  --val_path /data/c-mrohatgi/owt-valid-tokenized.npy \\
  --checkpoint_path /data/c-mrohatgi/checkpoints/{run_name}.pt \\
  --warmup_iters 1000 \\
  --min_lr .0002 \\
  --run_name {run_name}
"""

job_path = job_dir / f"{run_name}.sh"
with open(job_path, "w") as f:
    f.write(job_script)

os.system(f"sbatch {job_path}")
