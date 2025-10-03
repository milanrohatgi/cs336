#!/usr/bin/env python3
import os
from pathlib import Path


backends_devices = [
    {"backend": "gloo", "device": "CPU"},
    {"backend": "nccl", "device": "GPU"}
]

data_sizes = ["1MB", "10MB", "100MB", "1GB"]
process_counts = [2, 4, 6]

job_dir = Path("sbatch_jobs")
job_dir.mkdir(exist_ok=True)

for config in backends_devices:
    backend = config["backend"]
    device = config["device"]
    for size in data_sizes:
        for procs in process_counts:
            gpus_needed = procs if backend == "nccl" else 0
            run_name = f"allreduce_{backend}_{device.lower()}_{size}_{procs}procs"
            
            job_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --time=00:05:00
#SBATCH --output={run_name}_%j.out
#SBATCH --error={run_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks={procs}
#SBATCH --cpus-per-task=2
"""

            if backend == "nccl":
                job_script += f"#SBATCH --gpus={gpus_needed}\n"
                
            job_script += f"""
uv run python benchmark_all_reduce.py \\
  --backend {backend} \\
  --size {size} \\
  --num_processes {procs} \\
  --num_runs 10 \\
  --warmup_runs 5
"""

            job_path = job_dir / f"{run_name}.sh"
            with open(job_path, "w") as f:
                f.write(job_script)


submit_script = """#!/bin/bash
for f in sbatch_jobs/*.sh; do
    sbatch $f
done
"""


