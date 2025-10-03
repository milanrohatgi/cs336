import os
from pathlib import Path

context_lengths = [128, 256, 512, 1024]

model_configs = [
    {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"name": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
]


#model_configs = [{"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}]
bf16_options = [False]  
backwards_options = [False] 


job_dir = Path("sbatch_jobs")
job_dir.mkdir(exist_ok=True)


results_dir = Path("benchmark_results")
results_dir.mkdir(exist_ok=True)

for model in model_configs:
    for ctx in context_lengths:
        for use_bf16 in bf16_options:
            for backwards in backwards_options:
                precision_str = "bf16" if use_bf16 else "fp32"
                direction_str = "backward" if backwards else "forward"
                run_name = f"{model['name']}_{ctx}_{precision_str}_{direction_str}"
                
                job_script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --output={results_dir}/{run_name}_%j.out
#SBATCH --error={results_dir}/{run_name}_%j.err

uv run python benchmark.py \\
  --d_model {model['d_model']} \\
  --d_ff {model['d_ff']} \\
  --num_layers {model['num_layers']} \\
  --num_heads {model['num_heads']} \\
  --warmup_steps 5 \\
  --n_steps 10 \\
  --context_length {ctx} \\
  {"--backwards" if backwards else ""} \\
  {"--use_bf16" if use_bf16 else ""}
"""
                job_path = job_dir / f"{run_name}.sh"
                with open(job_path, "w") as f:
                    f.write(job_script)
                

