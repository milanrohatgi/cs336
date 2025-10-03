# submit_cc_filter.py

import math, pathlib, submitit
from tqdm import tqdm

def run_job(task_id: int, cpus_per_task: int, output_dir: str):
    import os, concurrent.futures
    from pathlib import Path
    from cs336_data.filter_data import process_single_wet_file

    wet_dir = Path("/data/CC")
    all_files = sorted(str(p) for p in wet_dir.glob("CC*.warc.wet.gz"))

    start = task_id * cpus_per_task
    end = min(start + cpus_per_task, len(all_files))
    group = all_files[start:end]

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [
            exe.submit(process_single_wet_file, fp, out_dir)
            for fp in group
        ]
        return [f.result() for f in concurrent.futures.as_completed(futures)]

if __name__ == "__main__":
    cpus_per_task = 4
    wet_dir = pathlib.Path("/data/CC")
    total_files = 5000
    num_jobs = math.ceil(total_files / cpus_per_task)
    output_dir = "/data/c-mrohatgi/filtered_cc_2"

    executor = submitit.AutoExecutor(folder="slurm_logs")
    executor.update_parameters(
        slurm_array_parallelism = 8,
        timeout_min = 15,
        mem_gb = 2,
        cpus_per_task= cpus_per_task,
        slurm_account= "student",
        slurm_partition = "a4-cpu",
        slurm_qos = "a4-cpu-qos",
    )

    futures = []
    with executor.batch():
        for task_id in range(num_jobs):
            futures.append(
                executor.submit(run_job, task_id, cpus_per_task, output_dir)
            )

    for result in tqdm(submitit.helpers.as_completed(futures), total=len(futures)):
        print("chunk done:", result)
