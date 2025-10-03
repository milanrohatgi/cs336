import submitit
from tokenize_data import run_job

if __name__ == "__main__":
    total_files = 5000
    input_dir = "/data/c-mrohatgi/filtered_cc_2"          
    output_dir = "/data/c-mrohatgi/tokenized_chunks_2"

    executor = submitit.AutoExecutor(folder="slurm_logs/tokenize")
    executor.update_parameters(
        slurm_array_parallelism=12,
        timeout_min=20,
        mem_gb=2,
        cpus_per_task=1,
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
        slurm_account="student",
    )

    futures = []
    with executor.batch():
        for file_index in range(total_files):
            futures.append(
                executor.submit(run_job, file_index, input_dir, output_dir)
            )

    print(f"Submitted {len(futures)} tokenization jobs.")
