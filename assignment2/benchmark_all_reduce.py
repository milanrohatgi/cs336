import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--size", type=str, default="1MB")
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--warmup_runs", type=int, default=5)
    return parser.parse_args()

def setup(rank, world_size, backend, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_tensor_size_bytes(size_str):
    sizes = {
        "1MB": 1024 * 1024 // 4,
        "10MB": 10 * 1024 * 1024 // 4,
        "100MB": 100 * 1024 * 1024 // 4,
        "1GB": 1024 * 1024 * 1024 // 4
    }
    return sizes[size_str]

def benchmark(rank, world_size, args, master_addr, master_port):
    setup(rank, world_size, args.backend, master_addr, master_port)
    
    tensor_size = get_tensor_size_bytes(args.size)
    
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
 
    torch.cuda.synchronize() if args.backend == "nccl" else None
    dist.barrier()
    
    for _ in range(args.warmup_runs):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    
    torch.cuda.synchronize() if args.backend == "nccl" else None
    dist.barrier()
    
    run_times = []
    
    for run in range(args.num_runs):
        dist.barrier()
        
        start_time = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        
        torch.cuda.synchronize() if args.backend == "nccl" else None
        
        end_time = time.time()
        run_time = (end_time - start_time) * 1000 
        run_times.append(run_time)
        
    if rank == 0:
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        print(f"- Mean time: {mean_time:.3f} ms")
        print(f"- Std dev: {std_time:.3f} ms")
    
    cleanup()

def main():
    args = parse_args()
    
    master_addr = "127.0.0.1"
    master_port = find_free_port()

    mp.spawn(
        benchmark,
        args=(args.num_processes, args, master_addr, master_port),
        nprocs=args.num_processes,
        join=True
    )


if __name__ == "__main__":
    main()
