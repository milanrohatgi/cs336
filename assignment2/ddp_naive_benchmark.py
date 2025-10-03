import os
import argparse
import numpy as np
from timeit import default_timer as timer
import time
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda import nvtx


import cs336_basics.model as transformer
import cs336_basics.optimizer as optim
import cs336_basics.data as data

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def generate_random_batch(batch_size, context_length, vocab_size, device, seed):
    torch.manual_seed(seed)
    random_data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return random_data

def naive_ddp_worker(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(args.seed)

    model = transformer.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    if args.compile:
        model = torch.compile(model)
    
    optimizer = optim.AdamW(model.parameters())

    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    
    examples_per_rank = args.batch_size // world_size

    if rank == 0:
        print(f"Running {args.warmup_steps} warmup iterations...")
        
    for i in range(args.warmup_steps):
        batch_seed = args.seed + i
        minibatch = generate_random_batch(
            examples_per_rank, 
            args.context_length, 
            args.vocab_size, 
            device,
            seed=batch_seed + rank 
        )

        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(minibatch)
            loss = outputs.mean()
            loss.backward()

        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        '''
        #### FLATTENED ######
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size

        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, unflat_grad in zip(grads, unflat_grads):
            grad.copy_(unflat_grad)
        ####################
        '''
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    dist.barrier()

    iteration_times = []
    comm_times = []
    
    if rank == 0:
        print(f"Running {args.n_steps} benchmarking iterations...")

    for i in range(args.n_steps):
        nvtx.range_push(f"iter_{i}")
        batch_seed = args.seed + args.warmup_steps + i
        minibatch = generate_random_batch(
            examples_per_rank, 
            args.context_length, 
            args.vocab_size, 
            device,
            seed=batch_seed + rank 
        )

        torch.cuda.synchronize()
        dist.barrier()
        iter_start = time.perf_counter()

        nvtx.range_push("forward")
        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(minibatch)
            loss = outputs.mean()
            nvtx.range_pop()

        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()

        comm_start = time.perf_counter()
        
        nvtx.range_push("gradient_sync")
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        nvtx.range_pop()
        '''
        #### FLATTENED ######
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        torch.cuda.synchronize()
        comm_start = time.perf_counter()

        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size

        torch.cuda.synchronize()
        comm_end = time.perf_counter()
        comm_time = comm_end - comm_start

        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, unflat_grad in zip(grads, unflat_grads):
            grad.copy_(unflat_grad)
        ####################
        '''
        comm_end = time.perf_counter()
        comm_time = comm_end - comm_start

        nvtx.range_push("optimizer_step")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()
        nvtx.range_pop()

        torch.cuda.synchronize()
        dist.barrier()
        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start

        iteration_times.append(iter_time)
        comm_times.append(comm_time)

    if rank == 0:
        avg_iter_time = statistics.mean(iteration_times)
        avg_comm_time = statistics.mean(comm_times)
        comm_percentage = (avg_comm_time / avg_iter_time) * 100
        
        print("\n==== BENCHMARK RESULTS ====")
        print(f"Model: XL (d_model={args.d_model}, d_ff={args.d_ff}, layers={args.num_layers}, heads={args.num_heads})")
        print(f"Batch size: {args.batch_size}, Context length: {args.context_length}")
        print(f"Average time per training iteration: {avg_iter_time:.4f} seconds")
        print(f"Average time spent communicating gradients: {avg_comm_time:.4f} seconds ({comm_percentage:.2f}% of iteration time)")
        
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Benchmark Naive DDP Training")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1600)  
    parser.add_argument("--d_ff", type=int, default=6400)  
    parser.add_argument("--num_layers", type=int, default=48) 
    parser.add_argument("--num_heads", type=int, default=25) 
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--rope_theta", type=int, default=10000)

    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--world_size", type=int, default=2)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed) 

    mp.spawn(
        naive_ddp_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()