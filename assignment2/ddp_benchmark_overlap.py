import os
import argparse
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import cs336_basics.model as transformer
import cs336_basics.optimizer as optim
from cs336_systems.ddp_bucketed import DDPBucketed
from cs336_systems.sharded_optimizer import ShardedOptimizer

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def generate_random_batch(batch_size, context_length, vocab_size, device, seed):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)

def overlap_ddp_worker(rank, world_size, args):
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

    ddp_model = DDPBucketed(model, args.bucket_size_mb)
    torch.cuda.synchronize()
    init_mem = torch.cuda.memory_allocated(device) / 1024**2

    if rank == 0:
        print(f"[Rank {rank}] Peak memory after model init: {init_mem:.2f} MB")

    if args.use_sharded_optim:
        optimizer = ShardedOptimizer(model.parameters(), optim.AdamW)
    else:
        optimizer = optim.AdamW(ddp_model.parameters())

    total_params = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    if rank == 0:
        print(f"[Rank {rank}] Optimizer covers {total_params/1e6:.1f}M parameters")

    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    per_rank_bs = args.batch_size // world_size

    for i in range(args.warmup_steps):
        xb = generate_random_batch(per_rank_bs, args.context_length, args.vocab_size, device, args.seed + i + rank)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = ddp_model(xb).mean()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dist.barrier() 

    xb = generate_random_batch(per_rank_bs, args.context_length, args.vocab_size, device, args.seed + args.warmup_steps + rank)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = ddp_model(xb).mean()
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    torch.cuda.synchronize()
    pre_step_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    if rank == 0:
        print(f"[Rank {rank}] Peak memory directly before optimizer step: {pre_step_peak:.2f} MB")

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    torch.cuda.synchronize()
    post_step_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    if rank == 0:
        print(f"[Rank {rank}] Peak memory directly after optimizer step:  {post_step_peak:.2f} MB")

    iteration_times = []
    for i in range(args.n_steps):
        xb = generate_random_batch(per_rank_bs, args.context_length, args.vocab_size, device, args.seed + args.warmup_steps + i + rank)
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = ddp_model(xb).mean()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end = time.perf_counter()

        iteration_times.append(end - start)
        if rank == 0:
            print(f"[Iter {i}] Time per iteration: {end - start:.4f} s")

    dist.barrier()
    if rank == 0:
        avg_time = sum(iteration_times) / len(iteration_times)
        print(f"[Rank {rank}] Average iteration time over {args.n_steps} steps: {avg_time:.4f} s")

    cleanup()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--context_length" ,type=int, default=1024)
    p.add_argument("--d_model", type=int, default=1600)
    p.add_argument("--d_ff",  type=int, default=6400)
    p.add_argument("--num_layers", type=int ,default=48)
    p.add_argument("--num_heads", type=int,default=25)
    p.add_argument("--vocab_size", type=int,default=10000)
    p.add_argument("--rope_theta",type=int,default=10000)
    p.add_argument("--warmup_steps",type=int, default=3)
    p.add_argument("--n_steps",type=int, default=10)
    p.add_argument("--world_size",type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--bucket_size_mb", type=int,   default=10)
    p.add_argument("--use_sharded_optim", action="store_true")
    args = p.parse_args()

    mp.spawn(
        overlap_ddp_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
