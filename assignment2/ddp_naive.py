import os
import argparse
import numpy as np
from timeit import default_timer as timer
from torch.cuda import nvtx

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cs336_basics.model as transformer
import cs336_basics.optimizer as optim
import cs336_basics.data as data

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
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

    for i in range(args.max_iters):
        nvtx.range_push(f"iter_{i}")
        batch_seed = args.seed + i
        minibatch = generate_random_batch(
            examples_per_rank, 
            args.context_length, 
            args.vocab_size, 
            device,
            seed=batch_seed + rank 
        )

        nvtx.range_push("forward")
        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(minibatch)
            loss = outputs.mean()
            nvtx.range_pop()
            nvtx.range_push("backward")
            loss.backward()
            nvtx.range_pop()

        nvtx.range_push("gradient_sync")
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size 
        nvtx.range_pop()

        nvtx.range_push("optimizer_step")
        optimizer.step()
        optimizer.zero_grad()
        nvtx.range_pop()
        nvtx.range_pop()

    if rank == 0:
        model_state = {
            'model_state_dict': model.state_dict(),
        }
        torch.save(model_state, 'ddp_model.pt')
        
    cleanup()

def train_single_worker(args):
    device = torch.device("cuda:0")

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
    
    if args.compile:
        model = torch.compile(model)
    
    optimizer = optim.AdamW(model.parameters())
    
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    for i in range(args.max_iters):
        full_batch = []
        for rank in range(args.world_size):
            batch_seed = args.seed + i
            rank_batch = generate_random_batch(
                args.batch_size // args.world_size,
                args.context_length,
                args.vocab_size,
                device,
                seed=batch_seed + rank  
            )
            full_batch.append(rank_batch)
        
        full_batch = torch.cat(full_batch, dim=0)

        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(full_batch)
            loss = outputs.mean()
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model_state = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(model_state, 'single_worker_model.pt')
    return model

def verify_models():

    ddp_state = torch.load('ddp_model.pt')
    single_state = torch.load('single_worker_model.pt')
    
    ddp_model_state = ddp_state['model_state_dict']
    single_model_state = single_state['model_state_dict']

    all_match = True
    max_diff = 0.0
    mismatched_keys = []
    
    for key in single_model_state:
        if key in ddp_model_state:
            diff = (single_model_state[key] - ddp_model_state[key]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-5:
                all_match = False
                mismatched_keys.append((key, diff))
    
    if all_match:
        print(f"PASSED")
        return True
    else:
        print(f"FUCK")
        return False

def main():
    parser = argparse.ArgumentParser(description="Naive Distributed Data Parallel Training")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--rope_theta", type=int, default=10000)

    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--use_bf16", action="store_true", help="Use mixed precision with BF16")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    parser.add_argument("--world_size", type=int, default=2, help="Number of processes/GPUs")

    parser.add_argument("--verify", action="store_true", help="Run single-process training to verify results")
    
    args = parser.parse_args()

    if args.d_ff is None:
        args.d_ff = int(8 * args.d_model / 3)

    torch.manual_seed(args.seed) 

    mp.spawn(
        naive_ddp_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )

    if args.verify:
        train_single_worker(args)
        verify_models()

if __name__ == "__main__":
    main()