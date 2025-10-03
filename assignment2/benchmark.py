import torch
import cs336_basics.model as transformer
import cs336_basics.optimizer as optim
import cs336_basics.nn_utils as nn_utils
import argparse
import numpy as np
from timeit import default_timer as timer
import torch.cuda.nvtx as nvtx
from einops import einsum
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default = 768)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_layers", type=int, default = 8)
    parser.add_argument("--num_heads", type=int, default = 12)
    parser.add_argument("--backwards", action="store_true")
    parser.add_argument("--vocab_size", type=int, default = 10000)
    parser.add_argument("--rope_theta", type=int, default = 10000)
    parser.add_argument("--warmup_steps", type=int, default = 5)
    parser.add_argument("--n_steps", type=int, default = 10)
    parser.add_argument("--use_bf16", action="store_true", help="Use mixed precision with BF16")
    parser.add_argument("--profile_memory", action="store_true")

    args = parser.parse_args()
    if args.d_ff is None:
        args.d_ff = int(8 * args.d_model / 3)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = transformer.BasicsTransformerLM(vocab_size = args.vocab_size, 
                                context_length = args.context_length, 
                                d_model = args.d_model, 
                                num_layers = args.num_layers, 
                                num_heads = args.num_heads, 
                                d_ff = args.d_ff, 
                                rope_theta = args.rope_theta).to(device)
    
    model = torch.compile(model)
    
    batch = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length)).to(device)

    optimizer = optim.AdamW(model.parameters())
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32

    if args.backwards:
        for i in range(args.warmup_steps):
            with torch.autocast(device_type="cuda", dtype=dtype):
                outs = model.forward(batch)
            
            outs.mean().backward()

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        times = []

        if args.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        for trial in range(args.n_steps):
            start_time = timer()

            with torch.autocast(device_type="cuda", dtype=dtype):
                outs = model.forward(batch)
            loss = outs.mean()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()

            end_time = timer()

            times.append((end_time - start_time))

        if args.profile_memory:
            if args.use_bf16:
                torch.cuda.memory._dump_snapshot(f"model_d{args.d_model}_c{args.context_length}_bf16_backward_memory_snapshot.pickle")
            else:                
                torch.cuda.memory._dump_snapshot(f"model_d{args.d_model}_c{args.context_length}_backward_memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

    else:
        # warmup
        for i in range(args.warmup_steps):
            with torch.autocast(device_type="cuda", dtype=dtype):
                outs = model.forward(batch)

        torch.cuda.synchronize()


        if args.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        times = []

        for trial in range(args.n_steps):
            start_time = timer()
            with torch.autocast(device_type="cuda", dtype=dtype):
                outs = model.forward(batch)

            torch.cuda.synchronize()

            end_time = timer()

            times.append((end_time - start_time))

        if args.profile_memory:
            if args.use_bf16:
                torch.cuda.memory._dump_snapshot(f"model_d{args.d_model}_c{args.context_length}_bf16_forward_memory_snapshot.pickle")
            else:                
                torch.cuda.memory._dump_snapshot(f"model_d{args.d_model}_c{args.context_length}_forward_memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
    mean_time = np.mean(times)
    var_times = np.var(times)

    print(f"mean_time: {mean_time}, var_times: {var_times}")


if __name__ == "__main__":
    main()