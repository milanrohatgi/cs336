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

def annotated_scaled_dot_product_attention(
    Q,
    K,
    V,
    mask = None,
):
    d_k = K.shape[-1]

    nvtx.range_push("attention scores")
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    nvtx.range_pop()


    nvtx.range_push("softmax")
    attention_weights = nn_utils.softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    nvtx.range_pop()

    nvtx.range_push("final matmul")
    out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    nvtx.range_pop()
    return out

transformer.scaled_dot_product_attention = annotated_scaled_dot_product_attention

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
    
    
    
    batch = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length)).to(device)
    

    optimizer = optim.AdamW(model.parameters())

    config_label = f"model_d{args.d_model}_c{args.context_length}"
    nvtx.range_push(config_label)
    
    nvtx.range_push("warmup_phase")
    if args.backwards:
        for i in range(args.warmup_steps):
            outs = model.forward(batch)
            loss = outs.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            nvtx.range_pop()  
    else:
        for i in range(args.warmup_steps):
            nvtx.range_push(f"warmup_step_{i}")
            outs = model.forward(batch)
            nvtx.range_pop() 
    
    torch.cuda.synchronize()
    nvtx.range_pop()  
    
    nvtx.range_push("measurement_phase")
    
    if args.backwards:
        for trial in range(args.n_steps):
            nvtx.range_push(f"trial_{trial}_forward")
            outs = model.forward(batch)
            nvtx.range_pop() 


            
            nvtx.range_push(f"trial_{trial}_backward")
            loss = outs.mean()
            loss.backward()
            nvtx.range_pop()  
            
            nvtx.range_push(f"trial_{trial}_optimizer")
            optimizer.step()
            optimizer.zero_grad()
            nvtx.range_pop()

            torch.cuda.synchronize()
    else:
        # Forward pass only
        for trial in range(args.n_steps):
            nvtx.range_push(f"trial_{trial}_forward")
            outs = model.forward(batch)
            nvtx.range_pop()

    torch.cuda.synchronize()
    
    nvtx.range_pop() 
    nvtx.range_pop()  

if __name__ == "__main__":
    main()

