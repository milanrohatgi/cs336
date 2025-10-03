import torch
import triton
import triton.testing as tt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cs336_systems.flash_fwd_triton import FlashAttentionTriton
from cs336_basics.model import scaled_dot_product_attention
from functools import partial
import time

def benchmark_flash_attention():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    compiled_attention = torch.compile(scaled_dot_product_attention)
    
    seq_lens = [2**i for i in range(7, 17)]
    embed_dims = [2**i for i in range(4, 8)]
    dtypes = [torch.bfloat16, torch.float32]
    
    results = []
    
    for seq_len in seq_lens:
        for embed_dim in embed_dims:
            for dtype in dtypes:
                try:
                    try:
                        Q = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)
                        K = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)
                        V = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM when creating tensors: seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                        results.append({
                            'seq_len': seq_len,
                            'embed_dim': embed_dim,
                            'dtype': str(dtype).split('.')[-1],
                            'forward_triton_ms': float('nan'),
                            'forward_pytorch_ms': float('nan'),
                            'backward_triton_ms': float('nan'),
                            'backward_pytorch_ms': float('nan'),
                            'e2e_triton_ms': float('nan'),
                            'e2e_pytorch_ms': float('nan'),
                            'forward_speedup': float('nan'),
                            'backward_speedup': float('nan'),
                            'e2e_speedup': float('nan'),
                            'error': 'OOM'
                        })
                        torch.cuda.empty_cache()
                        continue
                    
                    Q_triton = Q.clone().detach().requires_grad_(True)
                    K_triton = K.clone().detach().requires_grad_(True)
                    V_triton = V.clone().detach().requires_grad_(True)
                    
                    for _ in range(3):
                        try:
                            _ = FlashAttentionTriton.apply(Q_triton, K_triton, V_triton, True)
                            torch.cuda.synchronize()
                        except Exception as e:
                            print(f"Error during triton forward warmup: {e}")
                    
                    for _ in range(3):
                        try:
                            Q_warmup = Q.clone().detach().requires_grad_(True)
                            K_warmup = K.clone().detach().requires_grad_(True)
                            V_warmup = V.clone().detach().requires_grad_(True)
                            _ = compiled_attention(Q_warmup, K_warmup, V_warmup)
                            torch.cuda.synchronize()
                        except Exception as e:
                            print(f"Error during pytorch forward warmup: {e}")
                    
                    forward_triton_fn = partial(FlashAttentionTriton.apply, Q_triton, K_triton, V_triton, True)
                    try:
                        print("Benchmarking forward triton: ")
                        forward_triton_time = tt.do_bench(forward_triton_fn)
                    except Exception as e:
                        print(f"Error during triton forward benchmark: {e}")
                        forward_triton_time = float('nan')
                    
                    def forward_pytorch_fn():
                        Q_pytorch = Q.clone().detach().requires_grad_(True)
                        K_pytorch = K.clone().detach().requires_grad_(True)
                        V_pytorch = V.clone().detach().requires_grad_(True)
                        return compiled_attention(Q_pytorch, K_pytorch, V_pytorch)
                    
                    try:
                        forward_pytorch_time = tt.do_bench(forward_pytorch_fn)
                    except Exception as e:
                        print(f"Error during pytorch forward benchmark: {e}")
                        forward_pytorch_time = float('nan')
                    
                    try:
                        with torch.no_grad():
                            output_triton = FlashAttentionTriton.apply(Q_triton, K_triton, V_triton, True)
                        grad_output_triton = torch.randn_like(output_triton)
                    except Exception as e:
                        print(f"Error preparing for backward: {e}")
                        output_triton = None
                        grad_output_triton = None
                    
                    for _ in range(3):
                        try:
                            if output_triton is not None:
                                Q_tr_warm = Q.clone().detach().requires_grad_(True)
                                K_tr_warm = K.clone().detach().requires_grad_(True)
                                V_tr_warm = V.clone().detach().requires_grad_(True)
                                out_tr = FlashAttentionTriton.apply(Q_tr_warm, K_tr_warm, V_tr_warm, True)
                                out_tr.backward(grad_output_triton.clone())
                                torch.cuda.synchronize()
                                
                                Q_pt_warm = Q.clone().detach().requires_grad_(True)
                                K_pt_warm = K.clone().detach().requires_grad_(True)
                                V_pt_warm = V.clone().detach().requires_grad_(True)
                                out_pt = compiled_attention(Q_pt_warm, K_pt_warm, V_pt_warm)
                                out_pt.backward(grad_output_triton.clone())
                                torch.cuda.synchronize()
                        except Exception as e:
                            print(f"Error during backward warmup: {e}")
                    
                    def backward_triton_fn():
                        Q_tr = Q.clone().detach().requires_grad_(True)
                        K_tr = K.clone().detach().requires_grad_(True)
                        V_tr = V.clone().detach().requires_grad_(True)
                        output = FlashAttentionTriton.apply(Q_tr, K_tr, V_tr, True)
                        output.backward(grad_output_triton.clone())
                        return output
                    
                    try:
                        print("Benchmarking backward triton: ")
                        backward_triton_time = tt.do_bench(backward_triton_fn) if output_triton is not None else float('nan')
                    except Exception as e:
                        print(f"Error during triton backward benchmark: {e}")
                        backward_triton_time = float('nan')
                    
                    def backward_pytorch_fn():
                        Q_pt = Q.clone().detach().requires_grad_(True)
                        K_pt = K.clone().detach().requires_grad_(True)
                        V_pt = V.clone().detach().requires_grad_(True)
                        output = compiled_attention(Q_pt, K_pt, V_pt)
                        output.backward(grad_output_triton.clone())
                        return output
                    
                    try:
                        backward_pytorch_time = tt.do_bench(backward_pytorch_fn) if output_triton is not None else float('nan')
                    except Exception as e:
                        backward_pytorch_time = float('nan')
                    
                    e2e_triton_time = forward_triton_time + backward_triton_time if not (np.isnan(forward_triton_time) or np.isnan(backward_triton_time)) else float('nan')
                    e2e_pytorch_time = forward_pytorch_time + backward_pytorch_time if not (np.isnan(forward_pytorch_time) or np.isnan(backward_pytorch_time)) else float('nan')
                    
                    forward_speedup = forward_pytorch_time / forward_triton_time if not (np.isnan(forward_triton_time) or np.isnan(forward_pytorch_time) or forward_triton_time == 0) else float('nan')
                    backward_speedup = backward_pytorch_time / backward_triton_time if not (np.isnan(backward_triton_time) or np.isnan(backward_pytorch_time) or backward_triton_time == 0) else float('nan')
                    e2e_speedup = e2e_pytorch_time / e2e_triton_time if not (np.isnan(e2e_triton_time) or np.isnan(e2e_pytorch_time) or e2e_triton_time == 0) else float('nan')
                    
                    results.append({
                        'seq_len': seq_len,
                        'embed_dim': embed_dim,
                        'dtype': str(dtype).split('.')[-1],
                        'forward_triton_ms': forward_triton_time * 1000 if not np.isnan(forward_triton_time) else float('nan'),
                        'forward_pytorch_ms': forward_pytorch_time * 1000 if not np.isnan(forward_pytorch_time) else float('nan'),
                        'backward_triton_ms': backward_triton_time * 1000 if not np.isnan(backward_triton_time) else float('nan'),
                        'backward_pytorch_ms': backward_pytorch_time * 1000 if not np.isnan(backward_pytorch_time) else float('nan'),
                        'e2e_triton_ms': e2e_triton_time * 1000 if not np.isnan(e2e_triton_time) else float('nan'),
                        'e2e_pytorch_ms': e2e_pytorch_time * 1000 if not np.isnan(e2e_pytorch_time) else float('nan'),
                        'forward_speedup': forward_speedup,
                        'backward_speedup': backward_speedup,
                        'e2e_speedup': e2e_speedup,
                        'error': None
                    })
                    
                    print(f"Completed benchmark for seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                    
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM during benchmark: seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                    results.append({
                        'seq_len': seq_len,
                        'embed_dim': embed_dim,
                        'dtype': str(dtype).split('.')[-1],
                        'forward_triton_ms': float('nan'),
                        'forward_pytorch_ms': float('nan'),
                        'backward_triton_ms': float('nan'),
                        'backward_pytorch_ms': float('nan'),
                        'e2e_triton_ms': float('nan'),
                        'e2e_pytorch_ms': float('nan'),
                        'forward_speedup': float('nan'),
                        'backward_speedup': float('nan'),
                        'e2e_speedup': float('nan'),
                        'error': 'OOM'
                    })
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error during benchmark: {e} for seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                    results.append({
                        'seq_len': seq_len,
                        'embed_dim': embed_dim,
                        'dtype': str(dtype).split('.')[-1],
                        'forward_triton_ms': float('nan'),
                        'forward_pytorch_ms': float('nan'),
                        'backward_triton_ms': float('nan'),
                        'backward_pytorch_ms': float('nan'),
                        'e2e_triton_ms': float('nan'),
                        'e2e_pytorch_ms': float('nan'),
                        'forward_speedup': float('nan'),
                        'backward_speedup': float('nan'),
                        'e2e_speedup': float('nan'),
                        'error': str(e)
                    })
                    torch.cuda.empty_cache()
    
    try:
        df = pd.DataFrame(results)
        
        summary_columns = ['seq_len', 'embed_dim', 'dtype', 
                          'forward_triton_ms', 'forward_pytorch_ms', 'forward_speedup',
                          'backward_triton_ms', 'backward_pytorch_ms', 'backward_speedup',
                          'e2e_triton_ms', 'e2e_pytorch_ms', 'e2e_speedup', 'error']
        
        summary_table = df[summary_columns]
        summary_table = summary_table.sort_values(by=['seq_len', 'embed_dim', 'dtype'])
        
        print("\nBenchmark Results:")
        print(summary_table.to_string(index=False))
        
        df.to_csv('flash_attention_benchmark_results.csv', index=False)
        
        return df
    except Exception as e:
        print(f"Error creating results dataframe: {e}")
        print("Raw results:", results)
        return results

if __name__ == "__main__":
    print("Starting FlashAttention benchmarking...")
    results = benchmark_flash_attention()