import torch
import itertools
import time
from cs336_basics.model import scaled_dot_product_attention
from timeit import default_timer as timer

batch_size = 8
d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384, 32768]

results = []

compiled_attention = torch.compile(scaled_dot_product_attention)

for d_model, seq_len in itertools.product(d_model_list, seq_len_list):
    try:
        Q = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.float32).requires_grad_(True)
        K = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.float32)
        V = torch.randn(batch_size, seq_len, d_model, device="cuda", dtype=torch.float32)

        for _ in range(5):
            out = compiled_attention(Q, K, V)
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for _ in range(100):
            out = compiled_attention(Q, K, V)
        end.record()

        torch.cuda.synchronize()
        forward_time = start.elapsed_time(end) / 100  # milliseconds per pass

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = compiled_attention(Q, K, V)
        torch.cuda.synchronize()
        mem_before_backward = torch.cuda.max_memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            out = compiled_attention(Q, K, V)
            torch.cuda.synchronize()

            grad_tensor = torch.ones_like(out, device="cuda")

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out.backward(grad_tensor)
            end.record()
            torch.cuda.synchronize()

            Q.grad.zero_()

        torch.cuda.synchronize()
        backward_time = start.elapsed_time(end) / 100

        results.append((d_model, seq_len, forward_time, backward_time, mem_before_backward / (1024**2)))
    
    except RuntimeError as e:
        print(f"OOM")
        results.append((d_model, seq_len, "OOM", "OOM", "OOM"))

header = f"{'d_model':>8} {'seq_len':>8} {'forward_time (ms)':>20} {'backward_time (ms)':>20} {'memory (MB)':>15}"
print("\nBenchmark Results:")
print(header)
print("-" * len(header))

for d_model, seq_len, forward_time_ms, backward_time_ms, memory_mb in results:
    if forward_time_ms == "OOM":
        print(f"{d_model:8} {seq_len:8} {'OOM':>20} {'OOM':>20} {'OOM':>15}")
    else:
        print(f"{d_model:8} {seq_len:8} {forward_time_ms:20.2f} {backward_time_ms:20.2f} {memory_mb:15.2f}")