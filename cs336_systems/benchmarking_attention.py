import torch
import pandas as pd
import timeit

from cs336_basics.model import scaled_dot_product_attention


batch_size = 8
dmodels = [1024, 2048, 4096, 8192]
seq_lens = [1024, 4096, 8192, 16384]
n_warmup = 10
n_runs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for d_model in dmodels:
    for seq_len in seq_lens:
        try:
            print(d_model, seq_len)

            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            mask = None

            for _ in range(n_warmup):
                out = scaled_dot_product_attention(Q, K, V, mask)
                torch.cuda.synchronize()
                out.mean().backward()
                torch.cuda.synchronize()
                Q.grad = K.grad = V.grad = None

            def forward_pass():
                out = scaled_dot_product_attention(Q, K, V, mask)
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated(device)

            forward_timer = timeit.Timer(forward_pass)
            forward_time = forward_timer.timeit(number=n_runs) / n_runs

            mem_before_bwd = torch.cuda.memory_allocated(device)

            def backward_pass():
                out = scaled_dot_product_attention(Q, K, V, mask)
                out.mean().backward()
                torch.cuda.synchronize()
                Q.grad = K.grad = V.grad = None

            backward_timer = timeit.Timer(backward_pass)
            backward_time = backward_timer.timeit(number=n_runs) / n_runs

            peak_mem = torch.cuda.max_memory_allocated(device)
        except:
            forward_time = backward_time = mem_before_bwd = peak_mem = None

        results.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "forward_time_s": forward_time,
            "backward_time_s": backward_time,
            "mem_before_bwd_bytes": mem_before_bwd,
            "peak_mem_bytes": peak_mem
        })


df = pd.DataFrame(results)
print(df)
