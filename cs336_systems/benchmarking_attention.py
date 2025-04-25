import torch
import timeit
import pandas as pd

from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
DMODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
N_WARMUP = 10
N_RUNS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for d_model in DMODELS:
    for seq_len in SEQ_LENS:
        print(f"d_model={d_model}, seq_len={seq_len}")
        try:
            Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
            K = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
            V = torch.randn(BATCH_SIZE, seq_len, d_model, device=DEVICE, requires_grad=True)
            mask = None

            for _ in range(N_WARMUP):
                out = scaled_dot_product_attention(Q, K, V, mask)
                torch.cuda.synchronize()
                out.mean().backward()
                torch.cuda.synchronize()
                Q.grad = K.grad = V.grad = None

            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated(DEVICE)
            start = timeit.default_timer()
            for _ in range(N_RUNS):
                out = scaled_dot_product_attention(Q, K, V, mask)
                torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_time = (end - start) / N_RUNS

            mem_before_bwd = torch.cuda.memory_allocated(DEVICE)

            start = timeit.default_timer()
            for _ in range(N_RUNS):
                out.mean().backward(retain_graph=True)
                torch.cuda.synchronize()
                Q.grad = K.grad = V.grad = None
            end = timeit.default_timer()
            backward_time = (end - start) / N_RUNS

            # peak_mem = torch.cuda.max_memory_allocated(DEVICE)
        except Exception as e:
            print(e)
            forward_time = backward_time = mem_before_bwd = None

        results.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "forward_time_s": forward_time,
            "backward_time_s": backward_time,
            "mem_before_bwd_bytes": mem_before_bwd,
            # "peak_mem_bytes": peak_mem
        })

df = pd.DataFrame(results)
df.to_csv("attention.csv")
print(df)
