import torch
import pandas as pd
import triton.testing as tt

from cs336_systems.flashattn import FlashAttention, FlashAttentionTriton

# SEQUENCE_LENGTHS = [2 ** i for i in range(7, 17)]
SEQUENCE_LENGTHS = [128]
EMBED_DIMS = [2 ** i for i in range(4, 8)]
DTYPES = [torch.float32, torch.bfloat16]
BATCH_SIZE = 1
IS_CAUSAL = True
DEVICE = "cuda"

def bench(fn, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    return tt.do_bench(fn) * 1e3

results = []

for L in SEQUENCE_LENGTHS:
    for D in EMBED_DIMS:
        for dtype in DTYPES:
            print(L, D, dtype)

            Q = torch.randn(BATCH_SIZE, L, D, device=DEVICE, dtype=dtype, requires_grad=True)
            K = torch.randn_like(Q)
            V = torch.randn_like(Q)

            def torch_fwd():
                out = FlashAttention.apply(Q, K, V, IS_CAUSAL)
                torch.cuda.synchronize()
                return out

            def triton_fwd():
                out = FlashAttentionTriton.apply(Q, K, V, IS_CAUSAL)
                torch.cuda.synchronize()
                return out

            out_torch = torch_fwd()
            grad_torch = torch.randn_like(out_torch)
            out_triton = triton_fwd()
            grad_triton = torch.randn_like(out_triton)

            def torch_full():
                out = FlashAttention.apply(Q, K, V, IS_CAUSAL)
                out.backward(grad_torch)
                torch.cuda.synchronize()

            def triton_full():
                out = FlashAttentionTriton.apply(Q, K, V, IS_CAUSAL)
                out.backward(grad_triton)
                torch.cuda.synchronize()

            t_torch_fwd = bench(torch_fwd)
            t_torch_full = bench(torch_full)
            t_torch_bwd = t_torch_full - t_torch_fwd

            t_triton_fwd = bench(triton_fwd)
            t_triton_full = bench(triton_full)
            t_triton_bwd = t_triton_full - t_triton_fwd

            for impl, fwd_ms, bwd_ms, total_ms in [
                ("PyTorch", t_torch_fwd, t_torch_bwd, t_torch_full),
                ("Triton",  t_triton_fwd, t_triton_bwd, t_triton_full)
            ]:
                results.append({
                    "impl": impl,
                    "seq_len": L,
                    "embed_dim": D,
                    "dtype": dtype,
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "total_ms": total_ms,
                })


df = pd.DataFrame(results)
df.to_csv("triton.csv")
print(df)