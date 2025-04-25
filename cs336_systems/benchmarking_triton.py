import torch
import pandas as pd
import triton.testing as tt

from cs336_systems.flashattn import FlashAttention, FlashAttentionTriton

SEQUENCE_LENGTHS = [2 ** i for i in range(7, 17)]
EMBED_DIMS = [2 ** i for i in range(4, 8)] # TODO: support 3
DTYPES = [torch.float32, torch.bfloat16]
BATCH_SIZE = 1
IS_CAUSAL = True
DEVICE = "cuda"

def bench(fn, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    return tt.do_bench(fn, flush_l2=False).mean * 1e3

results = []

for L in SEQUENCE_LENGTHS:
    for D in EMBED_DIMS:
        for dtype in DTYPES:
            Q = torch.randn(BATCH_SIZE, L, D, device=DEVICE, dtype=dtype, requires_grad=True)
            K = torch.randn_like(Q)
            V = torch.randn_like(Q)

            def torch_fwd():
                return FlashAttention.apply(Q, K, V, IS_CAUSAL)

            def triton_fwd():
                return FlashAttentionTriton.apply(Q, K, V, IS_CAUSAL)

            out_torch = torch_fwd()
            grad_torch = torch.randn_like(out_torch)
            out_triton = triton_fwd()
            grad_triton = torch.randn_like(out_triton)

            def torch_bwd_only():
                out_torch.backward(grad_torch, retain_graph=True)

            def triton_bwd_only():
                out_triton.backward(grad_triton, retain_graph=True)

            t_torch_fwd = bench(torch_fwd)
            t_torch_bwd = bench(torch_bwd_only)
            t_torch_total = t_torch_fwd + t_torch_bwd

            t_triton_fwd = bench(triton_fwd)
            t_triton_bwd = bench(triton_bwd_only)
            t_triton_total = t_triton_fwd + t_triton_bwd

            for impl, fwd_ms, bwd_ms, total_ms in [
                ("PyTorch", t_torch_fwd, t_torch_bwd, t_torch_total),
                ("Triton",  t_triton_fwd, t_triton_bwd, t_triton_total)
            ]:
                results.append({
                    "impl": impl,
                    "seq_len": L,
                    "embed_dim": D,
                    "dtype": dtype.name,
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "total_ms": total_ms,
                })


df = pd.DataFrame(results)
print(df)