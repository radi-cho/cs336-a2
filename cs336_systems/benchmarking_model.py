import argparse
import timeit
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy


def benchmark_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int,
    warmup_steps: int,
    timing_steps: int,
    backward: bool,
    adam_step: bool,
    mixed_precision: bool,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    model.train()

    dtype_context = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed_precision else nullcontext()
    dataset = np.random.randint(low=0, high=vocab_size, size=(100000,))

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(warmup_steps):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        with dtype_context:
            outputs = model(inputs)
            if backward:
                loss = cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        if backward:
            loss.backward()
            if adam_step:
                optimizer.step()
        torch.cuda.synchronize()

    times = []
    forward_peaks = []
    train_peaks = []
    for _ in range(timing_steps):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        model.zero_grad()

        if backward:
            with nvtx.range("forward"):
                torch.cuda.reset_peak_memory_stats(device)
                with dtype_context:
                    outputs = model(inputs)
                torch.cuda.synchronize()
            forward_peaks.append(torch.cuda.max_memory_allocated(device))
            
            with dtype_context:
                loss = cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats(device)
            start = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()
            if adam_step:
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                    torch.cuda.synchronize()
            end = timeit.default_timer()
            train_peaks.append(torch.cuda.max_memory_allocated(device))
        else:
            torch.cuda.reset_peak_memory_stats(device)
            start = timeit.default_timer()
            with nvtx.range("forward"):
                with dtype_context:
                    outputs = model(inputs)
                torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_peaks.append(torch.cuda.max_memory_allocated(device))

        times.append(end - start)

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_fwd_peak = np.mean(forward_peaks) / (1024**2)
    avg_trn_peak = np.mean(train_peaks) / (1024**2)

    print(f"{context_length} & {avg_time:.4f} & {std_time:.4f} & {avg_fwd_peak:.0f} & {avg_trn_peak:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--timing_steps", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--adam_step", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    benchmark_model(**vars(args))
