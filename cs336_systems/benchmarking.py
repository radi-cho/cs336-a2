import argparse
import timeit
import torch
import numpy as np
from cs336_basics.model import BasicsTransformerLM
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

    model.train()

    dataset = np.random.randint(low=0, high=vocab_size, size=(100000,))

    for _ in range(warmup_steps):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        outputs = model(inputs)
        loss = cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        torch.cuda.synchronize()

    times = []
    for _ in range(timing_steps):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        model.zero_grad()

        if backward:
            outputs = model(inputs)
            loss = cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
            torch.cuda.synchronize()

            start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
        else:
            start = timeit.default_timer()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end = timeit.default_timer()

        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f" & {avg_time:.4f} & {std_time:.4f} \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--timing_steps", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    benchmark_model(**vars(args))
