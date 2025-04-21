import argparse
import timeit
import torch
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
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
    do_backward: bool,
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

    dataset = np.random.randint(low=0, high=vocab_size, size=(100_000,)) # dtype=np.int64
    inputs, targets = get_batch(dataset, batch_size, context_length, device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    def step():
        outputs = model(inputs)
        loss = cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        if do_backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    print(f"Running {warmup_steps} warm-up steps...")
    for _ in range(warmup_steps):
        step()

    times = []
    for _ in range(timing_steps):
        start = timeit.default_timer()
        step()
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
    parser.add_argument("--do_backward", action="store_true")
    args = parser.parse_args()

    benchmark_model(**vars(args))