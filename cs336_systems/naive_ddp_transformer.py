import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import Transformer
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def train_step(rank, world_size, args):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}"
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    ).to(device)

    model.train()
    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta0, args.beta1), 1e-8, args.decay)

    xb = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    yb = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    for _ in range(5):
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    t0 = time.time()
    logits = model(xb)
    loss = cross_entropy(logits, yb)
    torch.cuda.synchronize()
    t1 = time.time()

    loss.backward()
    torch.cuda.synchronize()
    t2 = time.time()

    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size
    torch.cuda.synchronize()
    t3 = time.time()

    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t4 = time.time()

    if rank == 0:
        print(f"Forward+Backward Time: {t2 - t1:.4f}s")
        print(f"Comm Time (all_reduce): {t3 - t2:.4f}s")
        print(f"Total Step Time: {t4 - t0:.4f}s")

    dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=6400)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--beta0", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.999)
    parser.add_argument("--decay", type=float, default=0.01)
    args = parser.parse_args()

    mp.spawn(train_step, args=(2, args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
