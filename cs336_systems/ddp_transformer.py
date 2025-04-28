import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from ddp import DDPBucket

def train_step(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}"
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    if args.wrapper:
        model = DDPBucket(model, args.bucket_size_mb)
    model.train()
    optimizer = AdamW(model.parameters(), args.max_lr, (args.beta0, args.beta1), 1e-5, args.decay)

    xb = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    yb = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    if args.flat and not args.wrapper:
        param_list = list(model.parameters())

    for _ in range(5):
        print("step")
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        if args.wrapper:
            model.finish_gradient_synchronization()
        elif args.flat:
            grads = [p.grad for p in param_list if p.grad is not None]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat)
            flat.div_(world_size)
            for buf, p in zip(torch._utils._unflatten_dense_tensors(flat, grads), grads):
                p.copy_(buf)
        else:
            for param in model.parameters():
                dist.all_reduce(param.grad)
                param.grad /= world_size
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()

    logits = model(xb)
    loss = cross_entropy(logits, yb)
    torch.cuda.synchronize()
    t1 = time.time()

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.time()

    if args.wrapper:
        model.finish_gradient_synchronization()
    elif args.flat:
        grads = [p.grad for p in param_list if p.grad is not None]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(world_size)
        for buf, p in zip(torch._utils._unflatten_dense_tensors(flat, grads), grads):
            p.copy_(buf)
    else:
        for param in model.parameters():
            dist.all_reduce(param.grad)
            param.grad /= world_size

    torch.cuda.synchronize()
    t3 = time.time()

    optimizer.step()
    torch.cuda.synchronize()
    t4 = time.time()

    if rank == 0:
        print(f"Mode: {'wrapper' if args.wrapper else 'flat' if args.flat else 'naive'}")
        print(f"Forward Time: {t1 - t0:.4f}s")
        print(f"Backward Time: {t2 - t1:.4f}s")
        print(f"Comm Time: {t3 - t2:.4f}s")
        print(f"Total Step Time: {t4 - t0:.4f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=1600)
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=25)
    parser.add_argument("--d_ff", type=int, default=6400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bucket_size_mb", type=int, default=1)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--beta0", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.999)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--wrapper", action="store_true")
    args = parser.parse_args()

    mp.spawn(train_step, args=(2, args), nprocs=2, join=True)
