import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.optim import SGD


def generate_data(seed, batch_size, input_dim):
    torch.manual_seed(seed)
    X = torch.randn(batch_size, input_dim)
    true_w = torch.randn(input_dim, 1)
    y = X @ true_w + 0.1 * torch.randn(batch_size, 1)
    return X, y


def build_model(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def train_single(seed, X, y, epochs, lr):
    torch.manual_seed(seed)
    model = build_model(X.size(1), 16, y.size(1))
    opt = SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        pred = model(X)
        loss = ((pred - y)**2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return {k: v.clone() for k, v in model.state_dict().items()}


def ddp_train(rank, world_size, seed, batch_size, input_dim, epochs, lr, tmp_file):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    X, y = generate_data(seed, batch_size, input_dim)
    per_rank = batch_size // world_size
    start = rank * per_rank
    end = start + per_rank
    X_shard, y_shard = X[start:end], y[start:end]

    torch.manual_seed(seed)
    model = build_model(input_dim, 16, y.size(1))
    opt = SGD(model.parameters(), lr=lr)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    for _ in range(epochs):
        pred = model(X_shard)
        loss = ((pred - y_shard)**2).mean()
        loss.backward()

        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

        opt.step()
        opt.zero_grad()

    if rank == 0:
        torch.save(model.state_dict(), tmp_file)

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    seed = 0
    tmp_file = "ddp_model.pt"

    X, y = generate_data(seed, args.batch_size, args.input_dim)
    baseline_sd = train_single(seed, X, y, args.epochs, args.lr)

    mp.spawn(
        ddp_train,
        args=(args.world_size, seed, args.batch_size, args.input_dim, args.epochs, args.lr, tmp_file),
        nprocs=args.world_size,
        join=True
    )

    ddp_sd = torch.load(tmp_file)
    max_diff = 0.0
    for k in baseline_sd:
        diff = (baseline_sd[k] - ddp_sd[k]).abs().max().item()
        max_diff = max(max_diff, diff)

    if max_diff < 1e-5:
        print("good!")
    else:
        print("bad!")


if __name__ == "__main__":
    main()
