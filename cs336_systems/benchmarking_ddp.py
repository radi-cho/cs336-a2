import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

BACKENDS = ["gloo", "nccl"]
SIZES_MB = [1, 10, 100, 1024]
NUM_PROCESSES_LIST = [2, 4, 6]
DTYPE = torch.float32
REPEAT = 10
RESULTS = []

def get_tensor(size_mb, device):
    numel = (size_mb * 1024 * 1024) // 4  # float32
    return torch.ones(numel, dtype=DTYPE, device=device)

def bench_all_reduce(rank, world_size, backend, size_mb, device, return_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    tensor = get_tensor(size_mb, device)
    torch.cuda.set_device(rank) if device == "cuda" else None

    for _ in range(5):
        dist.all_reduce(tensor, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()

    start = time.time()
    for _ in range(REPEAT):
        dist.all_reduce(tensor, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - start) / REPEAT * 1e3

    dist.destroy_process_group()
    return_dict[rank] = elapsed

def launch(world_size, backend, size_mb, device):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        bench_all_reduce,
        args=(world_size, backend, size_mb, device, return_dict),
        nprocs=world_size,
        join=True
    )
    return sum(return_dict.values()) / world_size

if __name__ == "__main__":
    for backend in BACKENDS:
        device = "cuda" if backend == "nccl" else "cpu"
        if backend == "nccl" and not torch.cuda.is_available():
            continue
        for size_mb in SIZES_MB:
            for world_size in NUM_PROCESSES_LIST:
                if backend == "nccl" and world_size > torch.cuda.device_count():
                    continue
                print(f"Benchmarking: {backend=}, {device=}, {size_mb=}, {world_size=}")
                avg_ms = launch(world_size, backend, size_mb, device)
                RESULTS.append({
                    "backend": backend,
                    "device": device,
                    "size_mb": size_mb,
                    "world_size": world_size,
                    "all_reduce_ms": avg_ms
                })

    df = pd.DataFrame(RESULTS)
    df.to_csv("ddp_allreduce_benchmark.csv")
    print(df)
