import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.grad_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        def hook(param):
            if param.grad is not None:
                self.grad_handles.append(dist.all_reduce(param.grad, async_op=True))

        self.grad_handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()

        self.grad_handles.clear()

        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= world_size


class DDPBucket(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.grad_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        buckets = []
        curr_bucket = []
        curr_size = 0
        for p in reversed(list(self.module.parameters())):
            p_size = p.numel() * p.element_size()
            if curr_bucket and curr_size + p_size > self.bucket_size_bytes:
                buckets.append(curr_bucket)
                curr_bucket = []
                curr_size = 0
            curr_bucket.append(p)
            curr_size += p_size
        if curr_bucket:
            buckets.append(curr_bucket)

        self.buckets = []
        self.lookup = {}

        for b in buckets:
            buf = torch.zeros(sum(p.numel() for p in b), dtype=b[0].dtype, device=b[0].device)
            offsets = {}
            offset = 0
            for p in b:
                n = p.numel()
                offsets[p] = (offset, offset + n)
                offset += n

            self.buckets.append({
                "params": b,
                "buffer": buf,
                "offsets": offsets
            })

            for p in b:
                self.lookup[p] = self.buckets[-1]

        for bucket in self.buckets:
            for p in bucket["params"]:
                def hook(grad):
                    b = self.lookup[p]
                    start, end = b["offsets"][p]
                    b["buffer"][start:end].copy_(grad.view(-1))
                    if p is b[-1]["buffer"]:
                        handle = dist.all_reduce(b["buffer"], async_op=True)
                        self.grad_handles.append(handle)
                    return grad
                p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()

        for bucket in self.buckets:
            buf = bucket["buffer"]
            for p in bucket["params"]:
                start, end = bucket["offsets"][p]
                p.grad.view(-1).copy_(buf[start:end])

        self.grad_handles.clear()

        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= world_size
