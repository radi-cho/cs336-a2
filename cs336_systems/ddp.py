import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.grad_handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        self.grad_handles = []
        for param in self.module.parameters():
            param.register_post_accumulate_grad_hook(
                lambda grad, p: self.grad_handles.append(
                    dist.all_reduce(grad, async_op=True)
                )
            )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()

        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= world_size
