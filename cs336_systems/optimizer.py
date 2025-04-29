import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from typing import Type, Any, Iterable

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any
    ):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        params = list(params)

        if len(params) > 0 and isinstance(params[0], dict):
            self.grouped = params
        else:
            self.grouped = [{ "params": params }]

        flat = []
        for g in self.grouped:
            flat.extend(g["params"])
        self.owner = { p: idx % self.world_size for idx, p in enumerate(flat) }

        for p in flat:
            dist.broadcast(p.data, src=0)

        cur_group = []
        for g in self.grouped:
            cur_params = [p for p in g["params"] if self.owner[p] == self.rank]
            if not cur_params:
                continue
            grp = { k: v for k, v in g.items() if k != "params" }
            grp["params"] = cur_params
            cur_group.append(grp)

        super().__init__(cur_group, kwargs)
        self.inopt = optimizer_cls(cur_group, **kwargs)

    def step(self, closure=None):
        loss = self.inopt.step(closure) if closure else self.inopt.step()

        for p, owner in self.owner.items():
            dist.broadcast(p.data, src=owner)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        if not hasattr(self, "inopt"):
            return super().add_param_group(param_group)

        self._grouped.append(param_group)
        base = len(self._owner)
        for i, p in enumerate(param_group["params"]):
            self._owner[p] = (base + i) % self.world_size

        local_ps = [p for p in param_group["params"] if self._owner[p] == self.rank]
        grp = { k: v for k, v in param_group.items() if k != "params" }
        grp["params"] = local_ps

        super().add_param_group(grp)
        if local_ps:
            self.inopt.add_param_group(grp)
