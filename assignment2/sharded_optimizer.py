import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any

class ShardedOptimizer():
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            input_groups = params
        else:
            input_groups = [{'params': list(params)}]
            
        self.all_params = []
        for g in input_groups:
            self.all_params.extend(g['params'])
            
        self.owner = {p: (i % self.world_size) for i, p in enumerate(self.all_params)}
        
        local_groups = []
        for g in input_groups:
            local_params = [p for p in g['params'] if self.owner[p] == self.rank]
            if local_params:
                new_group = dict(g)
                new_group['params'] = local_params
                local_groups.append(new_group)
                
        self.opt = optimizer_cls(local_groups, **kwargs)
        
        self.param_groups = self.opt.param_groups
        self.state = self.opt.state
        
    def add_param_group(self, param_group: dict[str, Any]):
        for p in param_group['params']:
            idx = len(self.all_params)
            self.owner[p] = idx % self.world_size
            self.all_params.append(p)

        local_params = [p for p in param_group['params'] if self.owner[p] == self.rank]
        if local_params:
            new_group = dict(param_group)
            new_group['params'] = local_params
            self.opt.add_param_group(new_group)
            self.param_groups = self.opt.param_groups
            
    def step(self, closure=None, **kwargs):
        loss = self.opt.step(closure, **kwargs)
        for p, owner in self.owner.items():
            dist.broadcast(p.data, src=owner)
            
        return loss
    
    def zero_grad(self, set_to_none=False):
        self.opt.zero_grad(set_to_none)