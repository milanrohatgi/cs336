import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()    
        self.module = module  
        self._handles = []

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            def _hook(param: torch.Tensor):
                handle = dist.all_reduce(param.grad, async_op=True)
                self._handles.append(handle)

            p.register_post_accumulate_grad_hook(_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self._handles:
            h.wait()

        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(world_size)

        self._handles.clear()