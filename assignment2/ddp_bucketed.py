import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self._handles = []

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

        bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        grads = [p for p in self.module.parameters() if p.requires_grad]
        buckets, cur, cur_bytes = [], [], 0
        for p in reversed(grads):
            sz = p.numel() * p.element_size()
            if cur and cur_bytes + sz > bucket_size_bytes:
                buckets.append(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += sz
        if cur:
            buckets.append(cur)

        self._buckets = buckets
        self._bucket_queued = [False] * len(buckets)

        for idx, bucket in enumerate(self._buckets):
            last_p = bucket[-1]

            def make_hook(bucket_idx, bucket_params):
                def hook(_):
                    if not self._bucket_queued[bucket_idx]:
                        grads_list = [p.grad for p in bucket_params]

                        valid = [(i, g) for i, g in enumerate(grads_list) if g is not None]
                        if not valid:
                            return  

                        indices, non_null_grads = zip(*valid)
                        flat = _flatten_dense_tensors(non_null_grads)
                        work = dist.all_reduce(flat, async_op=True)
                        self._handles.append((work, flat, grads_list, indices))
                        self._bucket_queued[bucket_idx] = True
                return hook

            last_p.register_post_accumulate_grad_hook(make_hook(idx, bucket))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        for work, flat, grads_list, indices in self._handles:
            work.wait()
            flat.div_(world_size)
            unflat = _unflatten_dense_tensors(flat, [grads_list[i] for i in indices])
            for i, new_grad in zip(indices, unflat):
                grads_list[i].copy_(new_grad)

        self._handles.clear()
        self._bucket_queued = [False] * len(self._buckets)
