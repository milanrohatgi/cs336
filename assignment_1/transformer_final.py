import torch 
from einops import rearrange, einsum
import math
import numpy as np

class Dropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training:    
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return mask * x / (1 - self.p)

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std = self.std, a = -3 * self.std, b = 3 * self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, '... in, out in ->... out')

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embedding_matrix = torch.nn.Parameter(torch.empty((self.vocab_size, self.d_model), **factory_kwargs))

        # modified initialization of weights for weight tying

        std = 1/math.sqrt(self.d_model)     
        a, b = -2 * std, 2 * std       

        torch.nn.init.trunc_normal_(
            self.embedding_matrix,
            mean=0.0,
            std=std,
            a=a, b=b
        )

        #torch.nn.init.trunc_normal_(self.embedding_matrix, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gain = torch.nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        x_norm = x / rms
        out = x_norm * self.gain

        return out.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff = None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.d_model = d_model

        if d_ff == None:
            raw = 8 / 3 * d_model
            self.d_ff = int(round(raw / 64) * 64)
        else:
            self.d_ff = d_ff
        
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        left_gate = self.w1.forward(x)
        right_gate = self.w3.forward(x)

        glu = (left_gate * torch.sigmoid(left_gate)) * right_gate

        return self.w2.forward(glu)    
    
class SiLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff = None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        d_ff = d_ff or 4 * d_model
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.w1(x)
        silu = up_proj * torch.sigmoid(up_proj)
        return self.w2(silu)

class RoPE(torch.nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0

        freqs = torch.arange(0, d_k, 2).to(torch.float32)
        inv_theta = theta ** (-freqs / d_k)      # shape: (d_k // 2,)

        positions = torch.arange(max_seq_len).to(torch.float32)   # shape :(max_seq_len,)

        angles = einsum(inv_theta, positions, "dk, max -> max dk")

        precomputed_sin = torch.sin(angles).to(torch.float32).to(device)
        precomputed_cos = torch.cos(angles).to(torch.float32).to(device)

        self.register_buffer("precomputed_sin", precomputed_sin, persistent=False)
        self.register_buffer("precomputed_cos", precomputed_cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        sin = self.precomputed_sin[token_positions]
        cos = self.precomputed_cos[token_positions]

        sin = rearrange(sin, '... n d -> ... 1 n d')
        cos = rearrange(cos, '... n d -> ... 1 n d')

        x_rot_even = cos * x_even - sin * x_odd
        x_rot_odd  = sin * x_even + cos * x_odd

        x_out = rearrange(
            torch.stack([x_rot_even, x_rot_odd], dim=-1),
            '... n d two -> ... n (d two)'
        )

        return x_out.to(in_dtype)
    
def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dimension, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dimension, keepdim=True)
    return x_exp / x_sum


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, max_seq_len, num_heads):
        super().__init__()

        g0 = math.log2(max_seq_len * max_seq_len - max_seq_len)
        self.qk_scale = torch.nn.Parameter(torch.ones(num_heads) * g0)

        self.eps = 1e-8

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor = None) -> torch.Tensor:
        d_k = Q.shape[-1]

        Q_norm = torch.sqrt((Q*Q).sum(-1, keepdim=True)) + self.eps
        K_norm = torch.sqrt((K*K).sum(-1, keepdim=True)) + + self.eps
        Q_hat, K_hat = Q / Q_norm, K / K_norm

        g = self.qk_scale.view(1, -1, 1, 1)     
        Q_scaled, K_scaled = Q_hat * g, K_hat

        pre_softmax = einsum(Q_scaled, K_scaled, "... n dk, ... m dk -> ... n m") # n, or seq_len x m, or d_k

        if M is not None:
            pre_softmax = torch.where(M, pre_softmax, float("-inf"))
        
        softmaxed = softmax(pre_softmax, -1)

        attn_weights = softmaxed

        return einsum(attn_weights, V, "... n m, ... m dv -> ... n dv")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE = None):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_QKV = Linear(d_model, 3 * d_model)
        self.W_O = Linear(d_model, d_model)

        self.use_rope = rope is not None
        if self.use_rope:
            self.rope = rope
            max_seq_len = rope.precomputed_sin.size(0)
        else:
            raise ValueError("implementation needs rope :(")

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

        self.attn = ScaledDotProductAttention(max_seq_len, num_heads)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        # x shape: ... seq_length d_model
        seq_len = x.shape[-2]
        causal = self.mask[:seq_len, :seq_len]

        QKV = self.W_QKV(x)
        Q, K, V = QKV.chunk(3, dim=-1)

        Q = rearrange(Q, "... seq_len (h dk) -> ... h seq_len dk", h = self.num_heads)
        K = rearrange(K, "... seq_len (h dk) -> ... h seq_len dk", h = self.num_heads)
        V = rearrange(V, "... seq_len (h dk) -> ... h seq_len dk", h = self.num_heads)

        if self.use_rope:
            if token_positions is None:
                raise ValueError("token_positions must be passed in when using RoPE")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        out = rearrange(self.attn(Q, K, V, M = causal), "... h seq_len dv -> ... seq_len (h dv)")

        return self.W_O(out)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope = rope)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor= None):
        attn_out = self.attn(self.norm1(x), token_positions)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()

        self.embedding = Embedding(vocab_size , d_model)

        self.rope = RoPE(theta=rope_theta, d_k = d_model // num_heads, max_seq_len=context_length)

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, self.rope)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.out_proj = Linear(d_model, vocab_size)

        self.out_proj.W = self.embedding.embedding_matrix # weight tying
    
    def forward(self, token_ids: torch.Tensor):
        x = self.embedding(token_ids)

        *batch_shape, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(*batch_shape, seq_len)

        for block in self.blocks:
            x = block(x, positions)

        x = self.out_proj(self.final_norm(x))

        return x
    
def cross_entropy_loss(logits: torch.Tensor, targets:torch.Tensor):
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values

    log_sum_exp = torch.logsumexp(logits_stable, dim=-1)

    true_logits = torch.gather(logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = -true_logits + log_sum_exp

    return loss.mean()

@torch.no_grad()
def newtonschulz(G: torch.Tensor, ns_steps: int) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(torch.bfloat16)
    transposed = False
    M, N = X.shape[-2], X.shape[-1]
    if M > N:
        X = X.T
        transposed = True

    spec_norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (spec_norm + 1e-7)

    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None or p.ndim < 2:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']

                buf.mul_(mu).add_(grad, alpha=1 - mu)

                update = grad.add(buf, alpha=mu)

                update_ortho = newtonschulz(update, ns_steps=steps)

                M, N = p.shape[-2], p.shape[-1]
                alpha = lr * (M / N) ** 0.5

                p.data.add_(update_ortho, alpha=-alpha)

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(p.data)
                    state['moment2'] = torch.zeros_like(p.data)


                moment1, moment2 = state['moment1'], state['moment2']
                beta1, beta2 = group['betas']
                state['step'] += 1


                moment1.mul_(beta1).add_(grad, alpha=1 - beta1)
                moment2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = moment2.sqrt().add_(group['eps'])

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(moment1, denom, value=-step_size)

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss

def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return ((t/T_w) * a_max)
    elif t < T_c:
        return (a_min + 0.5 * (1 + math.cos(((t - T_w)/(T_c - T_w))* math.pi)) * (a_max - a_min))
    else:
        return (a_min)

def gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return

    total_norm_sq = sum(grad.norm(2).item() ** 2 for grad in grads)
    total_norm = math.sqrt(total_norm_sq)
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = np.stack([x[i : i + context_length] for i in starts])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets


def save_checkpoint(model, adamw_opt, muon_opt, iteration, out_path):
    checkpoint = {
        'model_state': model.state_dict(),
        'adamw_state': adamw_opt.state_dict(),
        'muon_state': muon_opt.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out_path)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']