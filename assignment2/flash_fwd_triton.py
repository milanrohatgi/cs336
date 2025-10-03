import torch
import triton
import triton.language as tl
from einops import rearrange, einsum, reduce, repeat
import math
@triton.jit

def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, 
    stride_qb, stride_qq, stride_qd, 
    stride_kb, stride_kk, stride_kd, 
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, 
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    
    o = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)

    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    query_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    
    for key_tile_index in range(n_key_tiles):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        s = tl.dot(q, tl.trans(k)) * scale

        if is_causal:
            key_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            
            mask = query_indices[:, None] >= key_indices[None, :]
            
            s = tl.where(mask, s, -1e6)


        m_prev = m

        m_new = tl.maximum(m, tl.max(s, axis=1))

        p = tl.exp(s - m_new[:, None])

        l_new = tl.exp(m_prev - m_new) * l + tl.sum(p, axis=1)

        o = o * tl.exp(m_prev - m_new)[:, None]

        o += tl.dot(p.to(v.dtype), v)

        m = m_new
        l = l_new

    o = o / l[:, None]

    L_i = m + tl.log(l)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tl.store(O_block_ptr, o.to(O_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_ptr.type.element_ty))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        input_dtype = Q.dtype
        batch_dims = Q.shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
        N_q, d = Q.shape[-2], Q.shape[-1]
        N_k = K.shape[-2]
        
        Q_3d = Q.reshape(batch_size, N_q, d)
        K_3d = K.reshape(batch_size, N_k, d)
        V_3d = V.reshape(batch_size, N_k, d)
        
        Q_TILE_SIZE = min(16, N_q) 
        K_TILE_SIZE = min(16, N_k) 

        scale = 1.0 / (d ** 0.5)
        
        O = torch.empty_like(Q_3d)
        L = torch.empty((batch_size, N_q), dtype=input_dtype, device=Q.device)
        
        stride_qb, stride_qq, stride_qd = Q_3d.stride()
        stride_kb, stride_kk, stride_kd = K_3d.stride()
        stride_vb, stride_vk, stride_vd = V_3d.stride()
        stride_ob, stride_oq, stride_od = O.stride()
        stride_lb, stride_lq = L.stride()
        
        grid = (triton.cdiv(N_q, Q_TILE_SIZE), batch_size)
        
        flash_fwd_kernel[grid](
            Q_3d, K_3d, V_3d,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_q, N_k,
            scale,
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )
        
        O = O.reshape(*batch_dims, N_q, d)

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        
        return O
    
    @staticmethod
    @torch.compile()
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch_dims = Q.shape[:-2]
        N_q, d = Q.shape[-2], Q.shape[-1]
        N_k = K.shape[-2]
        
        pattern = '... n d -> (...) n d'
        Q_flat = rearrange(Q, pattern)
        K_flat = rearrange(K, pattern)
        V_flat = rearrange(V, pattern)
        O_flat = rearrange(O, pattern)
        dO_flat = rearrange(grad_output, pattern)
        L_flat = rearrange(L, '... n -> (...) n')

        dQ_flat = torch.zeros_like(Q_flat)
        dK_flat = torch.zeros_like(K_flat)
        dV_flat = torch.zeros_like(V_flat)
        
        scale = 1.0 / math.sqrt(d)

        for b in range(Q_flat.shape[0]):
            D = reduce(O_flat[b] * dO_flat[b], 'n d -> n', 'sum')

            QKt = einsum(Q_flat[b], K_flat[b], "... q d, ... k d -> q k")
            S = QKt * scale  

            if is_causal:
                mask = torch.triu(torch.ones(N_q, N_k, device=Q.device), diagonal=1).bool()
                S.masked_fill_(mask, -float('inf'))
            

            L_expanded = repeat(L_flat[b], 'n -> n k', k=N_k)
            P = torch.exp(S - L_expanded)

            dV_flat[b] = einsum(P, dO_flat[b], 'n k, n d-> k d')

            dP = einsum(dO_flat[b], V_flat[b], 'n d , k d-> n k')

            D_expanded = repeat(D, 'n -> n k', k=N_k)

            dS = P * (dP - D_expanded)

            dQ_flat[b] = einsum(dS, K_flat[b], 'n k,k d -> n d') * scale

            dK_flat[b] = einsum(dS, Q_flat[b], 'n k, n d -> k d') * scale
        
        dQ = dQ_flat.reshape(*batch_dims, N_q, d)
        dK = dK_flat.reshape(*batch_dims, N_k, d)
        dV = dV_flat.reshape(*batch_dims, N_k, d)
        return dQ, dK, dV, None