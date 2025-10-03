import torch
from einops import rearrange, einsum, reduce, repeat
import math

class FlashAttentionForwardNoTriton(torch.autograd.Function):
    @staticmethod
    @torch.compile()
    def forward(ctx, Q, K, V, is_causal=False):

        input_dtype = Q.dtype
        N_q, d = Q.shape[-2], Q.shape[-1] 
        N_k = K.shape[-2]
        batch_dims = Q.shape[:-2] 
        
        B_q = 16
        B_k = 16
        
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)
        
        def batched_diag(x):
            batch_dims = x.shape[:-1]
            last_dim = x.shape[-1]
            result = torch.zeros(*batch_dims, last_dim, last_dim, device=x.device, dtype=x.dtype)
            indices = torch.arange(last_dim, device=x.device)
            result[..., indices, indices] = x
            return result
        
        Q_tiled = rearrange(Q, '... (T_q B_q) d -> ... T_q B_q d', B_q=B_q)
        K_tiled = rearrange(K, '... (T_k B_k) d -> ... T_k B_k d', B_k=B_k)
        V_tiled = rearrange(V, '... (T_k B_k) d -> ... T_k B_k d', B_k=B_k)
        
        Os = []
        Ls = []
        for i in range(T_q):
            Q_i = Q_tiled[..., i, :, :]
            O_i = torch.zeros(*batch_dims, B_q, d, device=Q.device, dtype=input_dtype)
            l_i = torch.zeros(*batch_dims, B_q, device=Q.device, dtype=input_dtype)
            m_i = torch.full((*batch_dims, B_q), float('-inf'), device=Q.device, dtype=input_dtype)
            
            for j in range(T_k):
                K_j = K_tiled[..., j, :, :]
                V_j = V_tiled[..., j, :, :]
                
                S_ij = einsum(Q_i, K_j, '... B_q d,... B_k d->... B_q B_k') / math.sqrt(d)
                
                m_i_prev = m_i.clone()
                
                m_i = torch.maximum(m_i, torch.max(S_ij, dim=-1)[0])
                
                P_i = torch.exp(S_ij - m_i.unsqueeze(-1))
                
                l_i = torch.exp(m_i_prev - m_i) * l_i + torch.sum(P_i, dim=-1)
                
                diag_scale = batched_diag(torch.exp(m_i_prev - m_i))

                term_1 = einsum(diag_scale, O_i, "... B_q B_q , ... B_q d -> ... B_q d")
                term_2 = einsum(P_i, V_j, "... B_q B_k, ... B_k d -> ... B_q d")
                
                O_i = term_1 + term_2
            
            l_i_inv_diag = batched_diag(1.0 / l_i)
            O_i = einsum(l_i_inv_diag, O_i, '... B_q B_q, ... B_q d->... B_q d')
            
            L_i = m_i + torch.log(l_i)
            
            Os.append(O_i)
            Ls.append(L_i)
        
        O = torch.cat([o.view(*batch_dims, B_q, d) for o in Os], dim=-2)
        L = torch.cat([L.view(*batch_dims, B_q) for L in Ls], dim=-1)
        O = O.to(input_dtype)
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    @torch.compile()
    def backward(ctx, grad_output):

        L, Q, K, V, O = ctx.saved_tensors

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