import torch
import math
from typing import Tuple


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(Q.shape)
        B_q = max(16, Q.size(0) // 32)
        B_k = max(16, K.size(0) // 32)

        dim = Q.size(1)
        T_q = math.ceil(Q.size(0) / B_q)
        T_k = math.ceil(K.size(0) / B_k)

        Q_tiles = [Q[i * B_q : (i + 1) * B_q] for i in range(T_q)]
        K_tiles = [K[i * B_k : (i + 1) * B_k] for i in range(T_k)]
        V_tiles = [V[i * B_k : (i + 1) * B_k] for i in range(T_k)]

        O_tiles = []
        L_tiles = []

        for Qi in Q_tiles:
            O_i = torch.zeros(B_q, dim) # Qi.size(0)
            l_i = torch.zeros(B_q)
            m_i = torch.full((B_q,), float('-inf'))

            for Kj, Vj in zip(K_tiles, V_tiles):
                # Kj.size(0)
                Sij = Qi @ Kj.T / math.sqrt(dim)
                rowmax_S_ij = Sij.max(dim=1).values

                m_prev = m_i.clone()
                m_i = torch.maximum(m_i, rowmax_S_ij)

                P_tilde_ij = torch.exp(Sij - m_i.unsqueeze(1))
                l_i = torch.exp(m_prev - m_i) * l_i + P_tilde_ij.sum(dim=1)
                O_i = torch.exp(m_prev - m_i).unsqueeze(1) * O_i + P_tilde_ij @ Vj
            
            O_i = O_i / l_i.unsqueeze(1)
            L_i = m_i + torch.log(l_i)

            O_tiles.append(O_i)
            L_tiles.append(L_i)

        O = torch.cat(O_tiles, dim=0)
        L = torch.cat(L_tiles, dim=0)
        return O, L


    @staticmethod
    def backward(ctx, *grad_outputs):
        return NotImplementedError()
