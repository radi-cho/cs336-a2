import torch
import math
from typing import Tuple, Optional

import triton
import triton.language as tl

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = Q.shape[0]
        N_q, N_k = Q.shape[1], K.shape[1]
        dim = Q.shape[2]

        B_q = max(16, N_q // 32)
        B_k = max(16, N_k // 32)

        Q_tiles = [Q[:, i * B_q : (i + 1) * B_q, :] for i in range(math.ceil(N_q / B_q))]
        K_tiles = [K[:, i * B_k : (i + 1) * B_k, :] for i in range(math.ceil(N_k / B_k))]
        V_tiles = [V[:, i * B_k : (i + 1) * B_k, :] for i in range(math.ceil(N_k / B_k))]

        O_tiles = []
        L_tiles = []

        for Qi in Q_tiles:
            B_q_i = Qi.shape[1]
            O_i = torch.zeros((batch_size, B_q_i, dim), device=Q.device)
            l_i = torch.zeros((batch_size, B_q_i), device=Q.device)
            m_i = torch.full((batch_size, B_q_i), float('-inf'), device=Q.device)

            for Kj, Vj in zip(K_tiles, V_tiles):
                Sij = torch.matmul(Qi, Kj.transpose(-1, -2)) / math.sqrt(dim)
                rowmax_S_ij = Sij.max(dim=2).values

                m_prev = m_i.clone()
                m_i = torch.maximum(m_i, rowmax_S_ij)

                P_tilde_ij = torch.exp(Sij - m_i.unsqueeze(-1))
                l_i = torch.exp(m_prev - m_i) * l_i + P_tilde_ij.sum(dim=2)
                O_i = torch.exp(m_prev - m_i).unsqueeze(-1) * O_i + torch.matmul(P_tilde_ij, Vj)

            O_i = O_i / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i)

            O_tiles.append(O_i)
            L_tiles.append(L_i)

        O = torch.cat(O_tiles, dim=1)
        L = torch.cat(L_tiles, dim=1)

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    @torch.compile
    def backward(
        ctx,
        dO: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        Q, K, V, O, L = ctx.saved_tensors
        D = torch.sum(O * dO, dim=2)

        _, N_q, dim = Q.shape
        N_k = K.shape[1]

        S = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dim)
        if ctx.is_causal:
            mask = torch.triu(torch.ones(N_q, N_k, device=Q.device, dtype=torch.bool), diagonal=1)
            S = S.masked_fill(mask.unsqueeze(0), float('-inf'))

        P = P = torch.exp(S - L.unsqueeze(-1))
        dV = torch.matmul(P.transpose(-1, -2), dO)
        dP = torch.matmul(dO, V.transpose(-1, -2))
        dS = P * (dP - D.unsqueeze(-1))
        dQ = torch.matmul(dS, K) / math.sqrt(dim)
        dK = torch.matmul(dS.transpose(-1, -2), Q) / math.sqrt(dim)

        return dQ, dK, dV, None


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

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

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

    q = tl.load(Q_block_ptr).to(tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    for k_tile_start in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(K_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)

        S = tl.dot(q, k.T) * scale
        if is_causal:
            offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            offs_k = k_tile_start + tl.arange(0, K_TILE_SIZE)
            mask = offs_k[None, :] > offs_q[:, None]
            S = tl.where(mask, S - 1e6, S)

        m_ij = tl.max(S, axis=1)
        m_new = tl.maximum(m, m_ij)

        P = tl.exp(S - m_new[:, None])
        l_new = tl.exp(m - m_new) * l + tl.sum(P, axis=1)
        o = tl.exp(m - m_new)[:, None] * o + tl.dot(P.to(v.dtype), v)

        l = l_new
        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    l = m + tl.log(l)

    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l.to(L_block_ptr.type.element_ty))


class FlashAttentionTriton(FlashAttention):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_q, D = Q.shape
        N_k = K.shape[1]

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        O = torch.empty_like(Q)
        L = torch.empty(B, N_q, dtype=Q.dtype, device=Q.device)

        def round_to_16(x):
            return ((x + 15) // 16) * 16

        BLOCK_M = round_to_16(N_q)
        BLOCK_N = round_to_16(N_k)

        grid = (triton.cdiv(N_q, BLOCK_M), B)
        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, scale,
            D=D,
            Q_TILE_SIZE=BLOCK_M,
            K_TILE_SIZE=BLOCK_N,
            is_causal=is_causal
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        return O
