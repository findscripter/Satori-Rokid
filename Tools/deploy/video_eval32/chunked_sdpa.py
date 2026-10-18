"""Equivalent independent-cu_seqlens SDPA without a global quadratic mask.

Upstream vision SDPA attends only inside diagonal q_cu_seqlens blocks. Computing
each block separately preserves those visibility boundaries and all q/k/v data.
Kernel selection and floating-point rounding may differ from the dense path.
"""


def chunked_sdpa_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    import torch
    import torch.nn.functional as functional
    if q_cu_seqlens is None or k_cu_seqlens is None:
        raise ValueError("explicit_native_cu_seqlens_required")
    q_bounds, k_bounds = q_cu_seqlens.tolist(), k_cu_seqlens.tolist()
    if q_bounds != k_bounds or len(q_bounds) < 2 or q_bounds[0] != 0:
        raise ValueError("native_self_attention_blocks_must_match")
    if q_bounds[-1] != q.shape[0] or k_bounds[-1] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("native_block_lengths_must_cover_all_tokens")
    if any(a >= b for a, b in zip(q_bounds, q_bounds[1:])):
        raise ValueError("native_blocks_must_be_nonempty")
    output = []
    for start, end in zip(q_bounds, q_bounds[1:]):
        block = functional.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k[start:end].transpose(0, 1), v[start:end].transpose(0, 1),
            dropout_p=0.0,
        )
        output.append(block.transpose(0, 1))
    return torch.cat(output, dim=0).reshape(q.shape[0], -1)
