import torch
import triton
import triton.language as tl


@triton.jit
def rope_fwd_kernel(
    t_ptr,
    f_ptr,
    o_ptr,
    t_s_stride,
    f_s_stride,
    o_s_stride,
    d,
    d2,
    BLOCK_M: tl.constexpr,
):
    s_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    t_start_ptr = t_ptr + s_idx * t_s_stride
    f_start_ptr = f_ptr + s_idx * f_s_stride
    o_start_ptr = o_ptr + s_idx * o_s_stride

    d2_half = d2 // 2
    col_offsets = tl.arange(0, BLOCK_M)
    mask = col_offsets < d2_half
    f0_ptrs = f_start_ptr + col_offsets
    f1_ptrs = f_start_ptr + col_offsets + d2_half
    f0 = tl.load(f0_ptrs, mask=mask, other=0.0)
    cos0 = tl.cos(f0)
    sin0 = tl.sin(f0)
    f1 = tl.load(f1_ptrs, mask=mask, other=0.0)
    cos1 = tl.cos(f1)
    sin1 = tl.sin(f1)

    t0_ptrs = t_start_ptr + bh_idx * d + col_offsets
    t1_ptrs = t_start_ptr + bh_idx * d + col_offsets + d2_half

    t0 = tl.load(t0_ptrs, mask=mask, other=0.0)
    t1 = tl.load(t1_ptrs, mask=mask, other=0.0)

    o0 = t0 * cos0 - t1 * sin0
    o1 = t0 * sin1 + t1 * cos1

    o0_ptrs = o_start_ptr + bh_idx * d + col_offsets
    o1_ptrs = o_start_ptr + bh_idx * d + col_offsets + d2_half
    tl.store(o0_ptrs, o0, mask=mask)
    tl.store(o1_ptrs, o1, mask=mask)

    if d2 < d:
        remainder = d - d2
        t2_ptrs = t_start_ptr + bh_idx * d + col_offsets + d2
        o2_ptrs = o_start_ptr + bh_idx * d + col_offsets + d2
        mask = col_offsets < remainder
        t2 = tl.load(t2_ptrs, mask=mask, other=0.0)
        tl.store(o2_ptrs, t2, mask=mask)


def rope_fwd(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[0]
    assert cur_seq_len <= max_seq_len
    freqs = freqs[:cur_seq_len]
    d2 = freqs.shape[-1]

    s, b, h, d = t.shape
    BLOCK_M = triton.next_power_of_2(d2 // 2)
    num_warps = 4
    if BLOCK_M >= 2048:
        num_warps = 8
    if BLOCK_M >= 4096:
        num_warps = 16

    o = torch.empty_like(t)
    bh = b * h

    rope_fwd_kernel[(s, bh)](
        t,
        freqs,
        o,
        t.stride(0),
        freqs.stride(0),
        o.stride(0),
        d,
        d2,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    return o
