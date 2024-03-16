import os

import torch
import triton
import triton.language as tl
from transformer_engine.pytorch.attention import apply_rotary_pos_emb


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def rope_torch_jit(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[0]

    assert cur_seq_len <= max_seq_len

    freqs = freqs[:cur_seq_len]

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],  # Argument names to use as an x-axis for the plot
        x_vals=[64 * i for i in range(1, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["fused", "torch-jit", "triton"],
        # Label name for the lines
        line_names=["fused", "torch-jit", "triton"],
        # Line styles
        styles=[("green", "--"), ("green", "-"), ("blue", "-")],
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="rope-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(seq_len, provider):
    s, b, h, d = seq_len, 16, 8, 256
    s2, d2 = 4096, 256
    t = torch.randn([s, b, h, d], device="cuda")
    freqs = torch.randn([s2, 1, 1, d2], device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    if provider == "fused":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(t, freqs, fused=True), quantiles=quantiles)
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_torch_jit(t, freqs), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_fwd(t, freqs), quantiles=quantiles)
    gbps = lambda ms: 2 * t.nelement() * t.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


torch.manual_seed(0)
s, b, h, d = 32, 16, 12, 1024
s2, d2 = 48, 768
t = torch.randn([s, b, h, d], device="cuda")
freqs = torch.randn([s2, 1, 1, d2], device="cuda")

triton_output = rope_fwd(t, freqs)
torch_output = apply_rotary_pos_emb(t, freqs)
if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


os.makedirs("./results", exist_ok=True)
benchmark.run(show_plots=True, print_data=True, save_path="./results")
