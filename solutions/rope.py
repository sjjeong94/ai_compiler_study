import os
from typing import Union

import torch
import triton
import triton.language as tl


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def rope_torch(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[s, b, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """
    # if fused:
    #    assert (
    #        tensor_format != "thd" or cu_seqlens is not None
    #    ), "cu_seqlens must not be None when tensor_format is 'thd'."
    #    return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), "Only formats `sbhd` or `bshd` are supported for input tensor `t` " f"when fused is False, got {tensor_format}."

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


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
def rope_kernel(
    t_ptr,
    f_ptr,
    o_ptr,
    t_s_stride,
    f_s_stride,
    o_s_stride,
    bh,
    d,
    BLOCK_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    t_start_ptr = t_ptr + row_idx * t_s_stride
    f_start_ptr = f_ptr + row_idx * f_s_stride
    o_start_ptr = o_ptr + row_idx * o_s_stride

    col_offsets = tl.arange(0, BLOCK_M)
    f_ptrs = f_start_ptr + col_offsets

    f = tl.load(f_ptrs)

    cos_ = tl.math.cos(f)
    sin_ = tl.math.sin(f)
    d_half = d // 2
    t_ptrs = t_start_ptr + bh_idx * BLOCK_M + col_offsets
    o_ptrs = o_start_ptr + bh_idx * BLOCK_M + col_offsets
    t = tl.load(t_ptrs)

    t0 = tl.load(t_ptrs - d_half, mask=col_offsets >= d_half, other=0.0)
    t1 = tl.load(t_ptrs + d_half, mask=col_offsets < d_half, other=0.0)
    t_rotated = t0 - t1

    y = t * cos_ + t_rotated * sin_
    tl.store(o_ptrs, y)


def rope(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[0]
    assert cur_seq_len <= max_seq_len
    freqs = freqs[:cur_seq_len]
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    s, b, h, d = t.shape
    BLOCK_M = triton.next_power_of_2(d)
    num_warps = 4
    if BLOCK_M >= 2048:
        num_warps = 8
    if BLOCK_M >= 4096:
        num_warps = 16

    o = torch.empty_like(t)
    bh = b * h

    rope_kernel[(s, bh)](
        t,
        freqs,
        o,
        t.stride(0),
        freqs.stride(0),
        o.stride(0),
        bh,
        d,
        num_warps=num_warps,
        BLOCK_M=BLOCK_M,
    )

    return torch.cat((o, t_pass), dim=-1)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],  # Argument names to use as an x-axis for the plot
        x_vals=[64 * i for i in range(1, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["torch", "torch-jit", "triton"],
        # Label name for the lines
        line_names=["torch", "torch-jit", "triton"],
        # Line styles
        styles=[("green", "--"), ("green", "-"), ("blue", "-")],
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="rope-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(S, provider):
    s, b, h, d = S, 16, 8, 256
    s2, d2 = 2048, 256
    t = torch.randn([s, b, h, d], device="cuda")
    freqs = torch.randn([s2, 1, 1, d2], device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_torch(t, freqs), quantiles=quantiles)
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_torch_jit(t, freqs), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope(t, freqs), quantiles=quantiles)
    gbps = lambda ms: 2 * t.nelement() * t.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


torch.manual_seed(0)
s, b, h, d = 32, 16, 12, 256
s2, d2 = 48, 256
t = torch.randn([s, b, h, d], device="cuda")
freqs = torch.randn([s2, 1, 1, d2], device="cuda")

triton_output = rope(t, freqs)
torch_output = rope_torch(t, freqs)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


os.makedirs("./results", exist_ok=True)
benchmark.run(show_plots=True, print_data=True, save_path="./results")
