import os

import torch
import triton
from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from aicom import rope_fwd


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