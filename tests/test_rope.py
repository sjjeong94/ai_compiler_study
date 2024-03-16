import pytest
import torch

from aicom import rope_fwd


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def rope_ref(
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


@pytest.mark.parametrize(
    "t_shape, f_shape",
    ([[32, 16, 12, 1024], [48, 1, 1, 768]],),
)
def test_rope_fwd(t_shape, f_shape):
    t = torch.randn(t_shape, device="cuda")
    freqs = torch.randn(f_shape, device="cuda")
    output = rope_fwd(t, freqs)
    output_ref = rope_ref(t, freqs)
    torch.testing.assert_close(output_ref, output, atol=1e-3, rtol=0)
