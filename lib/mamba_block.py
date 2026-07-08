import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class DirectionalMambaSSM(nn.Module):
    """Drop-in replacement for `Attention`: (B, C, H, W) -> (B, C, H, W).

    Flattens the feature map into 4 sequences (row-major left->right,
    its reverse, column-major top->bottom, its reverse), runs each through
    the same shared Mamba core (VMamba-style SS2D), folds each back to
    (B, C, H, W), and averages the 4 results.
    """

    def __init__(self, channels, proj_drop=0.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for DirectionalMambaSSM but is not installed. "
                "Install with: pip install causal-conv1d>=1.2.0 mamba-ssm (Linux/CUDA only)."
            )
        self.proj_drop = proj_drop
        self.core = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C, H, W = x.shape

        seq_row = x.flatten(2).transpose(1, 2)                     # left->right, top->bottom
        seq_col = x.transpose(2, 3).flatten(2).transpose(1, 2)     # top->bottom, left->right

        directions = [seq_row, seq_row.flip(dims=[1]), seq_col, seq_col.flip(dims=[1])]
        batched = torch.cat(directions, dim=0).contiguous()         # (4B, L, C) -- single core call

        y = self.core(batched)

        outs = list(y.chunk(4, dim=0))
        for i in range(len(outs)):
            if i % 2 == 1:
                outs[i] = outs[i].flip(dims=[1])
            if i < 2:
                outs[i] = outs[i].transpose(1, 2).reshape(B, C, H, W)
            else:
                outs[i] = outs[i].transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

        out = torch.stack(outs, dim=0).mean(dim=0)
        out = F.dropout(out, self.proj_drop)
        return out
