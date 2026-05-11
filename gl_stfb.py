"""Global-Local Spatio-Temporal Feature Extraction Backbone (GL-STFB)."""

from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class LMSM(nn.Sequential):
    """Local Multi-Scale Spatio-Temporal Encoding Module"""

    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        conv1_kernel_size: int,
        conv2_kernel_size: int,
        depth_multiplier: int,
        pool1_size: int,
        pool2_size: int,
        drop_prob: float,
        activation: Type[nn.Module] = nn.ELU,
    ) -> None:
        spatial_filters = n_filters * depth_multiplier
        super().__init__(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(1, conv1_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(n_filters),
            nn.Conv2d(
                in_channels=n_filters,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                groups=n_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool1_size)),
            nn.Dropout(drop_prob),
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, conv2_kernel_size),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            activation(),
            nn.AvgPool2d(kernel_size=(1, pool2_size)),
            nn.Dropout(drop_prob),
        )


class _LearnableSequenceBias(nn.Module):
    """Trainable sequence bias used by the Global Temporal Module."""

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class GTM(nn.Module):
    """Global Temporal Module."""

    def __init__(
        self,
        seq_length: int,
        d_model: int,
        num_heads: int,
        ffn_expansion_factor: float,
        drop_prob: float = 0.5,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = _LearnableSequenceBias(seq_length + 1, d_model)

        ffn_dim = int(d_model * ffn_expansion_factor)
        self.input_dropout = nn.Dropout(drop_prob)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=drop_prob,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position(x)
        x = self.input_dropout(x)
        x = self.encoder(x)
        return x[:, 0]


class _ScaleEncoder(nn.Module):
    """Scale Encoder: cross-scale attention and branch-wise split."""

    def __init__(
        self,
        n_branches: int,
        d_model_per_branch: int,
        cross_scale_attn_heads: int = 2,
        drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.d_model_per_branch = d_model_per_branch

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model_per_branch,
            num_heads=min(cross_scale_attn_heads, d_model_per_branch),
            dropout=drop_prob,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model_per_branch)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        stacked = torch.stack(x_list, dim=2)
        bsz, seq_len, n_branches, d_model = stacked.shape
        x_flat = stacked.reshape(bsz * seq_len, n_branches, d_model)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        out_flat = self.norm(x_flat + self.dropout(attn_out))
        out = out_flat.reshape(bsz, seq_len, n_branches, d_model)
        out_list = [out[:, :, i, :] for i in range(n_branches)]
        x_cross = torch.cat(out_list, dim=2)
        split = [
            x_cross[:, :, i * self.d_model_per_branch : (i + 1) * self.d_model_per_branch]
            for i in range(self.n_branches)
        ]
        return split


class _DGASW(nn.Module):
    """Dual-Granularity Adaptive Scale Weight (DGASW)."""

    def __init__(
        self,
        n_branches: int,
        d_model_per_branch: int,
        hidden_ratio: float = 0.25,
        drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        d_total = n_branches * d_model_per_branch
        hidden = max(1, int(d_total * hidden_ratio))
        self.query_net = nn.Sequential(
            nn.Linear(d_total, hidden),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden, n_branches),
        )
        self.dropout = nn.Dropout(drop_prob)
        self.arch_alphas = nn.Parameter(torch.zeros(n_branches))

    def get_modal_weights(self) -> torch.Tensor:
        return F.softmax(self.arch_alphas, dim=-1)

    def forward(self, split: list[torch.Tensor]) -> torch.Tensor:
        # Input-adaptive sample fusion weights.
        concat = torch.cat(split, dim=2)
        pooled = concat.mean(dim=1)
        attn_logits = self.query_net(self.dropout(pooled))  # (B, n_branches)
        w_sample = F.softmax(attn_logits, dim=-1)

        # modal weights.
        w_modal = self.get_modal_weights()  # (n_branches,)

        # Combine sample and modal weights, then normalize.
        w_combined = w_sample * w_modal.unsqueeze(0)
        w_combined = w_combined / (w_combined.sum(dim=-1, keepdim=True) + 1e-8)

        weighted = [
            split[i] * w_combined[:, i : i + 1].unsqueeze(1)
            for i in range(self.n_branches)
        ]
        return torch.cat(weighted, dim=2)


class CSIM(nn.Module):
    """Cross-Scale Interaction Module."""
    def __init__(
        self,
        n_branches: int,
        d_model_per_branch: int,
        cross_scale_attn_heads: int = 2,
        hidden_ratio: float = 0.25,
        drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale_encoder = _ScaleEncoder(
            n_branches=n_branches,
            d_model_per_branch=d_model_per_branch,
            cross_scale_attn_heads=cross_scale_attn_heads,
            drop_prob=drop_prob,
        )
        self.dgasw = _DGASW(
            n_branches=n_branches,
            d_model_per_branch=d_model_per_branch,
            hidden_ratio=hidden_ratio,
            drop_prob=drop_prob,
        )

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        split = self.scale_encoder(x_list)
        return self.dgasw(split)


class GL_STFB(EEGModuleMixin, nn.Module):
    """Global-Local Spatio-Temporal Feature Extraction Backbone."""
    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        n_filters_list: tuple[int, ...] = (9, 9, 9, 9),
        conv1_kernels_size: tuple[int, ...] = (15, 31, 63, 125),
        conv2_kernel_size: int = 15,
        depth_multiplier: int = 2,
        pool1_size: int = 8,
        pool2_size: int = 7,
        drop_prob: float = 0.2,
        num_heads: int = 8,
        ffn_expansion_factor: float = 2,
        att_drop_prob: float = 0.2,
        num_layers: int = 2,
        cross_scale_attn_heads: int = 2,
        activation: Type[nn.Module] = nn.ELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        assert len(n_filters_list) == len(conv1_kernels_size), (
            "The length of n_filters_list and conv1_kernel_sizes should be equal."
        )

        self.ensure_dim = Rearrange("batch chans time -> batch 1 chans time")
        self.lmsm = nn.ModuleList(
            [
                nn.Sequential(
                    LMSM(
                        self.n_chans,
                        n_filters_list[b],
                        conv1_kernels_size[b],
                        conv2_kernel_size,
                        depth_multiplier,
                        pool1_size,
                        pool2_size,
                        drop_prob,
                        activation,
                    ),
                    Rearrange("batch channels 1 time -> batch time channels"),
                )
                for b in range(len(n_filters_list))
            ]
        )

        lmsm_out = self._forward_lmsm()
        seq_len, d_model_total = lmsm_out.shape[1:3]
        n_branches = len(n_filters_list)
        d_model_per = d_model_total // n_branches

        self.csim = CSIM(
            n_branches=n_branches,
            d_model_per_branch=d_model_per,
            cross_scale_attn_heads=cross_scale_attn_heads,
            hidden_ratio=0.25,
            drop_prob=att_drop_prob,
        )
        self.gtm = GTM(
            seq_len,
            d_model_total,
            num_heads,
            ffn_expansion_factor,
            att_drop_prob,
            num_layers,
        )
        self.final_layer = nn.Linear(d_model_total, self.n_outputs)

    def _forward_lmsm(
        self, cat: bool = True
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        x = torch.randn(1, 1, self.n_chans, self.n_times)
        x = [tsconv(x) for tsconv in self.lmsm]
        if cat:
            x = torch.cat(x, dim=2)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure_dim(x)
        x_list = [tsconv(x) for tsconv in self.lmsm]
        x = self.csim(x_list)
        x = self.gtm(x)
        x = self.final_layer(x)
        return x
