"""Round-Robin Cross-Attention Fusion Module (RCFM)."""

import torch
from torch.nn import Module

from gl_stfb import GL_STFB


NUM_CLASSES = 2
FUSION_NUM_HEADS = 16
FUSION_DROPOUT = 0.1
FUSION_NUM_STAGES = 4


def _register_feature_hook(model: Module) -> None:
    def _hook(_module, inp):
        model._last_features = inp[0]

    model._last_features = None
    model._hook_handle = model.final_layer.register_forward_pre_hook(_hook)


class R2F_Layer(Module):
    _NUM_MODALITIES = 4

    def _align_d_model_for_heads(self, d_raw: int, num_heads: int) -> tuple[int, int]:
        h = max(1, min(num_heads, d_raw))
        if d_raw % h == 0:
            return d_raw, h
        d_model = d_raw + (h - (d_raw % h))
        return d_model, h

    def __init__(
        self,
        modal_dims: tuple[int, int, int, int],
        n_outputs: int,
        num_heads: int = FUSION_NUM_HEADS,
        dropout: float = FUSION_DROPOUT,
        num_stages: int = FUSION_NUM_STAGES,
    ):
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        d_raw = max(modal_dims)
        d_model, nhead = self._align_d_model_for_heads(d_raw, num_heads)
        self.d_model = d_model
        self.num_stages = num_stages

        def _proj(in_dim: int) -> Module:
            if in_dim == d_model:
                return torch.nn.Identity()
            return torch.nn.Linear(in_dim, d_model)

        self.proj_eeg = _proj(modal_dims[0])
        self.proj_ecg = _proj(modal_dims[1])
        self.proj_eda = _proj(modal_dims[2])
        self.proj_gaze = _proj(modal_dims[3])

        self.cross_attn_stages = torch.nn.ModuleList()
        self.norms_stages = torch.nn.ModuleList()
        for _ in range(num_stages):
            self.cross_attn_stages.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.MultiheadAttention(
                            embed_dim=d_model,
                            num_heads=nhead,
                            dropout=dropout,
                            batch_first=True,
                        )
                        for _ in range(self._NUM_MODALITIES)
                    ]
                )
            )
            self.norms_stages.append(
                torch.nn.ModuleList(
                    [torch.nn.LayerNorm(d_model) for _ in range(self._NUM_MODALITIES)]
                )
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(self._NUM_MODALITIES * d_model, n_outputs)

    def forward(
        self,
        feat_eeg: torch.Tensor,
        feat_ecg: torch.Tensor,
        feat_eda: torch.Tensor,
        feat_gaze: torch.Tensor,
    ) -> torch.Tensor:
        pe = self.proj_eeg(feat_eeg)
        pc = self.proj_ecg(feat_ecg)
        pd = self.proj_eda(feat_eda)
        pg = self.proj_gaze(feat_gaze)
        features = [pe, pc, pd, pg]

        for stage in range(self.num_stages):
            fused_blocks: list[torch.Tensor] = []
            for i in range(self._NUM_MODALITIES):
                q = features[i].unsqueeze(1)
                others = [features[j] for j in range(self._NUM_MODALITIES) if j != i]
                kv = torch.cat([t.unsqueeze(1) for t in others], dim=1)
                attn_out, _ = self.cross_attn_stages[stage][i](q, kv, kv)
                fused_i = self.norms_stages[stage][i](
                    self.dropout(attn_out.squeeze(1)) + features[i]
                )
                fused_blocks.append(fused_i)
            features = fused_blocks

        return self.head(torch.cat(features, dim=1))


class RCFM(Module):
    def __init__(
        self,
        n_eeg_chans: int,
        n_ecg_chans: int,
        n_eda_chans: int,
        n_gaze_chans: int,
        n_eeg_times: int,
        n_ecg_times: int,
        n_eda_times: int,
        n_gaze_times: int,
        n_outputs: int = NUM_CLASSES,
        fusion_num_heads: int = FUSION_NUM_HEADS,
        fusion_dropout: float = FUSION_DROPOUT,
        fusion_num_stages: int = FUSION_NUM_STAGES,
    ):
        super().__init__()
        self.eeg_encoder = GL_STFB(
            n_chans=n_eeg_chans,
            n_outputs=n_outputs,
            n_times=n_eeg_times,
        )
        self.ecg_encoder = GL_STFB(
            n_chans=n_ecg_chans,
            n_outputs=n_outputs,
            n_times=n_ecg_times,
        )
        self.eda_encoder = GL_STFB(
            n_chans=n_eda_chans,
            n_outputs=n_outputs,
            n_times=n_eda_times,
        )
        self.gaze_encoder = GL_STFB(
            n_chans=n_gaze_chans,
            n_outputs=n_outputs,
            n_times=n_gaze_times,
        )

        _register_feature_hook(self.eeg_encoder)
        _register_feature_hook(self.ecg_encoder)
        _register_feature_hook(self.eda_encoder)
        _register_feature_hook(self.gaze_encoder)

        feat_eeg = self.eeg_encoder.final_layer.in_features
        feat_ecg = self.ecg_encoder.final_layer.in_features
        feat_eda = self.eda_encoder.final_layer.in_features
        feat_gaze = self.gaze_encoder.final_layer.in_features
        self.modal_fusion = R2F_Layer(
            modal_dims=(feat_eeg, feat_ecg, feat_eda, feat_gaze),
            n_outputs=n_outputs,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
            num_stages=fusion_num_stages,
        )

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_ecg: torch.Tensor,
        x_eda: torch.Tensor,
        x_gaze: torch.Tensor,
    ) -> torch.Tensor:
        _ = self.eeg_encoder(x_eeg)
        _ = self.ecg_encoder(x_ecg)
        _ = self.eda_encoder(x_eda)
        _ = self.gaze_encoder(x_gaze)
        feat_eeg = self.eeg_encoder._last_features
        feat_ecg = self.ecg_encoder._last_features
        feat_eda = self.eda_encoder._last_features
        feat_gaze = self.gaze_encoder._last_features
        return self.modal_fusion(feat_eeg, feat_ecg, feat_eda, feat_gaze)
