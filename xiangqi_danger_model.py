from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class DangerHeadConfig:
    in_channels: int = 14
    board_h: int = 10
    board_w: int = 9
    channels: int = 96
    blocks: int = 5
    out_dim: int = 5
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.blocks < 1:
            raise ValueError("blocks must be >= 1")
        if self.out_dim < 1:
            raise ValueError("out_dim must be >= 1")


class _DangerResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        padding = int(dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class CNNActionDangerHead(nn.Module):
    """Small action-conditioned tactical danger head.

    The input is the board after a candidate move has been applied and encoded
    from the opponent-to-move canonical perspective.  The head does not play
    Xiangqi directly; it predicts whether the opponent has a tactical
    refutation in that resulting position.
    """

    target_names = (
        "opponent_mate1",
        "opponent_mate2",
        "opponent_forcing_check",
        "value_collapse",
        "tactical_refuted",
    )

    def __init__(self, config: DangerHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or DangerHeadConfig()
        channels = int(self.config.channels)
        groups = 8 if channels % 8 == 0 else 1
        self.input = nn.Conv2d(int(self.config.in_channels), channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            _DangerResidualBlock(channels, dilation=1 + (idx % 3))
            for idx in range(int(self.config.blocks))
        )
        self.context_fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(channels, int(self.config.out_dim)),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected input [B,C,H,W], got {tuple(x.shape)}")
        x = x[:, : int(self.config.in_channels), :, :].float()
        h = self.input(x)
        for block in self.blocks:
            h = block(h)

        rank_context = h.mean(dim=3, keepdim=True).expand_as(h)
        file_context = h.mean(dim=2, keepdim=True).expand_as(h)
        h = self.context_fuse(torch.cat([h, rank_context, file_context], dim=1))
        h = self.dropout(F.gelu(self.norm(h)))

        pooled_mean = h.mean(dim=(2, 3))
        pooled_max = h.amax(dim=(2, 3))
        logits = self.head(torch.cat([pooled_mean, pooled_max], dim=1))
        risk_logit = logits[:, 4]
        if logits.shape[1] >= 3:
            risk_logit = risk_logit + 0.35 * logits[:, 1] + 0.25 * logits[:, 2]
        return {
            "danger_logits": logits,
            "risk_logit": risk_logit,
        }


def save_danger_checkpoint(
    path: str,
    *,
    model: CNNActionDangerHead,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "target_names": list(CNNActionDangerHead.target_names),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_danger_checkpoint(path: str, map_location: str | torch.device = "cpu") -> CNNActionDangerHead:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"unsupported danger checkpoint format: {path}")
    config = DangerHeadConfig(**dict(payload.get("model_config") or {}))
    model = CNNActionDangerHead(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


__all__ = [
    "CNNActionDangerHead",
    "DangerHeadConfig",
    "load_danger_checkpoint",
    "save_danger_checkpoint",
]
