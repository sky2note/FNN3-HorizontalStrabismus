"""
utils.losses
────────────
自定义损失函数集合（目前仅包含多标签 Focal Loss）。

Focal Loss (Lin et al. ICCV 2017) 能在类别不平衡场景下
专注 Hard 样本，提高少数类召回率。

公式：
    FL(p_t) = - α · (1 - p_t)^γ · log(p_t)
其中
    p_t = p      (y = 1)
    p_t = 1 - p  (y = 0)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    多标签 BCE 版本 Focal Loss

    Parameters
    ----------
    alpha : float | list[float]
        正样本调权因子 (>=0)。可给单值或逐标签 list。
        若为 None，则等价于 α = 1.
    gamma : float, default 2.0
        Focusing parameter γ (>=0)；γ=0 时退化为 BCE。
    reduction : {"mean","sum","none"}, default "mean"
        同 nn.BCEWithLogitsLoss
    """
    def __init__(
        self,
        alpha: float | list[float] | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha).float())
        else:
            self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : shape (N, L)
        targets : {0,1} same shape
        """
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # shape (N, L)

        p_t = prob * targets + (1 - prob) * (1 - targets)  # p_t

        # (1 - p_t)^γ
        focal_factor = (1.0 - p_t).pow(self.gamma)

        # α
        if self.alpha is None:
            alpha_factor = 1.0
        elif isinstance(self.alpha, torch.Tensor):
            # broadcasting 到 batch 维
            alpha_factor = self.alpha.view(1, -1)
        else:
            alpha_factor = self.alpha

        loss = alpha_factor * focal_factor * ce_loss  # shape (N, L)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss  # "none"
