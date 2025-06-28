"""
Classifier-Chain Head
─────────────────────
将基模型输出的表示向量 `rep` 依次生成 8 个标签的 *logits*，
后一个标签输入会串联前一个标签的 *概率*（sigmoid）。
实现方式：共享一层 `nn.Linear(rep_dim, hidden)` 提取中间
特征，再用 8 个单独的 `nn.Linear(hidden + 1⋯i, 1)` 逐步
累积条件信息。这样既轻量又能显式建模标签依赖。
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CCHead(nn.Module):
    def __init__(self, rep_dim: int, n_labels: int = 8, hidden: int = 32):
        super().__init__()
        self.n_labels = n_labels
        self.base = nn.Sequential(
            nn.Linear(rep_dim, hidden),
            nn.ReLU(inplace=True),
        )
        # 第 i 个标签的分类器：输入维 = hidden + i
        self.cls_layers = nn.ModuleList(
            [nn.Linear(hidden + i, 1) for i in range(n_labels)]
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        rep: shape (B, rep_dim) — 主干 MLP 的最后隐藏向量

        返回
        ----
        logits: shape (B, n_labels)
        """
        h0 = self.base(rep)            # (B, hidden)
        logits, probs = [], []
        for i, cls in enumerate(self.cls_layers):
            # 连接之前标签的概率
            if i == 0:
                h = h0
            else:
                h = torch.cat([h0, *probs], dim=1)
            logit = cls(h)             # (B, 1)
            logits.append(logit)
            probs.append(torch.sigmoid(logit))   # detach 可选
        return torch.cat(logits, dim=1)          # (B, 8)
