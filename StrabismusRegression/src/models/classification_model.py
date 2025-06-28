"""
models/classification_model.py

多标签手术指征 MLP（支持残差块 + 可选 Classifier-Chain Head）
  • 将所有 ReLU 的 inplace=False，避免 SHAP DeepExplainer 报错
  • 主干网络：Linear → BatchNorm → ReLU → Dropout × L + ResidualFC × N
  • 输出头：普通线性或 Classifier-Chain Head (--use_cc)
"""

from __future__ import annotations
from typing import Sequence, Tuple, List, Union

import torch
import torch.nn as nn

# ──────── 残差全连接块 ────────
class ResidualFC(nn.Module):
    """
    两层全连接残差块：
      x → Linear(in_dim, hid) → ReLU → Dropout → Linear(hid, in_dim) → + x
    """
    def __init__(self, in_dim: int, hid: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),                # 非原地 ReLU
            nn.Dropout(dropout),
            nn.Linear(hid, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ──────── Classifier-Chain Head ────────
class CCHead(nn.Module):
    """
    CCHead：在主干表示 vector 上依次生成 n_labels 个 logits，
    每一步将之前标签的预测概率 concat 到特征上，建模标签依赖。
    """
    def __init__(self, rep_dim: int, n_labels: int = 8, hidden: int = 32) -> None:
        super().__init__()
        # 共享特征变换层
        self.base = nn.Sequential(
            nn.Linear(rep_dim, hidden),
            nn.ReLU(),                # 非原地 ReLU
        )
        # 每个标签的分类器：输入维度 = hidden + i
        self.cls_layers = nn.ModuleList(
            [nn.Linear(hidden + i, 1) for i in range(n_labels)]
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        rep: Tensor, shape (B, rep_dim)
            从主干网络输出的表示向量

        返回
        ----
        logits: Tensor, shape (B, n_labels)
        """
        h0 = self.base(rep)             # (B, hidden)
        logits, probs = [], []
        for i, cls in enumerate(self.cls_layers):
            # 拼接之前的概率
            if i == 0:
                h = h0
            else:
                h = torch.cat([h0, *probs], dim=1)
            logit = cls(h)              # (B, 1)
            logits.append(logit)
            probs.append(torch.sigmoid(logit))  # 记录概率
        return torch.cat(logits, dim=1)  # (B, n_labels)


# ──────── 主模型 ────────
class SurgeryIndicatorModel(nn.Module):
    """
    深层 MLP，用于多标签分类：
      (Linear → BatchNorm → ReLU → Dropout) × len(hidden_dims)
      → ResidualFC × n_residual
      → [CCHead 或 Linear] 输出
    参数
    ----
    input_dim : int
        输入特征维度
    hidden_dims : Sequence[int]
        隐藏层神经元数列表，如 (64,128,64)
    dropout : float
        隐层 Dropout 比例
    n_residual : int
        残差块数量
    output_dim : int
        标签数
    use_cc : bool
        是否启用 Classifier-Chain Head
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[Sequence[int], Tuple[int, ...]] = (64, 128, 64),
        dropout: float = 0.3,
        n_residual: int = 2,
        output_dim: int = 8,
        use_cc: bool = False,
    ) -> None:
        super().__init__()
        self.use_cc = use_cc

        layers: List[nn.Module] = []
        dim_prev = input_dim

        # 构建多层全连接 + BN + ReLU + Dropout
        for dim in hidden_dims:
            layers += [
                nn.Linear(dim_prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),        # 非原地 ReLU
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim_prev = dim

        # 添加残差块
        for _ in range(n_residual):
            layers.append(
                ResidualFC(dim_prev, hid=max(32, dim_prev // 2), dropout=dropout)
            )

        # 主干特征网络
        self.feature_net = nn.Sequential(*layers)

        # 选择输出头
        if use_cc:
            self.head = CCHead(dim_prev, n_labels=output_dim)
        else:
            self.head = nn.Linear(dim_prev, output_dim)

        # 权重初始化
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        ----
        x: Tensor, shape (B, input_dim)
        返回 logits: Tensor, shape (B, output_dim)
        """
        rep = self.feature_net(x)
        return self.head(rep)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """对所有 Linear 权重做 Xavier 初始化，Bias 做 0 初始化"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
