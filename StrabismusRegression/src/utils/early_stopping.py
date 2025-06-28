# src/utils/early_stopping.py

"""
EarlyStopping 模块
------------------
封装训练过程中的早停机制，独立于其他工具文件，便于单独配置与调参。

类 EarlyStopping
---------------
监控单个指标（如验证集损失或验证集 MCC），在指标在
patience 个连续 epoch 内未改善时，触发 early_stop=True。
每当指标改善时自动保存当前最佳模型权重。

使用示例
--------
from src.utils.early_stopping import EarlyStopping

early_stopper = EarlyStopping(
    patience=5,
    mode="min",                       # 'min' 监控要减小的指标（如 val_loss）
    delta=1e-4,                       # 改善阈值
    save_path="saved_models/best.pth" # 最佳模型保存路径
)

for epoch in range(1, max_epochs+1):
    # ... 训练、验证逻辑，计算 val_loss 或 val_mcc ...
    current_metric = val_loss  # 或 val_mcc
    early_stopper(current_metric, model)
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
"""

import os
from pathlib import Path
import torch


class EarlyStopping:
    """
    早停辅助类。

    参数
    ----
    patience : int
        如果指标在连续 patience 个 epoch 内没有“改善”，则停止训练。
    mode : {'min', 'max'}
        'min' 表示指标越小越好（如验证损失），'max' 表示指标越大越好（如验证 MCC）。
    delta : float
        判断“改善”的最小差值，变化量小于等于 delta 视作未改善。
    save_path : str 或 Path
        当指标改善时，将模型权重保存到此路径。
    """

    def __init__(self, patience=10, mode="min", delta=0.0, save_path="checkpoint.pth"):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = Path(save_path)

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # 根据 mode 设定比较函数
        if mode == "min":
            # 指标需下降才算改善
            self._is_improved = lambda curr, best: (best - curr) > delta
        else:
            # 指标需上升才算改善
            self._is_improved = lambda curr, best: (curr - best) > delta

    def __call__(self, current_score: float, model: torch.nn.Module):
        """
        在每个 epoch 验证后调用。

        参数
        ----
        current_score : float
            本轮验证集指标值（如 val_loss 或 val_mcc）。
        model : torch.nn.Module
            训练中的模型实例；当指标改善时，此模型的 state_dict 会被保存。
        """
        # 初次调用，初始化 best_score 并保存一次模型
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
            return

        # 比较本轮指标与历史最佳
        if self._is_improved(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model: torch.nn.Module):
        """
        保存模型权重到 self.save_path。
        如果目录不存在，会自动创建。
        """
        os.makedirs(self.save_path.parent, exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
