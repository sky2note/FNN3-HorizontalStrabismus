# utils/metrics.py
# ──────────────────────────────────────────────────────────────
# 多标签分类常用指标集合（纯函数、无任何训练逻辑）
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Literal, Dict, Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    hamming_loss,
    balanced_accuracy_score,
)

__all__ = [
    "compute_roc_auc",
    "compute_mcc",
    "compute_f1",
    "compute_subset_accuracy",
    "compute_hamming",
    "compute_balanced_accuracy",
    "summary_all",
]

AverageType = Literal["macro", "micro", "samples", "weighted", None]


# ----------------------------------------------------------------------------- #
# 内部辅助：将 torch / pandas / list 统一转为 NumPy                              #
# ----------------------------------------------------------------------------- #
def _ensure_numpy(arr):
    if hasattr(arr, "detach"):        # torch.Tensor
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, "to_numpy"):    # pandas.Series / DataFrame
        arr = arr.to_numpy()
    return np.asarray(arr)


# ----------------------------------------------------------------------------- #
# 核心指标                                                                      #
# ----------------------------------------------------------------------------- #
def compute_roc_auc(
    y_true,
    y_prob,
    average: AverageType = "macro",
) -> float | np.ndarray:
    """
    ROC-AUC for multilabel tasks.

    * average=None 可返回 shape=(L,) 的逐标签 AUC。
    * 随机分类器二分类 AUC≈0.5。
    """
    y_true = _ensure_numpy(y_true)
    y_prob = _ensure_numpy(y_prob)
    return roc_auc_score(y_true, y_prob, average=average)


def compute_mcc(
    y_true,
    y_pred,
    average: AverageType = "macro",
) -> float | np.ndarray:
    """
    Matthews Correlation Coefficient（多标签）。

    默认宏平均；average=None 返回逐标签数组。
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)

    per_label = np.array(
        [matthews_corrcoef(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    )

    if average is None:
        return per_label
    if average == "macro":
        return per_label.mean()
    if average == "weighted":
        w = y_true.sum(axis=0)
        return np.average(per_label, weights=w)
    if average == "micro":
        return matthews_corrcoef(y_true.ravel(), y_pred.ravel())

    raise ValueError(f"Unsupported average: {average}")


def compute_f1(
    y_true,
    y_pred,
    average: AverageType = "macro",
) -> float | np.ndarray:
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)
    return f1_score(y_true, y_pred, average=average)


def compute_subset_accuracy(y_true, y_pred) -> float:
    """
    Exact-match accuracy（子集准确率）——多标签场景下非常严格。
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)
    return accuracy_score(y_true, y_pred)


def compute_hamming(y_true, y_pred) -> float:
    """
    Hamming Loss：标签级错误率，越小越好。
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)
    return hamming_loss(y_true, y_pred)


def compute_balanced_accuracy(
    y_true,
    y_pred,
    average: AverageType = "macro",
) -> float | np.ndarray:
    """
    Balanced Accuracy = (Sensitivity + Specificity) / 2

    sklearn 原实现针对二分类 / 多类；此处逐标签后再聚合。
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)

    per_label = np.array(
        [
            balanced_accuracy_score(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
    )

    if average == "macro":
        return per_label.mean()
    if average == "weighted":
        return np.average(per_label, weights=y_true.sum(axis=0))
    if average == "micro":
        return balanced_accuracy_score(y_true.ravel(), y_pred.ravel())

    raise ValueError(f"Unsupported average: {average}")


# ----------------------------------------------------------------------------- #
# 汇总：便于一次性打印所有核心指标                                              #
# ----------------------------------------------------------------------------- #
def summary_all(
    y_true,
    y_prob,
    y_pred=None,
) -> Dict[str, Any]:
    """
    返回 dict：
        auc_macro、mcc_macro、f1_macro、subset_acc、hamming
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc_macro": compute_roc_auc(y_true, y_prob, average="macro"),
        "mcc_macro": compute_mcc(y_true, y_pred, average="macro"),
        "f1_macro": compute_f1(y_true, y_pred, average="macro"),
        "subset_acc": compute_subset_accuracy(y_true, y_pred),
        "hamming": compute_hamming(y_true, y_pred),
    }
