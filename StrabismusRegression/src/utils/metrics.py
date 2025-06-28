"""
Unified metrics for multi-label classification (8 surgical decisions)
and multi-output regression (8 surgical doses).

**2025-06-09 更新**
-------------------
* 修复 `masked_*` 系列函数在 **numpy.ndarray × torch.Tensor** 相乘时的类型冲突：现在会自动把 `mask` 也转换为 NumPy；兼容 CPU/GPU。
* 其他逻辑保持一致（分类指标、整体回归指标）。
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import C

# ────────────────────────────────
# helper: numpy casting
# ────────────────────────────────

def _to_numpy(arr):
    """Accept numpy / torch ⇒ np.ndarray(float64)."""
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ────────────────────────────────
# classification metrics
# ────────────────────────────────

def multilabel_accuracy(y_true, y_pred, *, threshold: float = 0.5) -> float:
    """Exact-match accuracy (all 8 labels must match)."""
    y_t = _to_numpy(y_true).astype(bool)
    y_p = (_to_numpy(y_pred) >= threshold).astype(bool)
    return float(accuracy_score(y_t, y_p))


def multilabel_auc(y_true, y_prob, *, average: str = "macro") -> float:
    """ROC-AUC for multi-label outputs (macro / micro / weighted)."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_prob)
    try:
        return float(roc_auc_score(y_t, y_p, average=average))
    except ValueError:
        return float(np.nan)


def multilabel_mcc(y_true, y_pred, *, threshold: float = 0.5) -> float:
    """Macro-averaged Matthews correlation coefficient (MCC)."""
    y_t = _to_numpy(y_true).astype(int)
    y_p = (_to_numpy(y_pred) >= threshold).astype(int)
    mcc_list = [matthews_corrcoef(y_t[:, j], y_p[:, j]) for j in range(y_t.shape[1])]
    return float(np.nanmean(mcc_list))


# ────────────────────────────────
# helpers for regression metrics
# ────────────────────────────────

def _inverse_scale(arr):
    """Rescale normalized outputs back to millimetres."""
    return _to_numpy(arr) * C.TARGET_SCALE


# ────────────────────────────────
# regression metrics – overall
# ────────────────────────────────

def reg_mae(y_true, y_pred, *, multioutput="uniform_average") -> float:
    """Mean Absolute Error on original scale."""
    y_t = _inverse_scale(y_true)
    y_p = _inverse_scale(y_pred)
    return float(mean_absolute_error(y_t, y_p, multioutput=multioutput))


def reg_rmse(y_true, y_pred, *, multioutput="uniform_average") -> float:
    """
    Root Mean Squared Error on original scale.
    兼容 sklearn ≥1.6：无 squared 参数，手动开方。
    """
    y_t = _inverse_scale(y_true)
    y_p = _inverse_scale(y_pred)
    mse = mean_squared_error(y_t, y_p, multioutput=multioutput)
    return float(np.sqrt(mse))


def reg_r2(y_true, y_pred, *, multioutput="uniform_average") -> float:
    """R² (coefficient of determination) on original scale."""
    y_t = _inverse_scale(y_true)
    y_p = _inverse_scale(y_pred)
    return float(r2_score(y_t, y_p, multioutput=multioutput))


# ────────────────────────────────
# regression metrics – masked
# ────────────────────────────────

def _prep_mask(mask):
    """Ensure mask is NumPy array of same shape as predictions."""
    return _to_numpy(mask).astype(float)


def masked_mae_mm(pred, target, mask):
    """Masked MAE (mm) on selected outputs."""
    mask_np = _prep_mask(mask)
    pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
    err = np.abs(pred_mm - tgt_mm) * mask_np
    return float(err.sum() / mask_np.sum())


def masked_rmse_mm(pred, target, mask):
    """Masked RMSE (mm) on selected outputs."""
    mask_np = _prep_mask(mask)
    pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
    mse = ((pred_mm - tgt_mm) ** 2 * mask_np).sum() / mask_np.sum()
    return float(np.sqrt(mse))


def masked_r2_mm(pred, target, mask):
    """Masked R² on selected outputs."""
    mask_np = _prep_mask(mask)
    pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
    p = pred_mm[mask_np == 1].ravel()
    t = tgt_mm[mask_np == 1].ravel()
    return float(r2_score(t, p))


# ────────────────────────────────
# publicly exported names
# ────────────────────────────────

__all__ = [
    # classification
    "multilabel_accuracy",
    "multilabel_auc",
    "multilabel_mcc",
    # regression overall
    "reg_mae",
    "reg_rmse",
    "reg_r2",
    # regression masked
    "masked_mae_mm",
    "masked_rmse_mm",
    "masked_r2_mm",
]












# =======================   BK    ===============================================================







# """
# Unified metrics for multi‑label classification (8 surgical decisions)
# and multi‑output regression (8 surgical doses).
#
# **2025‑06‑09 更新**
# -------------------
# * 修复 `masked_*` 系列函数在 **numpy.ndarray × torch.Tensor** 相乘时的类型冲突：现在会自动把 `mask` 也转换为 NumPy；兼容 CPU/GPU。
# * 其他逻辑保持一致（分类指标、整体回归指标）。
#
# Functions
# ---------
# # ── Classification ─────────────
# multilabel_accuracy(y_true, y_pred, *, threshold=0.5)
# multilabel_auc(y_true, y_prob, *, average="macro")
# multilabel_mcc(y_true, y_pred, *, threshold=0.5)
#
# # ── Regression (overall) ───────
# reg_mae(y_true, y_pred, *, multioutput="uniform_average")
# reg_rmse(y_true, y_pred, *, multioutput="uniform_average")
# reg_r2(y_true, y_pred, *, multioutput="uniform_average")
#
# # ── Regression (masked, mm) ────
# masked_mae_mm(pred, target, mask)
# masked_rmse_mm(pred, target, mask)
# masked_r2_mm(pred, target, mask)
# """
#
# from __future__ import annotations
#
# import numpy as np
# import torch
# from sklearn.metrics import (
#     accuracy_score,
#     roc_auc_score,
#     matthews_corrcoef,
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
# )
#
# from src.config import C
#
# # ────────────────────────────────
# # helper: numpy casting
# # ────────────────────────────────
#
# def _to_numpy(arr):
#     """Accept numpy / torch ⇒ np.ndarray(float64)."""
#     if torch.is_tensor(arr):
#         return arr.detach().cpu().numpy()
#     return np.asarray(arr)
#
# # ────────────────────────────────
# # classification metrics
# # ────────────────────────────────
#
# def multilabel_accuracy(y_true, y_pred, *, threshold: float = 0.5) -> float:
#     """Exact‑match accuracy (all 8 labels must match)."""
#     y_t = _to_numpy(y_true).astype(bool)
#     y_p = (_to_numpy(y_pred) >= threshold).astype(bool)
#     return float(accuracy_score(y_t, y_p))
#
#
# def multilabel_auc(y_true, y_prob, *, average: str = "macro") -> float:
#     """ROC‑AUC for multi‑label outputs (macro / micro / weighted)."""
#     y_t = _to_numpy(y_true)
#     y_p = _to_numpy(y_prob)
#     try:
#         return float(roc_auc_score(y_t, y_p, average=average))
#     except ValueError:
#         return float(np.nan)
#
#
# def multilabel_mcc(y_true, y_pred, *, threshold: float = 0.5) -> float:
#     """Macro‑averaged Matthews correlation coefficient (MCC)."""
#     y_t = _to_numpy(y_true).astype(int)
#     y_p = (_to_numpy(y_pred) >= threshold).astype(int)
#     mcc_list = [matthews_corrcoef(y_t[:, j], y_p[:, j]) for j in range(y_t.shape[1])]
#     return float(np.nanmean(mcc_list))
#
# # ────────────────────────────────
# # helpers for regression metrics
# # ────────────────────────────────
#
# def _inverse_scale(arr):
#     """Rescale 0–1 outputs back to millimetres."""
#     return _to_numpy(arr) * C.TARGET_SCALE
#
# # ────────────────────────────────
# # regression metrics – overall
# # ────────────────────────────────
#
# def reg_mae(y_true, y_pred, *, multioutput="uniform_average") -> float:
#     y_t = _inverse_scale(y_true)
#     y_p = _inverse_scale(y_pred)
#     return float(mean_absolute_error(y_t, y_p, multioutput=multioutput))
#
#
# def reg_rmse(y_true, y_pred, *, multioutput="uniform_average") -> float:
#     y_t = _inverse_scale(y_true)
#     y_p = _inverse_scale(y_pred)
#     return float(mean_squared_error(y_t, y_p, squared=False, multioutput=multioutput))
#
#
# def reg_r2(y_true, y_pred, *, multioutput="uniform_average") -> float:
#     y_t = _inverse_scale(y_true)
#     y_p = _inverse_scale(y_pred)
#     return float(r2_score(y_t, y_p, multioutput=multioutput))
#
# # ────────────────────────────────
# # regression metrics – masked
# # ────────────────────────────────
#
# def _prep_mask(mask):
#     """Ensure mask is NumPy array of same shape as predictions."""
#     return _to_numpy(mask).astype(float)
#
#
# def masked_mae_mm(pred, target, mask):
#     mask_np = _prep_mask(mask)
#     pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
#     err = np.abs(pred_mm - tgt_mm) * mask_np
#     return float(err.sum() / mask_np.sum())
#
#
# def masked_rmse_mm(pred, target, mask):
#     mask_np = _prep_mask(mask)
#     pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
#     mse = ((pred_mm - tgt_mm) ** 2 * mask_np).sum() / mask_np.sum()
#     return float(np.sqrt(mse))
#
#
# def masked_r2_mm(pred, target, mask):
#     mask_np = _prep_mask(mask)
#     pred_mm, tgt_mm = _inverse_scale(pred), _inverse_scale(target)
#     p = pred_mm[mask_np == 1].ravel()
#     t = tgt_mm[mask_np == 1].ravel()
#     return float(r2_score(t, p))
#
# # ────────────────────────────────
# # publicly exported names
# # ────────────────────────────────
# __all__ = [
#     # classification
#     "multilabel_accuracy",
#     "multilabel_auc",
#     "multilabel_mcc",
#     # regression overall
#     "reg_mae",
#     "reg_rmse",
#     "reg_r2",
#     # regression masked
#     "masked_mae_mm",
#     "masked_rmse_mm",
#     "masked_r2_mm",
# ]
