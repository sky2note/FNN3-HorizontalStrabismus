"""
Visualization helpers: ROC / PR curves for 8-label classifier
and box plots of regression residuals.

All plots use Matplotlib (Agg backend) and save PNG
to `results/figs/`.

Example
-------
>>> from src.utils.plot_utils import plot_multilabel_roc
>>> plot_multilabel_roc(y_true, y_prob, "results/figs/roc.png")
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.config import C

FIG_DIR = Path(C.RESULTS_DIR) / "figs"
os.makedirs(FIG_DIR, exist_ok=True)


# ────────────────────────────────
# ROC & PR curves
# ────────────────────────────────
def _get_label_names():
    return [
        "MRRsOD",
        "MRRsOS",
        "LRRsOD",
        "LRRsOS",
        "MRRcOD",
        "MRRcOS",
        "LRRcOD",
        "LRRcOS",
    ]


def plot_multilabel_roc(y_true, y_prob, save_path: str | os.PathLike):
    """One-vs-rest ROC per label + micro-average."""
    y_t, y_p = np.asarray(y_true), np.asarray(y_prob)
    n_labels = y_t.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_labels):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_p[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_t.ravel(), y_p.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plot
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        lw=2,
        label=f"micro-avg (AUC = {roc_auc['micro']:.2f})",
    )
    for i, name in enumerate(_get_label_names()):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=1,
            label=f"{name} (AUC = {roc_auc[i]:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – 8-label")
    plt.legend(loc="lower right", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multilabel_pr(y_true, y_prob, save_path: str | os.PathLike):
    """Precision-Recall curves with AP per label + micro."""
    y_t, y_p = np.asarray(y_true), np.asarray(y_prob)
    n_labels = y_t.shape[1]
    precision, recall, ap = {}, {}, {}

    for i in range(n_labels):
        precision[i], recall[i], _ = precision_recall_curve(
            y_t[:, i], y_p[:, i]
        )
        ap[i] = average_precision_score(y_t[:, i], y_p[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_t.ravel(), y_p.ravel()
    )
    ap["micro"] = average_precision_score(y_t, y_p, average="micro")

    plt.figure(figsize=(6, 5))
    plt.plot(
        recall["micro"],
        precision["micro"],
        lw=2,
        label=f"micro-avg (AP = {ap['micro']:.2f})",
    )
    for i, name in enumerate(_get_label_names()):
        plt.plot(
            recall[i],
            precision[i],
            lw=1,
            label=f"{name} (AP = {ap[i]:.2f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves – 8-label")
    plt.legend(loc="lower right", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ────────────────────────────────
# Regression residual box-plot
# ────────────────────────────────
def plot_error_boxplot(
    pred, target, mask, save_path: str | os.PathLike
):
    """
    Boxplot of (predicted – target) mm for each of 8 outputs
    where mask == 1.
    """
    pred_mm = pred * C.TARGET_SCALE
    tgt_mm = target * C.TARGET_SCALE
    err = (pred_mm - tgt_mm) * mask
    data = [
        err[:, i][mask[:, i] == 1].cpu().numpy()
        if hasattr(err, "cpu")
        else err[:, i][mask[:, i] == 1]
        for i in range(err.shape[1])
    ]

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        data,
        labels=_get_label_names(),
        showfliers=False,
        vert=True,
    )
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.ylabel("Residual (mm)")
    plt.title("Surgical Dose Errors (Pred – True)")
    plt.xticks(rotation=45, ha="right", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



# ──────────────────────────────────────────────────────────────
#  多折残差箱线图   plot_residuals_across_folds
# ──────────────────────────────────────────────────────────────
def plot_residuals_across_folds(
    residuals_per_fold: list[np.ndarray],
    output_name: str,
    save_path: str | os.PathLike,
):
    """
    residuals_per_fold : 长度 = n_folds 的列表；
                         第 k 项是一维 np.array，存放该输出在第 k 折
                         （mask==1 的样本）的 [预测-真实] 残差（mm）
    output_name        : 例如 "MRRsOD"
    save_path          : PNG 路径
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        residuals_per_fold,
        positions=range(len(residuals_per_fold)),
        showfliers=False,
    )
    plt.axhline(0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Fold")
    plt.ylabel("Residual (mm)")
    plt.title(f"{output_name} Residuals by Fold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
