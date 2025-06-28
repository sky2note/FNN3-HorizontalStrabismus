"""
utils/metrics_ci.py

基于 Bootstrap 的多指标置信区间计算：
  - AUC-macro      : ROC-AUC 的宏平均
  - MCC-macro      : Matthews Correlation Coefficient 的宏平均
  - F1-macro       : F1-score 的宏平均
  - SubsetAcc      : 样本级 Exact-match（所有标签均正确）的平均
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score
)


def bootstrap_ci_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None
) -> dict:
    """
    对四种多标签指标通过 Bootstrap 计算置信区间。

    参数
    ----
    y_true   : np.ndarray, shape (N, L)
        真实标签（二值化矩阵）
    y_prob   : np.ndarray, shape (N, L)
        预测概率
    y_pred   : np.ndarray, shape (N, L)
        二值化预测（应用固定阈值后的结果）
    n_boot   : int
        重采样次数
    alpha    : float
        置信水平 1-alpha，例如 0.05 表示 95% CI
    random_state : int | None
        随机种子

    返回
    ----
    ci_dict: dict
        {
          "auc_macro": (lower, upper),
          "mcc_macro": (lower, upper),
          "f1_macro": (lower, upper),
          "subset_acc": (lower, upper)
        }
    """
    rng = np.random.default_rng(random_state)
    N, L = y_true.shape

    # 存放每次重采样的指标值
    aucs, mccs, f1s, subsets = [], [], [], []

    for _ in range(n_boot):
        # 1. 样本索引重采样
        idx = rng.integers(0, N, size=N)
        yt = y_true[idx]    # (N, L)
        pp = y_prob[idx]    # (N, L)
        pr = y_pred[idx]    # (N, L)

        # 2. 逐标签计算 AUC / MCC / F1，然后宏平均
        auc_vals = []
        mcc_vals = []
        f1_vals  = []
        for j in range(L):
            # 2.1 ROC-AUC
            try:
                auc_vals.append(roc_auc_score(yt[:, j], pp[:, j]))
            except ValueError:
                # 如果只有单一标签，跳过该标签
                continue

            # 2.2 MCC
            mcc_vals.append(matthews_corrcoef(yt[:, j], pr[:, j]))

            # 2.3 F1
            f1_vals.append(f1_score(yt[:, j], pr[:, j], zero_division=0))

        aucs.append(np.mean(auc_vals))
        mccs.append(np.mean(mcc_vals))
        f1s.append(np.mean(f1_vals))

        # 3. 样本级 Exact-match：所有标签都正确的比例
        #    sklearn 的 accuracy_score 在多标签二值矩阵上即做 exact-match
        subsets.append(accuracy_score(yt, pr))

    # 4. 排序，取 alpha/2 与 1-alpha/2 分位数
    def _ci(arr):
        arr = np.sort(arr)
        lo = int((alpha/2) * n_boot)
        hi = int((1 - alpha/2) * n_boot)
        return float(arr[lo]), float(arr[hi])

    ci_dict = {
        "auc_macro": _ci(np.array(aucs)),
        "mcc_macro": _ci(np.array(mccs)),
        "f1_macro":  _ci(np.array(f1s)),
        "subset_acc": _ci(np.array(subsets)),
    }
    return ci_dict
