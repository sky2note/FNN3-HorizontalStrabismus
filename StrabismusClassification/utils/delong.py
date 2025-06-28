"""
utils/delong.py

基于 DeLong 方法对比两组预测概率的 AUC 差异显著性检验。

用法示例：
  from utils.delong import delong_roc_test
  p_value, auc1, auc2 = delong_roc_test(y_true, prob_set1, prob_set2)
"""

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    计算中位秩（midrank）——DeLong 算法内部使用
    """
    sorted_x = np.sort(x)
    idx = np.argsort(x)
    n = len(x)
    mid = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j] == sorted_x[j+1]:
            j += 1
        mid_val = 0.5*(i + j) + 1
        mid[i:j+1] = mid_val
        i = j + 1
    # 转回原顺序
    inv = np.empty(n, dtype=int)
    inv[idx] = np.arange(n)
    return mid[inv]


def _fast_delong(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 DeLong AUC 的协方差矩阵
    返回 (auc, auc_cov)
    """
    # 分离正负样本
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)

    # 所有分数合并
    all_scores = np.concatenate((pos, neg))
    midranks = _compute_midrank(all_scores)

    # 拆分 midranks
    mr_pos = midranks[:m]
    mr_neg = midranks[m:]

    auc = (mr_pos.sum() - m*(m+1)/2) / (m*n)

    # 计算 V10 and V01
    V10 = (mr_pos - np.arange(1, m+1)).reshape(-1,1) / n
    V01 = (np.arange(1, n+1) - mr_neg).reshape(-1,1) / m

    # 协方差
    sx = np.cov(np.concatenate((V10, V01), axis=0), rowvar=False)
    return auc, sx


def delong_roc_test(
    y_true: np.ndarray,
    y_scores1: np.ndarray,
    y_scores2: np.ndarray
) -> tuple[float, float, float]:
    """
    对比两组概率预测的 AUC 是否有显著差异（双侧检验）。

    参数
    ----
    y_true     : 形状 (N,) 的真实标签
    y_scores1  : 形状 (N,) 的模型1预测分数/概率
    y_scores2  : 形状 (N,) 的模型2预测分数/概率

    返回
    ----
    p_value, auc1, auc2
    """
    auc1, var1 = _fast_delong(y_true, y_scores1)
    auc2, var2 = _fast_delong(y_true, y_scores2)
    # 只取对角线 var
    s1 = var1[0,0]
    s2 = var2[0,0]
    # Z 统计量
    z = (auc1 - auc2) / np.sqrt(s1 + s2)
    p = 2*(1 - stats.norm.cdf(abs(z)))
    return p, float(auc1), float(auc2)
