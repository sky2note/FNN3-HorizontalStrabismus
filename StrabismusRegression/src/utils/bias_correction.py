# src/utils/bias_correction.py

import numpy as np
from sklearn.linear_model import LinearRegression

def remove_mean_bias(y_pred: np.ndarray, y_true: np.ndarray):
    """
    恒定偏倚校正：将预测值减去平均偏差。

    参数
    ----
    y_pred : np.ndarray, shape (n,)
        原始预测值
    y_true : np.ndarray, shape (n,)
        真实值

    返回
    ----
    y_pred_adj : np.ndarray, shape (n,)
        校正后预测值 = y_pred - bias
    bias : float
        原始平均偏差 = mean(y_pred - y_true)
    """
    diffs = y_pred - y_true
    bias = float(np.mean(diffs))
    y_pred_adj = y_pred - bias
    return y_pred_adj, bias

def remove_proportional_bias(y_pred: np.ndarray, y_true: np.ndarray):
    """
    比例偏倚校正：对 (diff = y_pred - y_true) 与 mean 做线性回归，
    然后减去拟合的 α + β·mean。

    参数
    ----
    y_pred : np.ndarray, shape (n,)
    y_true : np.ndarray, shape (n,)

    返回
    ----
    y_pred_adj : np.ndarray, shape (n,)
        校正后预测值
    alpha : float
        回归截距
    beta : float
        回归斜率
    """
    means = (y_pred + y_true) / 2.0
    diffs = y_pred - y_true
    model = LinearRegression().fit(means.reshape(-1, 1), diffs)
    alpha = float(model.intercept_)
    beta  = float(model.coef_[0])
    correction = alpha + beta * means
    y_pred_adj = y_pred - correction
    return y_pred_adj, alpha, beta
