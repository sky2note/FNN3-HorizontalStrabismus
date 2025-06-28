"""
utils.calibration
──────────────────
概率校准工具集合：
1) Beta Calibration  (Kull & Silva, 2017)
2) Platt Scaling    （一维逻辑回归）
3) Temperature Scaling（单温度标量）

目前主脚本只用到 beta_calibration，其余可按需调用。
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

__all__ = [
    "beta_calibration",
    "platt_scaling",
    "temperature_scaling",
]

# ────────────────────────────────────────────────────────────
# 1) Beta Calibration
# ────────────────────────────────────────────────────────────
def beta_calibration(p: np.ndarray, y: np.ndarray):
    """
    Beta Calibration

    Parameters
    ----------
    p : ndarray shape (N,)
        原始（未校准）概率
    y : ndarray shape (N,)
        真值 {0,1}

    Returns
    -------
    f_cal : callable
        f_cal(prob) → calibrated_prob
    coef : list[float]
        [w1, w2, b]    对应论文中的 (a, b, c)
        方便后续推理阶段重建校准函数
    """
    eps = 1e-6
    z1 = np.log(np.clip(p, eps, 1 - eps))
    z2 = np.log1p(-p + eps)
    X  = np.vstack([z1, z2]).T  # shape (N,2)

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X, y)

    w1, w2 = lr.coef_[0]
    b      = lr.intercept_[0]

    def _beta_fn(prob: np.ndarray) -> np.ndarray:
        prob = np.clip(prob, eps, 1 - eps)
        zz1  = np.log(prob)
        zz2  = np.log1p(-prob + eps)
        logit = w1 * zz1 + w2 * zz2 + b
        return 1.0 / (1.0 + np.exp(-logit))

    return _beta_fn, [float(w1), float(w2), float(b)]

# ────────────────────────────────────────────────────────────
# 2) Platt Scaling
# ────────────────────────────────────────────────────────────
def platt_scaling(p: np.ndarray, y: np.ndarray):
    """
    一维逻辑回归 (Platt scaling)
    返回 f_cal，可直接作用于概率/对数几率
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(p.reshape(-1, 1), y)

    def _platt_fn(prob: np.ndarray) -> np.ndarray:
        return lr.predict_proba(prob.reshape(-1, 1))[:, 1]

    return _platt_fn

# ────────────────────────────────────────────────────────────
# 3) Temperature Scaling
# ────────────────────────────────────────────────────────────
def temperature_scaling(logits: np.ndarray, y: np.ndarray, max_iter: int = 50):
    """
    单温度标量 T：softmax(logits / T)
    这里只处理二类 / Sigmoid 场景（对数几率）。
    返回 f_cal 和温度 T
    """
    import torch
    import torch.optim as optim

    logit_torch = torch.as_tensor(logits, dtype=torch.float32)
    y_torch     = torch.as_tensor(y, dtype=torch.float32)

    T = torch.nn.Parameter(torch.ones(1))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    opt = optim.LBFGS([T], max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        loss = loss_fn(logit_torch / T, y_torch)
        loss.backward()
        return loss

    opt.step(_closure)

    T_val = T.item()

    def _temp_fn(prob: np.ndarray | torch.Tensor) -> np.ndarray:
        logit = np.log(prob / (1 - prob + 1e-6))
        return 1.0 / (1.0 + np.exp(-logit / T_val))

    return _temp_fn, T_val
