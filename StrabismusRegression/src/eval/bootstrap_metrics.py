# src/eval/bootstrap_metrics.py
"""
Compute bootstrap confidence intervals (CI) for the **multi‑output回归模型**
using **真值掩码** (y_reg > 0) 而非分类模型掩码。

运行示例
--------
$ python -m src.eval.bootstrap_metrics --n_boot 1000 --ci 0.95

生成文件
--------
results/bootstrap_ci.json
```json
{
  "mae": {"mean": 0.44, "ci_95": [0.41, 0.47]},
  "rmse": {"mean": 0.58, "ci_95": [0.54, 0.63]},
  "r2": {"mean": 0.85, "ci_95": [0.82, 0.87]}
}
```
(数值示例，取决于随机种子与数据集)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import C
from src.models.regression_model import RegressionModel
from src.utils.data_utils import load_data
from src.utils.metrics import (
    masked_mae_mm,
    masked_rmse_mm,
    masked_r2_mm,
)

# -----------------------------
# 文件名模板 (仅回归模型)
# -----------------------------
REG_PATTERN = "reg_fold{fold}.pth"  # 修改此处即可适配不同命名

# -----------------------------
# GPU → numpy helper
# -----------------------------
@torch.no_grad()
def _predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """一次性前向推理并返回 numpy."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=C.DEVICE)
    return model(X_t).cpu().numpy()


# -----------------------------
# 主函数
# -----------------------------

def main(n_boot: int, ci: float) -> None:
    """Bootstrap 评估 (真值掩码)."""
    # 1) 读取数据
    X, _y_cls, y_reg = load_data()

    # 2) 真值掩码：只对 target > 0 的元素计误差
    masks_full = (y_reg > 0).astype(float)  # shape (N, 8)

    # 3) 初始化预测矩阵
    preds_full = np.zeros_like(y_reg, dtype=np.float32)

    # 4) 读取 K 折索引 & 模型 → 逐折填充预测
    kfold_indices = pickle.load(open(C.FOLDS_DIR / f"mskfold_{C.KFOLD}fold.pkl", "rb"))

    for fold, split in enumerate(kfold_indices):
        reg_model = RegressionModel().to(C.DEVICE)
        state = torch.load(
            C.MODELS_DIR / REG_PATTERN.format(fold=fold),
            map_location=C.DEVICE,
            weights_only=True,  # ≥ PyTorch‑2.3；旧版可删
        )
        reg_model.load_state_dict(state)

        val_idx = split["val"]
        preds_full[val_idx] = _predict(reg_model, X[val_idx])

    # 5) 读取 bootstrap 索引
    boot_idx = pickle.load(open(C.FOLDS_DIR / f"bootstrap_{n_boot}.pkl", "rb"))

    # 6) 统计指标
    stats = {"mae": [], "rmse": [], "r2": []}
    for idx in tqdm(boot_idx, desc="Bootstrap"):
        t = y_reg[idx]
        p = preds_full[idx]
        m = masks_full[idx]
        stats["mae"].append(masked_mae_mm(p, t, m))
        stats["rmse"].append(masked_rmse_mm(p, t, m))
        stats["r2"].append(masked_r2_mm(p, t, m))

    # 7) 计算均值与 CI
    result = {}
    for k, arr in stats.items():
        arr_np = np.asarray(arr)
        lo, hi = np.percentile(arr_np, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        result[k] = {
            "mean": float(arr_np.mean()),
            f"ci_{int(ci * 100)}": [float(lo), float(hi)],
        }

    # 8) 保存结果
    Path(C.RESULTS_DIR).mkdir(exist_ok=True)
    out_path = C.RESULTS_DIR / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ bootstrap 结果已保存 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    main(args.n_boot, args.ci)
