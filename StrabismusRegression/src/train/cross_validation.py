# -*- coding: utf-8 -*-
"""
10-折交叉验证（仅回归） + 残差可视化

运行：
$ python -m src.train.cross_validation
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

# matplotlib 仅用于保存图，Agg 后端适合服务器
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import C
from src.utils.data_utils import load_data
from src.utils.metrics import (
    masked_mae_mm,
    masked_rmse_mm,
    masked_r2_mm,
)
from src.utils.plot_utils import plot_error_boxplot
from src.train.train_regression import train_fold

# ---------------------------------------------------
# 目录准备
# ---------------------------------------------------
RES_DIR = Path(C.RESULTS_DIR)
FIG_DIR = RES_DIR / "figs"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 输出变量名称（顺序必须与模型输出一致）
OUT_NAMES = [
    "MRRsOD", "MRRsOS", "LRRsOD", "LRRsOS",
    "MRRcOD", "MRRcOS", "LRRcOD", "LRRcOS",
]

# ---------------------------------------------------
# 读数据
# ---------------------------------------------------
X, y_cls, y_reg = load_data()           # y_cls 仅用作掩码
n_folds = C.KFOLD

# 收集指标与残差
reg_mae, reg_rmse, reg_r2, val_loss_list = [], [], [], []
residual_pool = [[] for _ in range(C.OUTPUT_DIM)]   # len=8

kf = KFold(n_folds, shuffle=True, random_state=C.RANDOM_SEED)

# ---------------------------------------------------
# 交叉验证主循环
# ---------------------------------------------------
for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=0):
    print(f"\n===== Fold {fold} / {n_folds-1} =====")

    # 1) 训练 / 载入回归模型
    model_reg, best_loss = train_fold(fold)
    val_loss_list.append(best_loss)

    # 2) 准备验证集张量
    xb   = torch.from_numpy(X[val_idx]).to(C.DEVICE)
    yb_r = torch.from_numpy(y_reg[val_idx]).to(C.DEVICE)
    mask = torch.from_numpy(y_cls[val_idx]).float().to(C.DEVICE)

    # 3) 推理
    with torch.no_grad():
        pred_r = model_reg(xb).cpu()            # 0–1 标度
    yb_r_cpu = yb_r.cpu()

    # 4) 计算回归指标
    mae  = masked_mae_mm (pred_r, yb_r_cpu, mask.cpu())
    rmse = masked_rmse_mm(pred_r, yb_r_cpu, mask.cpu())
    r2   = masked_r2_mm  (pred_r, yb_r_cpu, mask.cpu())

    reg_mae.append(mae)
    reg_rmse.append(rmse)
    reg_r2.append(r2)

    print(f"  ↪ MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")

    # 5) 保存单折残差箱线图
    err_path = FIG_DIR / f"fold{fold}_err.png"
    plot_error_boxplot(
        pred_r.numpy(),
        yb_r_cpu.numpy(),
        mask.cpu().numpy(),
        err_path,
    )

    # 6) 收集残差到池（全部放到 CPU 端，避免索引设备不匹配）
    mask_cpu = mask.cpu()
    res_mm = (pred_r * C.TARGET_SCALE - yb_r_cpu * C.TARGET_SCALE) * mask_cpu  # (n_val, 8)

    for j in range(C.OUTPUT_DIM):
        residual_pool[j].append(
            res_mm[:, j][mask_cpu[:, j] == 1].numpy()
        )


# ---------------------------------------------------
# 汇总指标
# ---------------------------------------------------
print("\n── 10-折汇总（仅回归）──")
print(f"MAE  {np.mean(reg_mae):.3f} ± {np.std(reg_mae,ddof=1):.3f}")
print(f"RMSE {np.mean(reg_rmse):.3f} ± {np.std(reg_rmse,ddof=1):.3f}")
print(f"R²   {np.mean(reg_r2):.3f} ± {np.std(reg_r2,ddof=1):.3f}")

# ---------------------------------------------------
# 写 CSV / JSON
# ---------------------------------------------------
with (RES_DIR / "cv_metrics.csv").open("w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["fold", "reg_mae", "reg_rmse", "reg_r2", "val_loss"])
    for k in range(n_folds):
        wr.writerow([k, reg_mae[k], reg_rmse[k], reg_r2[k], val_loss_list[k]])

json.dump(
    {
        "reg_mae":  reg_mae,
        "reg_rmse": reg_rmse,
        "reg_r2":   reg_r2,
        "val_loss": val_loss_list,
    },
    open(RES_DIR / "cv_metrics.json", "w"),
    indent=2,
)

# ---------------------------------------------------
# 画 8 输出 × 10 折 残差 2×4 子图
# ---------------------------------------------------
fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharey=True)

for j, name in enumerate(OUT_NAMES):
    ax = axs[j // 4, j % 4]
    ax.boxplot(
        residual_pool[j],
        positions=range(n_folds),
        showfliers=False,
    )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Fold")
    if j % 4 == 0:
        ax.set_ylabel("Residual (mm)")

plt.tight_layout()
grid_path = FIG_DIR / "acrossfold_residuals_grid.png"
plt.savefig(grid_path, dpi=300)
plt.close(fig)

print(f"\n全部图已保存至 {FIG_DIR.relative_to(Path.cwd())}/")
