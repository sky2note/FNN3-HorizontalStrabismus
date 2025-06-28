#!/usr/bin/env python
"""
optimise_thresholds.py
──────────────────────────────────────────────────────────
基于交叉验证保存的 fold*_val_preds.npz 进行统一阈值优化：

1. 读取 results/cv/fold*_val_preds.npz，合并所有验证集的
   y_true 及 y_prob。
2. 对每个标签执行 Beta Calibration，返回 [w1, w2, b] 系数。
3. 在 0.40–0.70（步长 0.01）的细网格上搜索
   Matthews CorrCoef 最优阈值。
4. 输出 JSON：
   {
     "thresholds": {...},
     "beta_coef":  {...},   # label -> [w1,w2,b]
     "mcc_perlabel": {...}
   }
"""

from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef as MCC, accuracy_score

import config
from utils.calibration import beta_calibration  # 需返回 (f_cal, [w1,w2,b])

# ────────── CLI ──────────
pa = argparse.ArgumentParser()
pa.add_argument("--cv-dir", default="results/cv",
                help="包含 fold*_val_preds.npz 的目录")
pa.add_argument("--out", default="best_thresholds.json",
                help="输出 JSON 文件")
args = pa.parse_args()

# ────────── 1. 汇总所有折概率 ──────────
npz_files = sorted(glob.glob(f"{args.cv_dir}/fold*_val_preds.npz"))
if not npz_files:
    raise FileNotFoundError("未找到 fold*_val_preds.npz，请先运行 cross_validation.py")

y_true_list, y_prob_list = [], []
for f in npz_files:
    data = np.load(f)
    y_true_list.append(data["y_true"])
    y_prob_list.append(data["y_prob"])

y_true = np.concatenate(y_true_list, axis=0)   # shape (N, L)
y_prob = np.concatenate(y_prob_list, axis=0)   # shape (N, L)

# ────────── 2. 校准 & 阈值搜索 ──────────
best_thr, best_mcc, beta_coef = {}, {}, {}

for idx, label in enumerate(config.LABEL_COLS):
    # 2.1 Beta Calibration
    f_beta, coef = beta_calibration(y_prob[:, idx], y_true[:, idx])
    p_cal = f_beta(y_prob[:, idx])
    beta_coef[label] = coef          # [w1, w2, b]

    # 2.2 细网格阈值搜索 (0.40–0.70，步长 0.01)
    grid = np.arange(0.40, 0.70 + 1e-9, 0.01)
    mccs = [MCC(y_true[:, idx], (p_cal >= t).astype(int)) for t in grid]
    j = int(np.argmax(mccs))
    best_thr[label] = float(grid[j])
    best_mcc[label] = float(mccs[j])

    # 覆写回整体矩阵，后面算 SubsetAcc 用
    y_prob[:, idx] = p_cal

# ────────── 3. 计算整体 Subset Accuracy ──────────
thr_vec = np.array([best_thr[l] for l in config.LABEL_COLS])
subset_acc = accuracy_score(
    y_true, (y_prob >= thr_vec).astype(int)
)
print(f"✨  Pooled SubsetAcc = {subset_acc:.4f}")

# ────────── 4. 保存结果 ──────────
payload = {
    "thresholds": best_thr,
    "beta_coef": beta_coef,
    "mcc_perlabel": best_mcc
}
Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"✅ 阈值 + β 系数写入 → {args.out}")
