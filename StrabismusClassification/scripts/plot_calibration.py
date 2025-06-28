#!/usr/bin/env python3
# StrabismusClassification/scripts/plot_calibration.py

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ─── 将项目根目录加入搜索路径 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

# 1. 定义数据目录
CV_DIR = PROJECT_ROOT / "results" / "cv"
OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(exist_ok=True)

# 2. 汇总所有折的真实标签和预测概率
y_true_list, y_prob_list = [], []
for fold in range(10):
    npz_path = CV_DIR / f"fold{fold}_val_preds.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到校准预测文件：{npz_path}")
    data = np.load(npz_path)
    # 假设文件内有 y_true (0/1) 与 y_prob (概率)
    y_true_list.append(data["y_true"])
    y_prob_list.append(data["y_prob"])

# 3. 将多折数据展平
y_true_flat = np.vstack(y_true_list).ravel()
y_prob_flat = np.vstack(y_prob_list).ravel()

# 4. 计算校准曲线（quantile 策略，更均匀地分配每箱样本量）:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
prob_true, prob_pred = calibration_curve(
    y_true_flat,
    y_prob_flat,
    n_bins=5,               # 可根据数据量调整
    strategy="quantile",    # 保证每个箱中样本数相同，消除无样本箱导致的断崖式跳变
    pos_label=1
)

# 5. 绘图
plt.figure(figsize=(6, 4))
plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibrated")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("True Positive Fraction")
plt.title("Reliability Diagram (Quantile Binning)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_curve.png", dpi=300)
plt.close()

print(f"✅ 校准曲线已保存至 {OUT_DIR/'calibration_curve.png'}")
