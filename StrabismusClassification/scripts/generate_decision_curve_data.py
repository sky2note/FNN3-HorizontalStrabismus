#!/usr/bin/env python3
# StrabismusClassification/scripts/generate_decision_curve_data.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

def net_benefit(tp, fp, n, p):
    """
    计算净获益：
      NB = TP/n − (FP/n) × (p/(1−p))
    """
    return tp/n - (fp/n) * (p/(1-p))

# 1. 载入所有折的多标签预测
root   = Path(__file__).parents[1]
cv_dir = root / "results" / "cv"

y_true_list = []
y_prob_list = []
for fold in range(10):
    arr = np.load(cv_dir / f"fold{fold}_val_preds.npz")
    y_true_list.append(arr["y_true"])  # shape=(n_val,8)
    y_prob_list.append(arr["y_prob"])

y_true = np.vstack(y_true_list)       # (583, 8)
y_prob = np.vstack(y_prob_list)

# 2. 定义二分类决策：任意标签被预测为 positive
any_true = (y_true.sum(axis=1) > 0).astype(int)
any_prob = y_prob.max(axis=1)

# 3. 枚举阈值，计算净获益
thresholds = np.linspace(0.01, 0.99, 99)
records = []
n = len(any_true)
prev = any_true.mean()

for p in thresholds:
    preds = (any_prob >= p).astype(int)
    # 指定 labels 确保 always 返回 2x2
    cm = confusion_matrix(any_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    nb      = net_benefit(tp, fp, n, p)
    nb_all  = prev - (1 - prev) * (p/(1-p))
    nb_none = 0.0

    records.append({
        "threshold":   round(p, 2),
        "net_benefit": round(nb, 4),
        "treat_all":   round(nb_all, 4),
        "treat_none":  nb_none,
    })

# 4. 保存到 CSV
df = pd.DataFrame(records)
out_csv = root / "results" / "decision_curve_data.csv"
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"✅ 已生成决策曲线数据：{out_csv}")
