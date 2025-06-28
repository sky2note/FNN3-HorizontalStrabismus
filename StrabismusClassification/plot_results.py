#!/usr/bin/env python
"""
plot_results.py
───────────────
依据 cross_validation.py 与 apply_thresholds.py 输出，
生成期刊所需 图表：STARD 流程图、ROC、校准、PR、阈值曲线、混淆矩阵，以及表格 CSV.
"""

import glob, json, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# ────── 1. 载入数据 ──────
# 多折概率 & 真实标签
npz_files = sorted(glob.glob("results/cv/fold*_val_preds.npz"))
y_true = np.concatenate([np.load(f)["y_true"] for f in npz_files], axis=0)
y_prob = np.concatenate([np.load(f)["y_prob"] for f in npz_files], axis=0)

# 最终阈值、beta_coef
data = json.load(open("best_thresholds.json", "r"))
thr_dict = data["thresholds"]
beta_dict = data["beta_coef"]

LABELS = [l for l in thr_dict.keys()]
thr = np.array([thr_dict[l] for l in LABELS])

# ────── 2. ROC 曲线 (Fig 2) ──────
plt.figure()
for i, lab in enumerate(LABELS):
    fpr, tpr, _ = roc_curve(y_true[:,i], y_prob[:,i])
    plt.plot(fpr, tpr, label=f"{lab} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(loc="lower right")
plt.savefig("fig2_roc.png", dpi=300)

# ────── 3. 校准曲线 (Fig 3) ──────
plt.figure(figsize=(6,6))
prob_true, prob_pred = calibration_curve(y_true.ravel(), y_prob.ravel(), n_bins=10)
plt.plot(prob_pred, prob_true, "s-")
plt.plot([0,1],[0,1],"--")
plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
plt.title("Calibration Plot")
plt.savefig("fig3_calibration.png", dpi=300)

# ────── 4. PR 曲线 (Fig 4) ──────
plt.figure()
for i, lab in enumerate(LABELS):
    prec, rec, _ = precision_recall_curve(y_true[:,i], y_prob[:,i])
    plt.plot(rec, prec, label=lab)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves"); plt.legend(loc="lower left")
plt.savefig("fig4_pr.png", dpi=300)

# ────── 5. 阈值 vs MCC 曲线 (Fig 5) ──────
from sklearn.metrics import matthews_corrcoef
plt.figure()
grid = np.arange(0.40,0.71,0.01)
for i, lab in enumerate(LABELS):
    mccs = [matthews_corrcoef(y_true[:,i], (y_prob[:,i]>=t).astype(int)) for t in grid]
    plt.plot(grid, mccs, label=lab)
plt.xlabel("Threshold"); plt.ylabel("MCC")
plt.title("Threshold vs MCC"); plt.legend(loc="best", ncol=2)
plt.savefig("fig5_thr_mcc.png", dpi=300)

# ────── 6. 混淆矩阵 (Fig 6) ──────
y_pred = (y_prob >= thr).astype(int)
cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Aggregated Confusion Matrix")
plt.savefig("fig6_confmat.png", dpi=300)

print("✅ 图表生成完毕，保存在当前目录。")
