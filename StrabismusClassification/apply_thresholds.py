#!/usr/bin/env python
"""
apply_thresholds.py
───────────────────
读取 cross_validation 生成的 fold*_val_preds.npz，再应用：
   1) Beta Calibration (若提供系数)
   2) 逐标签阈值
输出跨折平均 SubsetAcc 与 MCC-macro。
"""
from __future__ import annotations
import argparse, glob, json, numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, matthews_corrcoef

import config

# ───────────── CLI ─────────────
pa = argparse.ArgumentParser()
pa.add_argument("--cv-dir",  default="results/cv", help="cross_validation 输出目录")
pa.add_argument("--thr-file", default="best_thresholds.json")
args = pa.parse_args()

# 阈值 & β-系数
data_json = json.loads(Path(args.thr_file).read_text())
thr_vec   = np.array([data_json["thresholds"][l] for l in config.LABEL_COLS])
coef_dict = data_json.get("beta_coef", {})       # dict[label] = [w1,w2,b]

def beta_apply(p: np.ndarray, coef) -> np.ndarray:
    """按 beta_calibration 逻辑回归系数校准单标签概率"""
    z1 = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    z2 = np.log1p(-p + 1e-6)
    logits = coef[0] * z1 + coef[1] * z2 + coef[2]
    return 1 / (1 + np.exp(-logits))

# ───────────── 逐折评估 ─────────────
subset_accs, mccs = [], []
npz_files = sorted(glob.glob(f"{args.cv_dir}/fold*_val_preds.npz"))
if not npz_files:
    raise FileNotFoundError("⚠️  未找到 fold*_val_preds.npz，请先运行 cross_validation.py")

for f in npz_files:
    dat = np.load(f)
    y_true = dat["y_true"]
    y_prob = dat["y_prob"]

    # — Beta Calibration —
    if coef_dict:
        for j, lab in enumerate(config.LABEL_COLS):
            if lab in coef_dict:
                y_prob[:, j] = beta_apply(y_prob[:, j], coef_dict[lab])

    # — 阈值 —
    y_pred = (y_prob >= thr_vec).astype(int)

    subset_accs.append(accuracy_score(y_true, y_pred))
    mccs.append(matthews_corrcoef(y_true.ravel(), y_pred.ravel()))

print(f"SubsetAcc (mean ± sd) = {np.mean(subset_accs):.3f} ± {np.std(subset_accs):.3f}")
print(f"MCC-macro (mean ± sd) = {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")
