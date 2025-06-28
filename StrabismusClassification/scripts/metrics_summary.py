#!/usr/bin/env python3
# scripts/metrics_summary.py
"""
汇总 10 折交叉验证结果，计算多指标的 Bootstrap 置信区间，
并可选地对比两组预测的 AUC 差异（DeLong 检验）。

用法：
  python scripts/metrics_summary.py \
      --cv-dir results/cv \
      --thr-file best_thresholds.json \
      [--method2 path/to/other_preds.npz] \
      [--n_boot 2000]
"""

import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

# ─── 将项目根目录加入 sys.path，以便导入 utils 包 ───
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 从 utils 包中导入前文定义的函数
from utils.metrics_ci import bootstrap_ci_all
from utils.delong      import delong_roc_test


def load_cv_preds(cv_dir: Path):
    """
    批量读取 results/cv/fold*_val_preds.npz，
    并将 y_true, y_prob 堆叠成大矩阵。
    返回：
      y_true: np.ndarray, shape (N_total, L)
      y_prob: np.ndarray, shape (N_total, L)
    """
    true_list, prob_list = [], []
    for npz in sorted(glob.glob(str(cv_dir / "fold*_val_preds.npz"))):
        data = np.load(npz)
        true_list.append(data["y_true"])
        prob_list.append(data["y_prob"])
    y_true = np.vstack(true_list)
    y_prob = np.vstack(prob_list)
    return y_true, y_prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-dir",    required=True, type=Path,
                   help="交叉验证结果目录 (results/cv)")
    p.add_argument("--thr-file",  required=True, type=Path,
                   help="阈值+校准系数 JSON 文件 (best_thresholds.json)")
    p.add_argument("--method2",   type=Path,
                   help="可选：另一套 preds.npz，用于 DeLong 对比")
    p.add_argument("--n_boot",    type=int, default=1000,
                   help="Bootstrap 重采样次数 (默认 1000)")
    args = p.parse_args()

    # 1 ── 读取 CV 预测
    y_true, y_prob = load_cv_preds(args.cv_dir)

    # 2 ── 根据阈值 JSON 生成二值预测 y_pred
    thr = json.load(open(args.thr_file))["thresholds"]
    labels = list(thr.keys())
    thresh_vec = np.array([thr[l] for l in labels], dtype=float)
    y_pred = (y_prob >= thresh_vec).astype(int)

    # 3 ── Bootstrap 置信区间计算
    ci = bootstrap_ci_all(
        y_true, y_prob, y_pred,
        n_boot=args.n_boot, alpha=0.05, random_state=42
    )
    print("=== Bootstrap 95% CI (macro metrics) ===")
    for metric, (lo, hi) in ci.items():
        print(f"{metric}: [{lo:.3f}, {hi:.3f}]")

    # 4 ── 将 CI 写入文件
    csv_path = args.cv_dir / "metrics_ci.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["metric", "ci_lower", "ci_upper"])
        for m, (lo, hi) in ci.items():
            w.writerow([m, f"{lo:.3f}", f"{hi:.3f}"])
    json_path = args.cv_dir / "metrics_ci.json"
    with open(json_path, "w") as f:
        json.dump(ci, f, indent=2)
    print(f"CI 已写入 → {csv_path}, {json_path}")

    # 5 ── 可选：DeLong AUC 差异检验
    if args.method2:
        print("\n=== DeLong AUC Test ===")
        data2 = np.load(args.method2)
        y_prob2 = data2["y_prob"]
        L = y_true.shape[1]
        for i in range(L):
            pval, auc1, auc2 = delong_roc_test(
                y_true[:, i], y_prob[:, i], y_prob2[:, i]
            )
            print(f"Label {i}: AUC1={auc1:.3f}, AUC2={auc2:.3f}, p={pval:.3e}")
        # 宏平均层面对比
        pval, auc1, auc2 = delong_roc_test(
            y_true.ravel(), y_prob.ravel(), y_prob2.ravel()
        )
        print(f"Macro: AUC1={auc1:.3f}, AUC2={auc2:.3f}, p={pval:.3e}")


if __name__ == "__main__":
    main()
