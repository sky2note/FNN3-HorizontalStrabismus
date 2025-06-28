#!/usr/bin/env python3
# StrabismusClassification/scripts/plot_per_label_roc.py

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample


def load_config(project_root: Path):
    """动态加载项目根目录下的 config.py 并返回 C."""
    cfg_path = project_root / "config.py"
    spec = importlib.util.spec_from_file_location("proj_config", str(cfg_path))
    proj_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proj_cfg)
    return proj_cfg.C

def load_fold_preds(cv_dir: Path, n_splits: int):
    """读取所有折的 y_true 和 y_prob，返回合并后的数组."""
    y_true_list, y_prob_list = [], []
    for fold in range(n_splits):
        f = cv_dir / f"fold{fold}_val_preds.npz"
        if not f.exists():
            raise FileNotFoundError(f"找不到预测文件：{f}")
        data = np.load(f)
        y_true_list.append(data["y_true"])
        y_prob_list.append(data["y_prob"])
    return np.vstack(y_true_list), np.vstack(y_prob_list)

def compute_per_label_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000):
    """计算每个标签的 AUC 及其 95% 引导式置信区间."""
    n_labels = y_true.shape[1]
    records = []
    for j in range(n_labels):
        yt, yp = y_true[:, j], y_prob[:, j]
        auc = roc_auc_score(yt, yp)
        # Bootstrap
        boots = [
            roc_auc_score(yt[idx], yp[idx])
            for idx in (
                resample(np.arange(len(yt)), replace=True)
                for _ in range(n_bootstrap)
            )
        ]
        lower, upper = np.percentile(boots, [2.5, 97.5])
        records.append({
            "label_idx": j,
            "auc": round(auc, 3),
            "ci_lower": round(lower, 3),
            "ci_upper": round(upper, 3),
        })
    return pd.DataFrame(records)

def plot_per_label_roc_curve(records: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, out_png: Path):
    """绘制并保存每标签的 ROC 曲线."""
    plt.figure(figsize=(8, 6))
    for rec in records.to_dict("records"):
        j = rec["label_idx"]
        fpr, tpr, _ = roc_curve(y_true[:, j], y_prob[:, j])
        plt.plot(fpr, tpr, label=f"Label {j} (AUC={rec['auc']})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Label ROC Curves with 95% CI")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# —— 主流程 —— #
project_root = Path(__file__).parents[1]                     # StrabismusClassification/
C = load_config(project_root)                                # 动态加载 C
cv_dir = project_root / "results" / "cv"                     # 预测文件目录
y_true, y_prob = load_fold_preds(cv_dir, C.KFOLD)            # 读数据

# 计算 AUC 与 CI
df_auc_ci = compute_per_label_auc_ci(y_true, y_prob, n_bootstrap=1000)
csv_out = project_root / "results" / "per_label_roc_ci.csv"
df_auc_ci.to_csv(csv_out, index=False, float_format="%.3f")
print(f"✅ 已保存每标签 AUC & CI：{csv_out}")

# 绘图
png_out = project_root / "results" / "per_label_roc.png"
plot_per_label_roc_curve(df_auc_ci, y_true, y_prob, png_out)
print(f"✅ 已保存每标签 ROC 曲线：{png_out}")
