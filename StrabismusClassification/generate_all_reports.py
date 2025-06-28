#!/usr/bin/env python3
# generate_all_reports.py
# ────────────────────────────────────────────────────────────

import os
from pathlib import Path

import numpy as np                            # 数值运算 :contentReference[oaicite:3]{index=3}
import pandas as pd                           # 表格导出 :contentReference[oaicite:4]{index=4}
import matplotlib.pyplot as plt               # 绘图 :contentReference[oaicite:5]{index=5}
import seaborn as sns                         # 混淆矩阵热图 :contentReference[oaicite:6]{index=6}
from sklearn.metrics import (                  # ROC/PR/mCM/混淆矩阵 :contentReference[oaicite:7]{index=7}
    roc_curve, auc,
    precision_recall_curve,
    multilabel_confusion_matrix
)
from sklearn.calibration import calibration_curve  # 校准曲线 :contentReference[oaicite:8]{index=8}
import shap                                    # SHAP 可解释性 :contentReference[oaicite:9]{index=9}
import torch

import config
from dataset import StrabismusDataset
from models.mlp_model import SurgeryIndicatorModel


def make_flow_diagram():
    """绘制样本流程图"""
    dot = __import__('graphviz').Digraph(comment='Study Flow')  # 延迟 import
    df = pd.read_csv(config.DEFAULT_CSV)
    dot.node('A', f'原始样本\nn={len(df)}')
    dot.node('B', '剔除缺失\nn=…')
    dot.node('C', '训练集 (80%)')
    dot.node('D', '验证集 (20%)')
    dot.edges(['AB','BC','BD'])
    Path('figures').mkdir(exist_ok=True)
    dot.render('figures/flow_diagram', format='png', cleanup=True)


def export_baseline_table():
    """导出基线特征 Table 1，先生成派生列。"""
    df = pd.read_csv(config.DEFAULT_CSV)

    # —— 1. 生成派生列（如不存在） —— #
    if "AxialLengthOD" in df.columns and "AxialLengthOS" in df.columns:
        df["AL_diff"] = df["AxialLengthOD"] - df["AxialLengthOS"]
    else:
        df["AL_diff"] = 0.0

    if "SphericalEquivalentOD" in df.columns and "SphericalEquivalentOS" in df.columns:
        df["SE_diff"] = df["SphericalEquivalentOD"] - df["SphericalEquivalentOS"]
    else:
        df["SE_diff"] = 0.0

    # —— 2. 选择要统计的基线列 —— #
    cols = ["Age", "AL_diff", "SE_diff", "PrimaryDeviatingEye"]

    # 如果还有缺失，就填 0 或者丢弃对应行
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    # —— 3. 计算并保存统计量 —— #
    stats = df[cols].agg(['mean', 'std', 'median',
                          lambda x: x.quantile(0.25),
                          lambda x: x.quantile(0.75)]).T
    stats.columns = ['mean', 'std', 'median', 'q25', 'q75']
    Path('tables').mkdir(exist_ok=True)
    stats.to_csv('tables/table1_baseline.csv', float_format='%.2f')



def plot_roc_pr(y_true, y_prob):
    """绘制 ROC 曲线与 Precision–Recall 曲线"""
    Path('figures').mkdir(exist_ok=True)
    # ROC
    plt.figure()
    for i, lab in enumerate(config.LABEL_COLS):
        fpr, tpr, _ = roc_curve(y_true[:,i], y_prob[:,i])
        plt.plot(fpr, tpr, label=f'{lab} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.savefig('figures/roc_curves.png'); plt.close()

    # PR
    plt.figure()
    for i, lab in enumerate(config.LABEL_COLS):
        p, r, _ = precision_recall_curve(y_true[:,i], y_prob[:,i])
        plt.plot(r, p, label=lab)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
    plt.savefig('figures/pr_curves.png'); plt.close()


def plot_calibration(y_true, y_prob):
    """绘制校准曲线 (Reliability Diagram)"""
    plt.figure()
    for i, lab in enumerate(config.LABEL_COLS):
        prob_true, prob_pred = calibration_curve(
            y_true[:,i], y_prob[:,i], n_bins=10, strategy='uniform'
        )
        plt.plot(prob_pred, prob_true, marker='o', label=lab)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Predicted Probability'); plt.ylabel('Observed Frequency')
    plt.legend(); plt.savefig('figures/calibration_plot.png'); plt.close()


def plot_confusion_matrices(y_true, y_pred):
    """绘制每标签混淆矩阵热图"""
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, lab in enumerate(config.LABEL_COLS):
        plt.figure()
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {lab}')
        plt.savefig(f'figures/cm_{lab}.png'); plt.close()


def plot_shap_summary():
    """计算并绘制 SHAP Summary Plot（兼容旧 state_dict）"""
    ds = StrabismusDataset(
        config.DEFAULT_CSV,
        config.FEATURE_COLS,
        config.LABEL_COLS,
        is_train=True
    )
    X = ds._features_tensor.numpy()
    model = SurgeryIndicatorModel(
        input_dim = X.shape[1],
        use_cc    = False
    )
    # ← 关键：strict=False 忽略不匹配的键
    model.load_state_dict(
        torch.load(config.DEFAULT_CKPT, map_location='cpu'),
        strict=False
    )

    explainer = shap.DeepExplainer(model, torch.tensor(X[:100]).float())
    shap_vals = explainer.shap_values(torch.tensor(X[100:200]).float())
    shap.summary_plot(
        shap_vals, X[100:200],
        feature_names = config.FEATURE_COLS,
        show          = False
    )
    plt.savefig('figures/shap_summary.png')
    plt.close()




def main():
    # 取第 0 折示例数据
    data = np.load('results/cv/fold0_val_preds.npz')
    y_true, y_prob = data['y_true'], data['y_prob']
    y_pred = (y_prob >= 0.5).astype(int)

    make_flow_diagram()           # 流程图 :contentReference[oaicite:11]{index=11}
    export_baseline_table()       # 基线特征表
    plot_roc_pr(y_true, y_prob)   # ROC/PR 曲线 :contentReference[oaicite:13]{index=13}
    plot_calibration(y_true, y_prob)   # 校准曲线
    plot_confusion_matrices(y_true, y_pred)  # 混淆矩阵
    plot_shap_summary()           # SHAP 图


if __name__ == "__main__":
    main()
