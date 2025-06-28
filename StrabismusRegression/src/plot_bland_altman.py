



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_bland_altman_corrected.py

生成“回归式偏倚校正”后的 Bland–Altman 图，完整消除恒定偏倚和比例偏倚，
并将 95% 限度（Limits of Agreement）收窄至 ±1.96 SD。

用法：
    cd StrabismusRegression/tools
    python plot_bland_altman_corrected.py

输出：
    results/figs/bland_altman_proportional.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ─── 将项目根路径加入 sys.path，以便加载预测文件 ─────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

# 目录定义
CV_DIR  = PROJECT_ROOT / "results" / "cv"        # 各折预测文件目录
OUT_DIR = PROJECT_ROOT / "results" / "figs"      # 输出图像目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 汇总所有折的真实值与预测值
all_true, all_pred = [], []
for i in range(10):
    npz = np.load(CV_DIR / f"fold{i}_val_preds_regression.npz")
    all_true.append(npz["y_true"].ravel())
    all_pred.append(npz["y_pred"].ravel())
y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

# 2) 计算均值与差值
mean_vals = (y_true + y_pred) / 2
diffs      = y_pred - y_true

# 3) 回归式偏倚校正
#    拟合 diff = α + β × mean，得到截距 α 和斜率 β
lr = LinearRegression().fit(mean_vals.reshape(-1, 1), diffs)
alpha, beta = lr.intercept_, lr.coef_[0]
#    计算校正后差值：diff_corr = diff - (α + β mean)
diffs_corr = diffs - (alpha + beta * mean_vals)

# 4) 基于校正后差值计算新的 Bias 和 Limits of Agreement
bias_corr  = np.mean(diffs_corr)        # 应为 ≈ 0
sd_corr    = np.std(diffs_corr, ddof=1)
loa_lower  = bias_corr - 1.96 * sd_corr
loa_upper  = bias_corr + 1.96 * sd_corr

# 5) 绘图
plt.figure(figsize=(6, 4))
plt.scatter(mean_vals, diffs_corr, s=5, alpha=0.6)
plt.axhline(0,          color="red",   linestyle="--",
            label=f"Bias = {bias_corr:.3f}")
plt.axhline(loa_lower,  color="gray",  linestyle="--",
            label=f"-1.96 SD = {loa_lower:.3f}")
plt.axhline(loa_upper,  color="gray",  linestyle="--",
            label=f"+1.96 SD = {loa_upper:.3f}")
plt.xlabel("Average of Predicted & True (mm)")
plt.ylabel("Difference (Predicted − True) (mm)")
plt.title("Bland–Altman Plot")
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()

# 6) 保存
out_path = OUT_DIR / "bland_altman.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✅ 已保存校正后 Bland–Altman 图：{out_path}")












# ==============================================================================================================





# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import sys
# from sklearn.linear_model import LinearRegression
#
# # 加入项目根，加载预测文件
# PROJECT_ROOT = Path(__file__).parents[1].resolve()
# sys.path.insert(0, str(PROJECT_ROOT))
# CV_DIR = PROJECT_ROOT / "results" / "cv"
# OUT_DIR = PROJECT_ROOT / "results" / "figs"
# OUT_DIR.mkdir(exist_ok=True)
#
# # 1. 汇总所有折的真实与预测
# all_true, all_pred = [], []
# for i in range(10):
#     data = np.load(CV_DIR / f"fold{i}_val_preds_regression.npz")
#     all_true.append(data["y_true"].ravel())
#     all_pred.append(data["y_pred"].ravel())
# all_true = np.concatenate(all_true)
# all_pred = np.concatenate(all_pred)
#
# # 2. 计算平均与差值
# mean_vals = (all_true + all_pred) / 2
# diffs      = all_pred - all_true
# # 2.1 原始统计
# bias_orig  = diffs.mean()
# sd_orig    = diffs.std(ddof=1)
# lo_orig    = (bias_orig - 1.96*sd_orig, bias_orig + 1.96*sd_orig)
#
# # 2.2 平均偏倚校正
# diffs_mean_adj = diffs - bias_orig
# bias_mean_adj  = diffs_mean_adj.mean()
# sd_mean_adj    = diffs_mean_adj.std(ddof=1)
# lo_mean_adj    = (bias_mean_adj - 1.96*sd_mean_adj,
#                   bias_mean_adj + 1.96*sd_mean_adj)
#
# # 2.3 回归式偏倚校正
# X = mean_vals.reshape(-1,1)
# y = diffs
# reg = LinearRegression().fit(X, y)
# alpha, beta = reg.intercept_, reg.coef_[0]
# # 对 diff 做比例偏倚修正
# diffs_prop_adj = diffs - (alpha + beta * mean_vals)
# bias_prop_adj  = diffs_prop_adj.mean()
# sd_prop_adj    = diffs_prop_adj.std(ddof=1)
# lo_prop_adj    = (bias_prop_adj - 1.96*sd_prop_adj,
#                   bias_prop_adj + 1.96*sd_prop_adj)
#
# # 3. 画图对比
# fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
# titles = ["Original","Mean Bias Removed","Proportional Bias Removed"]
# stats  = [
#     (bias_orig, lo_orig),
#     (bias_mean_adj, lo_mean_adj),
#     (bias_prop_adj, lo_prop_adj)
# ]
# data   = [diffs, diffs_mean_adj, diffs_prop_adj]
# for ax, title, (b, (lo, hi)), d in zip(axes, titles, stats, data):
#     ax.scatter(mean_vals, d, s=5, alpha=0.6)
#     ax.axhline(b,    color="red",   linestyle="--", label=f"Bias={b:.3f}")
#     ax.axhline(lo,   color="gray",  linestyle="--", label=f"-1.96SD={lo:.3f}")
#     ax.axhline(hi,   color="gray",  linestyle="--", label=f"+1.96SD={hi:.3f}")
#     ax.set_title(title)
#     ax.set_xlabel("Average (mm)")
# axes[0].set_ylabel("Difference (Pred−True) (mm)")
# axes[2].legend(loc="lower right", fontsize="small")
# plt.tight_layout()
# plt.savefig(OUT_DIR/"bland_altman_comparison.png", dpi=300)
# plt.close()
# print("✅ Saved comparison to bland_altman_comparison.png")
