#!/usr/bin/env python3
# scripts/export_fold_stats.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1. 将项目根目录加入模块搜索路径（假设本文件在 PROJECT_ROOT/scripts/ 下）
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# 2. 导入 load_data 用于获取多标签分类的真实标签
# ──────────────────────────────────────────────────────────────────────────────
from src.utils.data_utils import load_data

# ──────────────────────────────────────────────────────────────────────────────
# 3. 加载数据：X, y_class, y_reg
#    y_class 的形状为 (n_samples, n_labels)，值为 0/1
# ──────────────────────────────────────────────────────────────────────────────
_, y_class, _ = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# 4. 目录准备
# ──────────────────────────────────────────────────────────────────────────────
FOLDS_DIR   = PROJECT_ROOT / "data" / "folds"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5. 逐折统计
# ──────────────────────────────────────────────────────────────────────────────
rows = []
n_samples = y_class.shape[0]
all_idx   = np.arange(n_samples)

for fold in range(10):
    # 5.1 读取验证集索引
    val_path = FOLDS_DIR / f"fold{fold}_val.txt"
    val_idx  = np.loadtxt(val_path, dtype=int)

    # 5.2 计算训练集索引
    val_set   = set(val_idx.tolist())
    train_idx = np.array([i for i in all_idx if i not in val_set], dtype=int)

    # 5.3 样本数
    n_train = train_idx.size
    n_val   = val_idx.size

    # 5.4 阳性率（所有标签的平均正例率）
    pos_rate_train = y_class[train_idx].mean()
    pos_rate_val   = y_class[val_idx].mean()

    rows.append({
        "fold": fold,
        "n_train": n_train,
        "n_val": n_val,
        "pos_rate_train": pos_rate_train,
        "pos_rate_val": pos_rate_val
    })

# ──────────────────────────────────────────────────────────────────────────────
# 6. 存为 CSV
# ──────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_file = RESULTS_DIR / "fold_stats.csv"
df.to_csv(out_file, index=False, float_format="%.4f")

print(f"✅ 折分统计已生成：{out_file}")
print(df)
