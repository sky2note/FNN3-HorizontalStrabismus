#!/usr/bin/env python3
# baseline_tables.py
# 需要的第三方包：pandas, numpy, matplotlib, openpyxl (写 Excel 用)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

############################
# 1. 读取数据
############################
DATA_FILE = "Data_M1D5.csv"        # ← 这里改成你的完整数据文件名
OUT_DIR   = Path(".")                   # 输出目录（当前文件夹）

df = pd.read_csv(DATA_FILE)

############################
# 2. 定义变量分组
############################
continuous_cols = [
    "Age", "PrismCoverTest",
    "AxialLengthOD", "AxialLengthOS",
    "SphericalEquivalentOD", "SphericalEquivalentOS"
]

categorical_cols = [
    "PrimaryDeviatingEye", "EqualVisionOpptyObserver"
]

# 所有二进制手术标签
muscle_binary_cols = [
    "MRRsOD_Binary", "MRRsOS_Binary", "LRRsOD_Binary", "LRRsOS_Binary",
    "MRRcOD_Binary", "MRRcOS_Binary", "LRRcOD_Binary", "LRRcOS_Binary"
]

# 所有剂量列（mm）
dose_cols = [
    "MRRsOD", "MRRsOS", "LRRsOD", "LRRsOS",
    "MRRcOD", "MRRcOS", "LRRcOD", "LRRcOS"
]

############################
# 3. 连续变量描述性统计
############################
cont_summary = df[continuous_cols].describe().T
cont_summary.rename(columns={
    "50%": "Median"
}, inplace=True)
cont_path = OUT_DIR / "BaselineTable1.xlsx"
cont_summary.to_excel(cont_path)
print(f"✅ BaselineTable1 保存至: {cont_path.resolve()}")

############################
# 4. 分类 / 二元变量频数
############################
cat_frames = []

# a) 主斜眼 & 其他分类
for col in categorical_cols:
    freq = df[col].value_counts(dropna=False).to_frame("Count")
    freq["Percent"] = 100 * freq["Count"] / len(df)
    freq["Variable"] = col
    cat_frames.append(freq.reset_index().rename(columns={"index": "Level"}))

# b) 8 个二元手术标签
binary_df = pd.DataFrame({
    col: [df[col].sum(), 100 * df[col].mean()]
    for col in muscle_binary_cols
}, index=["Count", "Percent"]).T.reset_index().rename(columns={"index": "Level"})
binary_df["Variable"] = "SurgeryBinary"
cat_frames.append(binary_df)

cat_summary = pd.concat(cat_frames, ignore_index=True)
cat_path = OUT_DIR / "CategoricalFreq.xlsx"
cat_summary.to_excel(cat_path, index=False)
print(f"✅ CategoricalFreq 保存至: {cat_path.resolve()}")

############################
# 5. 剂量直方图（联合左右眼且去掉 0 mm）
############################
all_doses = df[dose_cols].values.flatten()
all_doses = all_doses[all_doses > 0]     # 去掉未手术的 0 mm

plt.figure(figsize=(8, 5))
plt.hist(all_doses, bins=np.arange(3.5, 10.5, 0.5))  # 不指定颜色，遵守期刊要求
plt.xlabel("Planned displacement (mm)")
plt.ylabel("Frequency")
plt.title(f"Distribution of recession/resection length (n={len(all_doses)})")
plt.tight_layout()

hist_path = OUT_DIR / "DoseHistogram.png"
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"✅ DoseHistogram 保存至: {hist_path.resolve()}")

############################
# 6. 额外分析示例 —— 近/远偏斜角相关系数
############################
if {"PrismCoverTestDistance", "PrismCoverTestNear"}.issubset(df.columns):
    corr = df["PrismCoverTestDistance"].corr(df["PrismCoverTestNear"])
    print(f"近/远偏斜角相关系数: {corr:.2f}")

print("🎉 所有文件生成完毕。")



import pandas as pd
from pathlib import Path

csv = Path("Data_M1D5_Preproc.csv")      # 换成你的最终文件
cont_vars = ["Age", "PrismCoverTest", "AxL_mean", "AxL_diff",
             "SphEq_mean", "SphEq_diff", "CorrectedVisualAcuityOD",
             "CorrectedVisualAcuityOS"]
cat_vars  = ["PrimaryDeviatingEye", "EqualVisionOpptyObserver"]

df = pd.read_csv(csv)

rows = []
for v in cont_vars:
    rows.append({
        "Variable": v,
        "Mean ± SD / n (%)":
            f"{df[v].mean():.2f} ± {df[v].std():.2f}",
        "Missing (%)": f"{df[v].isna().mean()*100:.0f}"
    })
for v in cat_vars:
    n = df[v].value_counts().iloc[0]
    pct = n / len(df) * 100
    rows.append({
        "Variable": v,
        "Mean ± SD / n (%)": f"{n} ({pct:.1f})",
        "Missing (%)": f"{df[v].isna().mean()*100:.0f}"
    })

table1 = pd.DataFrame(rows)
table1.to_csv("table1_baseline_pre.csv", index=False)
print(table1)





# =======================================================================================

import pandas as pd
from pathlib import Path

# ========== 用户只需改这两行 =============
DATA_FILE = "Data_M1D5.csv"     # 数据文件名
SUBGROUP_COL = "PrimaryDeviatingEye"     # 用于流程图 n 的分组列
# =========================================

# 1. 读入数据
df = pd.read_csv(DATA_FILE)

# 2. 连续预测变量（按文件列名重写，如有缺失自动忽略并提醒）
continuous_cols = [
    "Age", "PrismCoverTest",
    "AxialLengthOD", "AxialLengthOS",
    "SphericalEquivalentOD", "SphericalEquivalentOS",
    "CorrectedVisualAcuityOD", "CorrectedVisualAcuityOS",
    "AxL_mean", "AxL_diff",
    "SphEq_mean", "SphEq_diff"
]
missing = [c for c in continuous_cols if c not in df.columns]
if missing:
    print(f"⚠️  以下连续列在数据中找不到，已跳过: {missing}")
continuous_cols = [c for c in continuous_cols if c in df.columns]

# 3. 计算 mean ± SD
stats = df[continuous_cols].agg(["mean", "std"]).T.round(2)        # :contentReference[oaicite:6]{index=6}
stats["mean ± SD"] = stats["mean"].astype(str) + " ± " + stats["std"].astype(str)
stats.to_csv("supp_table_S1_continuous.csv", index_label="Variable")   # :contentReference[oaicite:7]{index=7}
print("✓ 已生成 supp_table_S1_continuous.csv")

# 4. 统计分组样本量（包括 NaN）
if SUBGROUP_COL not in df.columns:
    raise KeyError(f"找不到分组列 {SUBGROUP_COL}，请在脚本顶部修改 SUBGROUP_COL")
counts = (
    df.groupby(SUBGROUP_COL, dropna=False)      # 包含 NaN 计数 :contentReference[oaicite:8]{index=8}
      .size()
      .reset_index(name="n")                    # :contentReference[oaicite:9]{index=9}
      .sort_values("n", ascending=False)
)
counts.to_csv("flowdiagram_counts.csv", index=False)                 # :contentReference[oaicite:10]{index=10}
print("✓ 已生成 flowdiagram_counts.csv")



