




import pandas as pd

# 1) 读入完整手术登记 CSV
df = pd.read_csv("Data_M1D5.csv")     # ←把文件名改成自己的

# 2) 需要检查的 8 个二进制列
bin_cols = [
    "MRRsOD_Binary", "MRRsOS_Binary",
    "LRRsOD_Binary", "LRRsOS_Binary",
    "MRRcOD_Binary", "MRRcOS_Binary",
    "LRRcOD_Binary", "LRRcOS_Binary"
]

# ------------- A) 统计“全 0” 行 ----------------
mask_all_zero = (df[bin_cols].sum(axis=1) == 0)
num_all_zero  = mask_all_zero.sum()
print(f"【全 0 行】共有 {num_all_zero} 条")
df[mask_all_zero].to_csv("rows_all_zero.csv", index=False)

# ------------- B) 统计“混合标记”行 --------------
def has_mixed_flags(row, side):
    """side='OD' 或 'OS'；若同眼既有 _s 又有 _c 标记，返回 True"""
    shorten = (row[f"MRRs{side}_Binary"] == 1) or (row[f"LRRs{side}_Binary"] == 1)
    recess  = (row[f"MRRc{side}_Binary"] == 1) or (row[f"LRRc{side}_Binary"] == 1)
    return shorten and recess

mask_mixed = df.apply(lambda r: has_mixed_flags(r, "OD") or has_mixed_flags(r, "OS"), axis=1)
num_mixed  = mask_mixed.sum()
print(f"【同眼既缩短又后徙】共有 {num_mixed} 条")
df[mask_mixed].to_csv("rows_mixed_flags.csv", index=False)

# ------------- C) 输出覆盖率 ---------------------
total_cases = len(df)
print("\n====== 覆盖率概览 ======")
print(f"总病例数        : {total_cases}")
print(f"完整编码病例数  : {total_cases - num_all_zero}")
print(f"覆盖率（%）     : {100*(total_cases-num_all_zero)/total_cases:.2f}")
print("（若覆盖率≠100%，请检查 rows_all_zero.csv & rows_mixed_flags.csv）")







# ==========================================================================================================






# -*- coding: utf-8 -*-
"""
make_table2_and_figure2.py
生成 Table2_MuscleFreq.xlsx 与 Figure2_DoseHistogram.png
作者：<你的名字>  日期：2025-06-21
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1. 读入数据 ==========
csv_path = Path("Data_M1D5.csv")
if not csv_path.exists():
    raise FileNotFoundError("未找到 surgery_registry.csv，请确认文件路径")

df = pd.read_csv(csv_path)

# ========== 2. 定义 OD / OS 标签映射 ==========
right_map = {
    "MRRsOD_Binary": "MRR (OD)",   # 内直肌缩短
    "LRRsOD_Binary": "LRR (OD)",   # 外直肌缩短
    "MRRcOD_Binary": "MRRc (OD)",  # 内直肌后徙
    "LRRcOD_Binary": "LRRc (OD)"   # 外直肌后徙"
}
left_map = {
    "MRRsOS_Binary": "MRR (OS)",
    "LRRsOS_Binary": "LRR (OS)",
    "MRRcOS_Binary": "MRRc (OS)",
    "LRRcOS_Binary": "LRRc (OS)"
}


def pick_muscle(row, mapping):
    """
    若同眼出现多个 1 => Mixed
    若全 0 => None
    否则返回唯一 1 对应的肌肉名称
    """
    chosen = [label for col, label in mapping.items() if row.get(col, 0) == 1]
    if len(chosen) == 0:
        return "None"
    if len(chosen) == 1:
        return chosen[0]
    return "Mixed"


df["OD_Muscle"] = df.apply(lambda r: pick_muscle(r, right_map), axis=1)
df["OS_Muscle"] = df.apply(lambda r: pick_muscle(r, left_map), axis=1)

# ========== 3. 生成 Table 2 ==========
cross = pd.crosstab(df["OD_Muscle"], df["OS_Muscle"], margins=True,
                    margins_name="Total (OD/OS)")     # 行列总计

# 单/双眼手术计数
bilat_mask = (df["OD_Muscle"] != "None") & (df["OS_Muscle"] != "None")
bilateral = bilat_mask.sum()
unilateral = len(df) - bilateral

laterality = pd.DataFrame(
    {"Cases": [unilateral, bilateral]},
    index=["Unilateral", "Bilateral"]
)

with pd.ExcelWriter("Table2_MuscleFreq.xlsx") as writer:
    cross.to_excel(writer, sheet_name="Muscle_CrossFreq")
    laterality.to_excel(writer, sheet_name="Laterality")

print("✅ 已保存 Table2_MuscleFreq.xlsx")

# ========== 4. 生成剂量直方图 ==========
dose_cols = ["MRRsOD", "LRRsOD", "MRRcOD", "LRRcOD",
             "MRRsOS", "LRRsOS", "MRRcOS", "LRRcOS"]

dose = df[dose_cols].to_numpy().flatten()
dose = dose[dose > 0]    # 去除未手术的 0

plt.figure(figsize=(6, 4))
plt.hist(dose, bins=18, edgecolor="black")
plt.xlabel("Planned displacement (mm)")
plt.ylabel("Frequency")
plt.title(f"Distribution of recession/resection length (n={len(dose)})")
plt.tight_layout()
plt.savefig("Figure2_DoseHistogram.png", dpi=300)
plt.close()

print("✅ 已保存 Figure2_DoseHistogram.png")
