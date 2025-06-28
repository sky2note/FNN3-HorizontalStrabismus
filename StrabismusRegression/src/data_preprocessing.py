"""数据预处理模块（StrabismusProject）
================================================
单一函数 **`preprocess_data`**：
``../data/Data_M1D5.csv → 清洗/标准化/编码 → ../data/Data_M1D5_Preproc.csv``
已消除 `FutureWarning`（pandas 3.0）——改用显式赋值而非链式 `inplace=True` 调用。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────────
# 列名配置
# ────────────────────────────────────────────────────────────────────────────────
INPUT_COLS: List[str] = [
    "Age",
    "PrismCoverTest",
    "AxialLengthOD",
    "AxialLengthOS",
    "EqualVisionOpptyObserver",
    "SphericalEquivalentOD",
    "SphericalEquivalentOS",
    "CorrectedVisualAcuityOD",
    "CorrectedVisualAcuityOS",
]
CATEGORICAL_COLS: List[str] = ["PrimaryDeviatingEye"]
OUTPUT_BINARY_COLS: List[str] = [
    "MRRsOD_Binary",
    "MRRsOS_Binary",
    "LRRsOD_Binary",
    "LRRsOS_Binary",
    "MRRcOD_Binary",
    "MRRcOS_Binary",
    "LRRcOD_Binary",
    "LRRcOS_Binary",
]
OUTPUT_CONTINUOUS_COLS: List[str] = [
    "MRRsOD",
    "MRRsOS",
    "LRRsOD",
    "LRRsOS",
    "MRRcOD",
    "MRRcOS",
    "LRRcOD",
    "LRRcOS",
]
ALL_FEATURES = CATEGORICAL_COLS + INPUT_COLS
ALL_OUTPUTS = OUTPUT_BINARY_COLS + OUTPUT_CONTINUOUS_COLS

# ────────────────────────────────────────────────────────────────────────────────
# 简单日志
# ────────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    pid = os.getpid()
    print(f"[预处理·{pid}] {msg}")

# ────────────────────────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_data(
    raw_path: Union[str, Path] | None = None,
    out_path: Union[str, Path] | None = None,
) -> pd.DataFrame:
    """读取、清洗并保存手术数据，返回 `DataFrame`。

    参数
    ------
    raw_path : 原始 CSV 路径（默认 ``../data/Data_M1D5.csv``）。
    out_path : 输出 CSV 路径（默认 ``../data/Data_M1D5_Preproc.csv``）。
    """

    # ── 0. 路径设置 ──────────────────────────────────────────────────────────
    raw_path = Path(raw_path or "../data/Data_M1D5.csv").expanduser().resolve()
    out_path = Path(out_path or "../data/Data_M1D5_Preproc.csv").expanduser().resolve()

    _log(f"读取原始数据 ⇒ {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到原始文件: {raw_path}")
    df = pd.read_csv(raw_path)

    # 仅保留预期列
    expected_cols = ["ID"] + ALL_FEATURES + ALL_OUTPUTS
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise KeyError(f"CSV 缺少列: {missing}")
    df = df[expected_cols].copy()

    # ── 1. 处理缺失值 ────────────────────────────────────────────────────────
    for col in INPUT_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # ── 2. 裁剪极端值 (IQR) ────────────────────────────────────────────────
    for col in INPUT_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    # ── 3. 特征工程 ──────────────────────────────────────────────────────────
    df["PrimaryDeviatingEye"] = df["PrimaryDeviatingEye"].map({"OD": 0, "OS": 1}).astype("int8")

    scaler = StandardScaler()
    df[INPUT_COLS] = scaler.fit_transform(df[INPUT_COLS])

    # ── 4. 保存结果 ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(f"已保存清洗数据 ⇒ {out_path} (shape={df.shape})")

    return df



# preprocess_data()


# ======================================       数据清洗和缩放           ====================================================



import pandas as pd

def analyze_FNN2_columns(input_csv='../data/Data_M1D6_Preproc.csv',
                         output_txt='../data/Data_M1D6_Preproc_analysis.txt'):
    """
    从 input_csv 中读取以下列：

      - Age
      - PrismCoverTest
      - AxialLengthOD
      - AxialLengthOS
      - EqualVisionOpptyObserver
      - SphericalEquivalentOD
      - SphericalEquivalentOS
      - CorrectedVisualAcuityOD
      - CorrectedVisualAcuityOS

    对每一列进行统计分析，包括：
      1. 非空值计数 (count) 及缺失值数 (missing)
      2. 均值 (mean)、标准差 (std)、最小值 (min)、四分位数 (Q1, Q2, Q3)、最大值 (max)
      3. 偏度 (skewness)、峰度 (kurtosis)
      4. 基于 IQR (四分位距) 的离群点检测:
         - IQR = Q3 - Q1
         - 下界 = Q1 - 1.5*IQR
         - 上界 = Q3 + 1.5*IQR
         - 统计并示例前 5 个离群观测

    将所有结果按文本格式写入 output_txt。
    """
    # 1. 明确需要分析的列
    cols_to_analyze = [

        'Age',
        'PrismCoverTest',
        'AxL_mean',
        'AxL_diff',
        'SphEq_mean',
        'SphEq_diff',
        'PrimaryDeviatingEye',
        'MRRsOD',
        'MRRsOS',
        'LRRsOD',
        'LRRsOS',
        'MRRcOD',
        'MRRcOS',
        'LRRcOD',
        'LRRcOS'
    ]

    # 2. 读取 CSV，仅加载所需列
    df = pd.read_csv(input_csv, usecols=cols_to_analyze)  # :contentReference[oaicite:3]{index=3}

    # 3. 打开输出文本文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("Analysis Report for Selected Columns\n")
        f.write("===================================\n\n")

        # 4. 对每一列依次做统计
        for col in cols_to_analyze:
            # 如果列不存在，则提醒并跳过
            if col not in df.columns:
                f.write(f"Column '{col}' not found in the dataset.\n\n")
                continue

            series = df[col]
            # 4.1 计数与缺失值
            count_non_null = series.count()         # 非空值数量:contentReference[oaicite:4]{index=4}
            missing_vals = series.isna().sum()      # 缺失值数量:contentReference[oaicite:5]{index=5}

            # 4.2 描述性统计指标
            mean_val = series.mean()                # 均值:contentReference[oaicite:6]{index=6}
            std_val = series.std()                  # 标准差:contentReference[oaicite:7]{index=7}
            min_val = series.min()                  # 最小值:contentReference[oaicite:8]{index=8}
            q1 = series.quantile(0.25)              # 25% 分位数:contentReference[oaicite:9]{index=9}
            q2 = series.quantile(0.50)              # 中位数:contentReference[oaicite:10]{index=10}
            q3 = series.quantile(0.75)              # 75% 分位数:contentReference[oaicite:11]{index=11}
            max_val = series.max()                  # 最大值:contentReference[oaicite:12]{index=12}

            # 4.3 偏度与峰度
            skewness = series.skew()                # 偏度:contentReference[oaicite:13]{index=13}
            kurt = series.kurtosis()                # 峰度:contentReference[oaicite:14]{index=14}

            # 4.4 IQR 及离群值检测
            iqr_val = q3 - q1                       # IQR = Q3 - Q1 :contentReference[oaicite:15]{index=15}
            lower_bound = q1 - 1.5 * iqr_val        # 下界
            upper_bound = q3 + 1.5 * iqr_val        # 上界
            # 标记并统计离群值
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            num_outliers = outliers_mask.sum()      # 离群值数量:contentReference[oaicite:16]{index=16}
            # 列举前 5 个离群值索引及对应数值示例
            outlier_indices = list(series[outliers_mask].index[:5])
            outlier_values = [series.loc[i] for i in outlier_indices]

            # 5. 将该列所有统计结果写入文件
            f.write(f"Column: {col}\n")
            f.write(f"{'-'*len(f'Column: {col}')}\n")
            f.write(f"Count (non-null)        : {count_non_null}\n")
            f.write(f"Missing values          : {missing_vals}\n")
            f.write(f"Mean                    : {mean_val:.4f}\n")
            f.write(f"Std Deviation           : {std_val:.4f}\n")
            f.write(f"Minimum                 : {min_val:.4f}\n")
            f.write(f"25th Percentile (Q1)    : {q1:.4f}\n")
            f.write(f"Median (Q2)             : {q2:.4f}\n")
            f.write(f"75th Percentile (Q3)    : {q3:.4f}\n")
            f.write(f"Maximum                 : {max_val:.4f}\n")
            f.write(f"Skewness                : {skewness:.4f}\n")
            f.write(f"Kurtosis                : {kurt:.4f}\n")
            f.write(f"IQR (Q3 - Q1)           : {iqr_val:.4f}\n")
            f.write(f"Lower bound (Q1 - 1.5IQR): {lower_bound:.4f}\n")
            f.write(f"Upper bound (Q3 + 1.5IQR): {upper_bound:.4f}\n")
            f.write(f"Number of outliers      : {num_outliers}\n")
            if num_outliers > 0:
                f.write(f"Sample outlier indices  : {outlier_indices}\n")
                f.write(f"Sample outlier values   : {[round(v,4) for v in outlier_values]}\n")
            f.write("\n")

        # 最后写入结尾标志
        f.write("End of Analysis\n")

    print(f"Analysis completed. Results saved to '{output_txt}'.")

# 直接调用函数示例（运行时请确保当前目录下有 'Data_FNN2_MHS1.csv'）：

analyze_FNN2_columns()



# ======================================         统计分析所有列         ====================================================



"""数据预处理：M1D6_pre1.csv → M1D6_Preproc.csv
================================================
单一函数 **`preprocess_data_v2`** ，专为新版手术数据（587 行）设计。

### 输入特征
``PrimaryDeviatingEye, Age, PrismCoverTest, AxialLengthOD, AxialLengthOS,
  SphericalEquivalentOD, SphericalEquivalentOS``

### 处理流程
1. **读取** `../data/Data_M1D6_pre1.csv`。
2. **缺失值**：数值→中位数；类别→众数。
3. **IQR 裁剪**：±1.5×IQR 剪除极端值。
4. **派生特征**：
   * `AxL_mean`, `AxL_diff` ← AxialLengthOD/OS；
   * `SphEq_mean`, `SphEq_diff` ← SphericalEquivalentOD/OS；
   * 删除原 4 列避免共线性。
5. **编码**：`PrimaryDeviatingEye` → 0(OD) / 1(OS)。
6. **标准化**：`StandardScaler` 处理 `Age`, `PrismCoverTest` 与 4 个派生列。
7. **列顺序调整**：将 4 个派生特征 *插入* `PrismCoverTest` 之后、`MRRsOD_Binary` 之前。
8. **保存** `../data/Data_M1D6_Preproc.csv`。

> 其余标签与剂量列保持原顺序，便于下游掩码回归。
"""
# from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────────
# 列配置
# ────────────────────────────────────────────────────────────────────────────────
BASE_NUM_COLS: List[str] = [
    "Age",
    "PrismCoverTest",
    "AxialLengthOD",
    "AxialLengthOS",
    "SphericalEquivalentOD",
    "SphericalEquivalentOS",
]
CATEGORICAL_COLS = ["PrimaryDeviatingEye"]
DERIVED_COLS = ["AxL_mean", "AxL_diff", "SphEq_mean", "SphEq_diff"]
SCALE_COLS = ["Age", "PrismCoverTest"] + DERIVED_COLS

# ────────────────────────────────────────────────────────────────────────────────
# 日志
# ────────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[M1D6预处理] {msg}")

# ────────────────────────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_data_v2(
    raw_path: Union[str, Path] | None = None,
    out_path: Union[str, Path] | None = None,
) -> pd.DataFrame:
    """清洗并保存手术数据。

    参数
    ------
    raw_path : 原 CSV，默认 ``../data/Data_M1D6_pre1.csv``。
    out_path : 预处理后 CSV，默认 ``../data/Data_M1D6_Preproc.csv``。
    """

    # 0. 路径
    raw_path = Path(raw_path or "../data/Data_M1D6_pre1.csv").resolve()
    out_path = Path(out_path or "../data/Data_M1D6_Preproc.csv").resolve()

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    _log(f"读取 ⇒ {raw_path}")

    df = pd.read_csv(raw_path)

    # 1. 缺失值
    for col in BASE_NUM_COLS:
        df[col] = df[col].fillna(df[col].median())
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna(df[CATEGORICAL_COLS].mode().iloc[0])

    # 2. IQR 裁剪
    for col in BASE_NUM_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # 3. 派生特征
    df["AxL_mean"] = (df["AxialLengthOD"] + df["AxialLengthOS"]) / 2
    df["AxL_diff"] = (df["AxialLengthOD"] - df["AxialLengthOS"]).abs()
    df["SphEq_mean"] = (df["SphericalEquivalentOD"] + df["SphericalEquivalentOS"]) / 2
    df["SphEq_diff"] = (df["SphericalEquivalentOD"] - df["SphericalEquivalentOS"]).abs()

    df.drop(columns=[
        "AxialLengthOD",
        "AxialLengthOS",
        "SphericalEquivalentOD",
        "SphericalEquivalentOS",
    ], inplace=True)

    # 4. 编码类别
    df["PrimaryDeviatingEye"] = df["PrimaryDeviatingEye"].map({"OD": 0, "OS": 1}).astype("int8")

    # 5. 标准化
    df[SCALE_COLS] = StandardScaler().fit_transform(df[SCALE_COLS])

    # 6. 列顺序调整
    def _reorder_columns(frame: pd.DataFrame) -> List[str]:
        cols = list(frame.columns)
        # 目标位置 = PrismCoverTest 之后
        insert_pos = cols.index("PrismCoverTest") + 1
        # 先移除派生列
        for c in DERIVED_COLS:
            cols.remove(c)
        # 找到第一个标签列 MRRsOD_Binary 的位置
        first_label_idx = cols.index("MRRsOD_Binary")
        # 将派生列插入 PrismCoverTest 后
        cols = cols[:insert_pos] + DERIVED_COLS + cols[insert_pos:]
        # 确保标签顺序不变
        return cols

    df = df[_reorder_columns(df)]

    # 7. 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(f"已保存 ⇒ {out_path} (shape={df.shape})")

    return df



# preprocess_data_v2()

# ======================================        按规划处理相关列          ====================================================





