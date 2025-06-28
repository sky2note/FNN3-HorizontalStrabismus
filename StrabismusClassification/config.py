# config.py  ·  8→14-feature baseline + derived columns
# ────────────────────────────────────────────────
FEATURE_COLS = [
    # 原始 8 列
    "Age", "PrimaryDeviatingEye", "PrismCoverTest",
    "AxialLengthOD", "AxialLengthOS", "EqualVisionOpptyObserver",
    "SphericalEquivalentOD", "SphericalEquivalentOS",
    # 新派生 6 列（在 dataset.py 中动态生成，写在这里便于直观查看）
    "AL_diff",           # AxialLengthOD - AxialLengthOS
    "SE_diff",           # SphericalEquivalentOD - SphericalEquivalentOS
    "SE_ratio",          # SE_diff / (|SE_OD|+|SE_OS|+1e-3)
    "Age_bin",           # 年龄分桶 (0-4)
    "Age_sin", "Age_cos" # 年龄周期化
]

LABEL_COLS = [
    "MRRsOD", "MRRsOS", "LRRsOD", "LRRsOS",
    "MRRcOD", "MRRcOS", "LRRcOD", "LRRcOS",
]

DEFAULT_CSV   = "data/Data_M1D5_Preproc.csv"
DEFAULT_VAL   = "data/val.csv"
DEFAULT_CKPT  = "best_model.pth"
