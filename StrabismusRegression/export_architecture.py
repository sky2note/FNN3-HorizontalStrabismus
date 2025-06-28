#!/usr/bin/env python
# export_architecture.py

import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. 将项目根目录加入 Python 模块搜索路径
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# 2. 导入依赖
# ──────────────────────────────────────────────────────────────────────────────
from torchinfo import summary
from src.config import C
from src.models.classification_model import SurgeryIndicatorModel
from src.models.regression_model     import RegressionModel

# ──────────────────────────────────────────────────────────────────────────────
# 3. 导出函数
# ──────────────────────────────────────────────────────────────────────────────
def dump_model(model, input_size, outfile):
    """
    导出 model 的层级结构：
      - input_size: tuple, summary() 的 input_size
      - outfile:    Path 或 str, 输出文本文件路径
    文本末尾自动追加 ReLU 激活与 Xavier Uniform 初始化说明。
    """
    s = summary(
        model,
        input_size=input_size,
        col_names=("input_size","output_size","num_params"),
        verbose=0
    )
    txt = s.__str__()
    txt += "\nActivation   : ReLU\n"
    txt += "Weight Init  : Xavier Uniform\n"
    Path(outfile).write_text(txt, encoding="utf-8")
    print(f"✅ 写入：{outfile}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 确保 results 目录存在
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # —— 4.1 导出分类模型 ——
    # 根据 src/models/classification_model.py 的 __init__ 签名来传参
    cls_model = SurgeryIndicatorModel(
        input_dim   = getattr(C, "INPUT_DIM",  C.INPUT_DIM),
        hidden_dims = getattr(C, "HIDDEN_SIZES", (64,128,64)),
        dropout     = getattr(C, "DROPOUT",      0.3),
        n_residual  = getattr(C, "N_RESIDUAL",    2),
        output_dim  = getattr(C, "OUTPUT_DIM",    8),
        use_cc      = getattr(C, "USE_CC",       False),
    )
    dump_model(
        model      = cls_model,
        input_size = (1, C.INPUT_DIM),
        outfile    = results_dir / "model_arch_classifier.txt"
    )

    # —— 4.2 导出回归模型 ——
    # RegressionModel 构造无需外部参数
    reg_model = RegressionModel()
    dump_model(
        model      = reg_model,
        input_size = (1, C.INPUT_DIM),
        outfile    = results_dir / "model_arch_regressor.txt"
    )

    print("🎉 分类与回归模型结构均已导出至 results/ 目录下。")
