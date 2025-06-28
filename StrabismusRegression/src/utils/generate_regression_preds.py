#!/usr/bin/env python3
# src/utils/generate_regression_preds.py

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

# ─────────────────────────────────────────────────────────────────────────────
# 1. 计算 PROJECT_ROOT：本文件在 PROJECT_ROOT/src/utils/，向上两级即项目根
PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

from src.config import C
from src.models.regression_model import RegressionModel
from src.utils.data_utils import load_data
from src.utils.model_utils import set_seed


def main():
    # 2. 加载数据与折分器
    X, _, y_reg = load_data()
    kf = KFold(n_splits=C.KFOLD, shuffle=True, random_state=C.RANDOM_SEED)

    # 3. 输出目录准备
    out_dir = PROJECT_ROOT / "results" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 逐折推断
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        set_seed(C.RANDOM_SEED + fold)  # 每折不同随机种子，保持可复现

        # 4.1 实例化模型并加载对应折的权重
        model = RegressionModel().to(C.DEVICE)
        ckpt_path = PROJECT_ROOT / "saved_models" / f"reg_fold{fold}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到模型权重文件：{ckpt_path}")
        state = torch.load(ckpt_path, map_location=C.DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()

        # 4.2 对验证集做预测
        X_val = X[val_idx]
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float().to(C.DEVICE)).cpu().numpy()

        # 4.3 保存真实值与预测值
        np.savez(
            out_dir / f"fold{fold}_val_preds_regression.npz",
            y_true=y_reg[val_idx],
            y_pred=preds
        )
        print(f"✅ 折 {fold} 预测已保存至 {out_dir / f'fold{fold}_val_preds_regression.npz'}")


if __name__ == "__main__":
    main()
