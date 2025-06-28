#!/usr/bin/env python3
# inference.py
"""
推理脚本 —— 加载分类与回归模型，生成手术指征与剂量预测

流程：
1. 读取输入特征（6 维），并使用训练时保存的 StandardScaler 做同样的预处理。
2. 对预处理后特征，使用所有折的分类模型做前向，平均概率，阈值化生成初步二值 mask。
3. 对每个样本的冲突预测（同一眼内直肌“缩短”与“后徙”同时为 1），只保留概率更高的一项。
4. 使用所有折的回归模型在 mask 掩码下做前向，平均预测剂量。
5. 将最终的手术指征（0/1）与手术剂量（mm）组合输出。

使用：
$ python inference.py --input features.csv --output predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import C
from src.models.classification_model import ClassificationModel
from src.models.regression_model    import RegressionModel


def load_scaler() -> joblib.load:
    scaler_path = C.SCALER_DIR / "standard.pkl"
    return joblib.load(scaler_path)


def load_models():
    """加载所有折的分类和回归模型到列表"""
    device = C.DEVICE
    cls_models = []
    reg_models = []
    for fold in range(C.KFOLD):
        # 分类模型
        cls = ClassificationModel().to(device)
        cls_ckpt = C.MODELS_DIR / "classification" / f"fold{fold}_best.pth"
        cls.load_state_dict(torch.load(cls_ckpt, map_location=device))
        cls.eval()
        cls_models.append(cls)
        # 回归模型
        reg = RegressionModel().to(device)
        reg_ckpt = C.MODELS_DIR / "regression" / f"fold{fold}_best.pth"
        reg.load_state_dict(torch.load(reg_ckpt, map_location=device))
        reg.eval()
        reg_models.append(reg)
    return cls_models, reg_models


def apply_rules(mask: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    对同一眼的内直肌“缩短”(s)和“后徙”(c)同时被预测为1的情况，保留更大概率项：
      - MRRsOD_Binary vs MRRcOD_Binary
      - MRRsOS_Binary vs MRRcOS_Binary
      - LRRsOD_Binary vs LRRcOD_Binary
      - LRRsOS_Binary vs LRRcOS_Binary
    """
    # 8 标签索引映射
    pairs = [
        (0, 4),  # MRRsOD vs MRRcOD
        (1, 5),  # MRRsOS vs MRRcOS
        (2, 6),  # LRRsOD vs LRRcOD
        (3, 7),  # LRRsOS vs LRRcOS
    ]
    for i, j in pairs:
        # 同时预测为 1 的样本
        both = (mask[:, i] == 1) & (mask[:, j] == 1)
        # 对于这些样本，比较概率，保留较大那一项
        mask[both, i] = (probs[both, i] >= probs[both, j]).astype(int)
        mask[both, j] = (probs[both, j] >  probs[both, i]).astype(int)
    return mask


def predict(
    X_raw: np.ndarray,
    scaler,
    cls_models: list,
    reg_models: list,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    对原始特征 X_raw 做推理并返回 DataFrame：
    - X_raw: (n_samples,6)
    """
    device = C.DEVICE
    # 1. 预处理
    X = scaler.transform(X_raw).astype("float32")
    X_t = torch.from_numpy(X).to(device)

    # 2. 分类 ensemble
    probs = np.zeros((X.shape[0], 8), dtype=float)
    with torch.no_grad():
        for m in cls_models:
            p = m(X_t, return_logits=False).cpu().numpy()
            probs += p
    probs /= len(cls_models)

    # 3. 阈值化
    mask = (probs >= threshold).astype(int)
    # 4. 规则过滤冲突
    mask = apply_rules(mask, probs)

    # 5. 回归 ensemble
    preds = np.zeros((X.shape[0], 8), dtype=float)
    with torch.no_grad():
        for m in reg_models:
            pred = m(X_t, mask=torch.from_numpy(mask).to(device).float())
            preds += pred.cpu().numpy()
    preds /= len(reg_models)
    # 裁剪到 [0,10]
    preds = np.clip(preds, 0.0, 10.0)

    # 6. 输出 DataFrame
    cols_bin = [
        "MRRsOD_Binary","MRRsOS_Binary","LRRsOD_Binary","LRRsOS_Binary",
        "MRRcOD_Binary","MRRcOS_Binary","LRRcOD_Binary","LRRcOS_Binary",
    ]
    cols_reg = [
        "MRRsOD","MRRsOS","LRRsOD","LRRsOS",
        "MRRcOD","MRRcOS","LRRcOD","LRRcOS",
    ]
    df = pd.DataFrame(mask, columns=cols_bin)
    for i, col in enumerate(cols_reg):
        df[col] = preds[:, i]
    return df


def main():
    parser = argparse.ArgumentParser(description="Strabismus surgery inference")
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="原始特征 CSV 文件，包含6列：Age, PrismCoverTest, AxL_mean, AxL_diff, SphEq_mean, SphEq_diff"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("predictions.csv"),
        help="保存预测结果的 CSV 文件路径"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="分类概率阈值（默认 0.5）"
    )
    args = parser.parse_args()

    # 加载数据
    raw = pd.read_csv(args.input)
    X_raw = raw[["Age","PrismCoverTest","AxL_mean","AxL_diff","SphEq_mean","SphEq_diff"]].values

    # 加载预处理器与模型
    scaler = load_scaler()
    cls_models, reg_models = load_models()

    # 预测
    df_pred = predict(X_raw, scaler, cls_models, reg_models, args.threshold)
    df_pred.to_csv(args.output, index=False)
    print(f"✅ 推理完成，结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
