#!/usr/bin/env python3
"""
Optuna TPE 超参数优化 — 分类模型（多组合记录版，修复 AUC 计算）
---------------------------------
功能：
1. 基于给定超参数在 10 折 StratifiedKFold 中训练分类模型，并记录平均 AUC。
2. 持续搜索，收集 **5** 组满足 AUC ≥ 阈值的超参数组合，并在每找到一组时立即保存到 JSON。
3. 最终生成包含 5 组配置的 JSON 文件。
用法：
$ python -m src.tuning.optuna_classification --n-trials 100 --threshold 0.9
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import C
from src.models.classification_model import ClassificationModel
from src.utils.data_utils import load_data


def objective(trial: optuna.trial.Trial) -> float:
    # 1. 采样超参数
    epochs         = trial.suggest_int("epochs",      50,  200)
    seed           = trial.suggest_int("seed",         1,  1000)
    batch_size     = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    lr             = trial.suggest_float("lr",         1e-5, 1e-2, log=True)
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    dropout        = trial.suggest_float("dropout",    0.0,  0.5)
    hidden1        = trial.suggest_int("hidden1",     32,   128)
    hidden2        = trial.suggest_int("hidden2",     16,    64)

    # 2. 加载数据与构建分层折
    X, y_class, _ = load_data()
    # 将多标签 one-hot 转为单标签用于分层
    stratify_labels = y_class.argmax(axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []

    # 3. 交叉验证
    for train_idx, val_idx in skf.split(X, stratify_labels):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_class[train_idx], y_class[val_idx]

        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
            batch_size=batch_size, shuffle=False
        )

        # 构建模型
        model = ClassificationModel(X.shape[1], (hidden1, hidden2), dropout).to(C.DEVICE)
        optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        model.train()
        for _ in range(epochs):
            for xb, yb in tr_loader:
                xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # 验证
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(C.DEVICE)
                preds.append(torch.sigmoid(model(xb)).cpu().numpy())
                trues.append(yb.numpy())
        y_true = np.vstack(trues)
        y_prob = np.vstack(preds)

        # 仅对每个标签中同时存在正负样本的维度计算 AUC
        valid_idx = [i for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) == 2]
        if not valid_idx:
            return 0.0  # 当前 trial 直接判为不合格
        try:
            auc_val = roc_auc_score(y_true[:, valid_idx], y_prob[:, valid_idx], average='macro')
        except ValueError:
            return 0.0
        aucs.append(auc_val)

    return float(np.mean(aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100,
                        help="每批搜索试验次数")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="AUC 阈值，满足则记录")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
    found = []  # 存储已记录的参数列表
    output_path = Path("results/best_params/best_params_classification.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 持续搜索直到收集到 5 组合
    while len(found) < 5:
        study.optimize(objective, n_trials=args.n_trials)
        # 遍历最新一批 trials
        for trial in study.trials[-args.n_trials:]:
            if trial.value is not None and trial.value >= args.threshold:
                params = trial.params
                if params not in found:
                    found.append(params)
                    with open(output_path, 'w') as f:
                        json.dump(found, f, indent=2)
                    print(f"记录第 {len(found)} 组超参数 (Trial {trial.number})，AUC={trial.value:.4f}")
    print(f"已收集 5 组满足 AUC ≥ {args.threshold} 的超参数，保存在 {output_path}")

if __name__ == "__main__":
    main()
