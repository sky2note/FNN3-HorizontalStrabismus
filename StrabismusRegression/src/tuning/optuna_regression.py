
#!/usr/bin/env python
# src/tuning/optuna_regression.py

"""
Optuna TPE 超参数优化 — 回归模型
---------------------------------
功能：
1. 定义超参数搜索空间：包括隐藏层宽度、dropout、学习率、batch_size 等。
2. 在每次 trial 中，将采样到的超参数写回到 C.HIDDEN_SIZES、C.DROPOUT，然后实例化 RegressionModel。
3. 在第一折数据上训练若干 epoch，在验证集上以 RMSE 最小化为优化目标。
4. 完成所有 trial 后，将最优超参数保存到 results/best_params_regression.json，并持久化到 regression.db。
"""

import sys
from pathlib import Path

# ─── 将项目根目录加入 Python 搜索路径 ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import json
import optuna
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from src.config import C
from src.models.regression_model import RegressionModel
from src.utils.data_utils import load_data
from src.utils.metrics import reg_mae, reg_rmse, reg_r2
from src.utils.model_utils import set_seed


def objective(trial: optuna.trial.Trial) -> float:
    # 1. 采样超参数
    h1 = trial.suggest_int("hidden1", 32, 128)
    h2 = trial.suggest_int("hidden2", 16, 64)
    dp = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 2. 把采样结果写回全局 C
    C.HIDDEN_SIZES = (h1, h2)
    C.DROPOUT      = dp

    # 3. 加载并划分数据（仅第一折）
    X, _, y_reg = load_data()
    kf = KFold(n_splits=C.KFOLD, shuffle=True, random_state=C.RANDOM_SEED)
    train_idx, val_idx = next(kf.split(X))
    X_tr, y_tr = X[train_idx], y_reg[train_idx]
    X_va, y_va = X[val_idx],   y_reg[val_idx]

    # 4. DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    # 5. 固定随机种子
    set_seed(C.RANDOM_SEED)

    # 6. 实例化模型（无参）+ 优化器
    device = C.DEVICE
    model = RegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 7. 训练若干 epoch
    for _ in range(10):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 8. 评估：计算 RMSE
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device).float()
            preds.append(model(xb).cpu().numpy())
            targets.append(yb.numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(targets)

    # 9. 返回 RMSE（最小化目标）
    return reg_rmse(y_true, y_pred)


def main():
    # 1. 创建/加载带持久化的 Study
    study = optuna.create_study(
        study_name="opt_reg",
        storage="sqlite:///regression.db",
        sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED),
        direction="minimize",
        load_if_exists=True,
    )

    # 2. 开始优化
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 3. 保存最优超参数
    best = study.best_trial.params
    out_dir = PROJECT_ROOT / "results" / "best_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_params_regression.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存最佳回归超参数：{out_dir / 'best_params_regression.json'}")

    # 4. 快速查看几条 trial
    print(study.trials_dataframe().head())


if __name__ == "__main__":
    main()













# ==============================  BK  ===========================================================



# # src/tuning/optuna_regression.py
#
# """
# Optuna TPE 超参数优化 — 回归模型
# ---------------------------------
# 功能：
# 1. 定义超参数搜索空间：包括隐藏层宽度、dropout、学习率、batch_size 等。
# 2. 在每次 trial 中，用采样到的超参数实例化 RegressionModel，并在第一折数据上训练若干 epoch。
# 3. 在同一折的验证集上评估回归性能（RMSE），以“最小化 RMSE”为优化目标。
# 4. 完成所有 trial 后，将最优超参数保存到 results/best_params_regression.json。
# """
#
# import json
# from pathlib import Path
#
# import optuna
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import KFold
#
# from src.config import C
# from src.models.regression_model import RegressionModel
# from src.utils.data_utils import load_data
# from src.utils.metrics import compute_regression_metrics
# from src.utils.model_utils import set_seed
#
#
# def objective(trial: optuna.trial.Trial) -> float:
#     # 1. 超参数采样
#     hidden1    = trial.suggest_int("hidden1", 32, 128)
#     hidden2    = trial.suggest_int("hidden2", 16, 64)
#     dropout    = trial.suggest_float("dropout", 0.0, 0.5)
#     lr         = trial.suggest_loguniform("lr", 1e-4, 1e-2)
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
#
#     # 2. 加载数据并做简单  KFold 划分（只取第一折做快速评估）
#     X, _, y_reg = load_data()
#     kf = KFold(n_splits=C.KFOLD, shuffle=True, random_state=C.RANDOM_SEED)
#     train_idx, val_idx = next(kf.split(X))
#     X_train, y_train = X[train_idx], y_reg[train_idx]
#     X_val,   y_val   = X[val_idx],   y_reg[val_idx]
#
#     # 3. 构造 DataLoader
#     train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
#     val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
#
#     # 4. 固定随机种子
#     set_seed(C.RANDOM_SEED)
#
#     # 5. 实例化模型、损失与优化器
#     device = C.DEVICE
#     model = RegressionModel(input_dim=6, hidden_dims=(hidden1, hidden2), dropout=dropout).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     # 6. 快速训练若干 epoch
#     epochs = 10
#     for _ in range(epochs):
#         model.train()
#         for Xb, yb in train_loader:
#             Xb, yb = Xb.to(device).float(), yb.to(device).float()
#             optimizer.zero_grad()
#             preds = model(Xb)  # 不使用 mask
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#
#     # 7. 验证并计算 RMSE
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for Xb, yb in val_loader:
#             Xb = Xb.to(device).float()
#             preds = model(Xb).cpu().numpy()
#             all_preds.append(preds)
#             all_targets.append(yb.numpy())
#     y_pred = np.vstack(all_preds)
#     y_true = np.vstack(all_targets)
#
#     mae, rmse, r2 = compute_regression_metrics(y_true, y_pred)
#     return rmse  # 以 RMSE 最小化为目标
#
#
# def main():
#     study = optuna.create_study(
#         direction="minimize",
#         sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED),
#     )
#     study.optimize(objective, n_trials=50, show_progress_bar=True)
#
#     best_params = study.best_trial.params
#     out_dir = Path("results") / "best_params"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / "best_params_regression.json"
#     with open(out_path, "w") as f:
#         json.dump(best_params, f, indent=2)
#     print(f"最佳回归超参数已保存到 {out_path}")
#
#     # 可选：打印搜索历史前几条
#     print(study.trials_dataframe().head())
#
#
# if __name__ == "__main__":
#     main()
