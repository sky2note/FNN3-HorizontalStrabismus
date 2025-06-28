## 项目概述

本项目实现了用于水平斜视手术剂量预测的多输出深度回归与多标签分类模型，覆盖数据预处理、模型训练、超参数搜索、交叉验证、Bootstrap 不确定性评估及推断流程，旨在为临床提供高精度的手术规划辅助工具([docs.github.com][1], [github.com][2])。阅读本 README 可快速了解项目功能、环境搭建、使用方法及代码结构([makeareadme.com][3], [hackernoon.com][4])。

## 目录

* [功能亮点](#功能亮点)
* [环境与安装](#环境与安装)
* [数据预处理](#数据预处理)
* [模型结构](#模型结构)
* [训练流程](#训练流程)
* [交叉验证与评估](#交叉验证与评估)
* [Bootstrap 与不确定性](#bootstrap-与不确定性)
* [仓库结构](#仓库结构)
* [许可证](#许可证)
* [引用方式](#引用方式)

## 功能亮点

* **多标签分类+多输出回归**：同时预测 8 种手术指征和对应剂量，支持分类掩码回归([github.com][5])。
* **模块化代码**：配置（`src/config.py`）、数据预处理、模型定义、训练、评估、推断与可视化完全解耦，易于维护与扩展([medium.datadriveninvestor.com][6])。
* **可复现性保障**：统一随机种子、多折交叉验证、早停、学习率调度及 AMP 混合精度训练，结果稳定可重复。
* **可视化报告**：自动生成 ROC/PR 曲线、残差箱线图、子图网格，便于性能分析与论文插图([eheidi.dev][7])。
* **Bootstrap 置信区间**：提供 MAE、RMSE、R² 的置信区间估计，量化预测不确定性([hackernoon.com][4])。

## 环境与安装

1. 克隆仓库并进入目录：

   ```bash
   git clone https://github.com/your-repo/strabismus-surgery-ml.git  
   cd strabismus-surgery-ml  
   ```
2. 创建 Conda 环境并安装依赖：

   ```bash
   conda env create -f environment.yml    # 或 pip install -r requirements.txt
   ```
3. 下载原始数据至 `data/` 目录，并确保文件名与配置一致([docs.github.com][1])。

## 数据预处理

使用 `src/prepare_data.py` 中的 `preprocess_data_v2()`：

```bash
python -m src.prepare_data_v2 --raw data/Data_M1D6_pre1.csv --out data/Data_M1D6_Preproc.csv
```

该脚本完成缺失值填充、IQR 裁剪、派生特征计算（AxL\_mean、AxL\_diff、SphEq\_mean、SphEq\_diff）、标准化及列顺序调整([saiprakashspr.medium.com][8])。

## 模型结构

* **分类模型**：多标签 MLP（6→8，Sigmoid 输出），用于生成手术指征掩码([github.com][9])。
* **回归模型**：多输出 MLP（6→8，ReLU+Dropout 隐藏层，线性输出并 clamp 至 \[0,10] mm），结合分类掩码进行损失计算([git.wur.nl][10])。

## 训练流程

1. **超参数搜索**：

   ```bash
   python -m src.tuning.optuna_regression --trials 50  
   ```

   在第一折数据上以 RMSE 为目标进行 TPE 优化，并将最优参数保存至 `results/best_params_regression.json`([medium.com][11])。
2. **单折训练**：

   ```bash
   python -m src.train.train_regression --fold 0  
   ```

   包括混合精度训练、梯度裁剪、EarlyStopping 和学习率调度，最终模型保存在 `saved_models/reg_fold{fold}.pth`([hatica.io][12])。

## 交叉验证与评估

执行完整 10 折验证并生成性能文件：

```bash
python -m src.train.cross_validation  
```

结果保存为：

* `results/cv_metrics.csv` / `.json`（MAE、RMSE、R²、验证损失）
* `results/figs/fold*_err.png` & `acrossfold_residuals_grid.png`（残差箱线图）([github.com][2])。

## Bootstrap 与不确定性

```bash
python -m src.eval.bootstrap_metrics --n_boot 1000 --ci 0.95  
```

基于真值掩码和预先生成的 bootstrap 索引，计算并保存置信区间至 `results/bootstrap_ci.json`([makeareadme.com][3])。



该脚本加载所有折分类 & 回归模型，执行 ensemble + 冲突规则过滤，输出最终手术指征与剂量预测结果([github.com][5])。

## 仓库结构

```text
├── data/                     # 原始与预处理数据  
├── src/                      # 核心代码  
│   ├── config.py            # 配置管理  
│   ├── prepare_data.py      
│   ├── models/              
│   ├── train/               
│   ├── eval/                
│   ├── tuning/              
│   ├── utils/               
│   └── inference.py         
├── results/                  # CV、Bootstrap、Fig  
├── saved_models/             # 最优模型  
├── environment.yml           # 依赖环境  
└── README.md                 # 本文件  
```



## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件([docs.github.com][1])。

