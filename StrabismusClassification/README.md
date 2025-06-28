```markdown
# Horizontal Strabismus Surgery Indicator Model
```
## 项目简介
```
本项目基于多标签分类方法，利用患者临床特征预测 8 种斜视手术指征。整体流程包含数据预处理、模型训练、交叉验证、阈值与概率校准、最终评估和可视化报告。
```
## 目录结构
```
.
├── apply\_thresholds.py         # 应用 β 校准 & 最优阈值评估
├── bootstrap\_eval.py           # Bootstrap 不确定性评估
├── calibrate\_temperature.py    # 温度缩放概率校准
├── config.py                   # 全局配置：特征、标签、路径
├── cross\_validation.py         # 10 折多标签分层 CV
├── dataset.py                  # 数据集定义与预处理
├── evaluate.py                 # 测试集推理与最终评估
├── generate\_all\_reports.py     # 自动生成报告
├── models/
│   └── mlp\_model.py            # MLP 模型定义（可选 Classifier-Chain）
├── optimise\_thresholds.py      # β 校准 + MCC 最优阈值搜索
├── plot\_results.py             # 绘制 ROC/PR/校准/阈值敏感度/混淆矩阵
├── scripts/
│   ├── make\_val\_split.py       # 验证集划分脚本
│   └── metrics\_summary.py      # CV 指标汇总脚本
├── train.py                    # 单折训练脚本（80/20 划分）
├── tree.py                     # 打印文件夹结构
├── utils/
│   ├── calibration.py          # Beta 校准实现
│   ├── delong.py               # DeLong AUC 置信区间
│   ├── losses.py               # FocalLoss 等
│   ├── metrics.py              # 多标签指标计算
│   ├── metrics\_ci.py           # 指标置信区间
│   └── uncertainty.py          # 不确定性评估工具
└── README.md                   # 本文件

````

## 安装与依赖
```
1. 克隆仓库：
   ```bash
   git clone https://github.com/YourUser/StrabismusClassification.git
   cd StrabismusClassification
````

2. 创建并激活虚拟环境（推荐 Python ≥ 3.8）：

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. 安装依赖：

   ```bash
   pip install -r requirements.txt[dataset.py](dataset.py)
   ```

   若无 `requirements.txt`，手动安装：

   ```bash
   pip install numpy pandas scikit-learn torch matplotlib iterstrat
   ```

## 数据准备

* 在 `data/` 目录放置：

  * `Data_M1D5_Preproc.csv`：训练集
  * `val.csv`、`test.csv`：验证集与测试集
  * `feature_scaler.pkl`：标准化器（可选）
* 如需重新预处理，请参看 `dataset.py` 中的逻辑。

## 配置说明

* **config.py** 中：

  * `FEATURE_COLS`：14 个输入特征（8 原始 + 6 派生）
  * `LABEL_COLS`：8 个目标标签
  * `DEFAULT_CSV`、`DEFAULT_VAL`、`DEFAULT_CKPT`：默认文件路径

## 训练与验证

1. **单折训练**

   ```bash
   python train.py \
     --csv data/Data_M1D5_Preproc.csv \
     --out best_model.pth \
     --epochs 300 \
     --batch_size 32 \
     --loss focal \
     --use_cc
   ```
2. **10 折交叉验证**

   ```bash
   python cross_validation.py \
     --csv data/Data_M1D5_Preproc.csv \
     --out-dir results/cv \
     --epochs 60 \
     --batch_size 32 \
     --use_cc
   ```

## 阈值与校准

1. **β 校准 + 阈值搜索**

   ```bash
   python optimise_thresholds.py \
     --cv-dir results/cv \
     --out best_thresholds.json
   ```
2. **温度缩放**

   ```bash
   python calibrate_temperature.py \
     --csv data/val.csv \
     --ckpt best_model.pth \
     --out calibration.json
   ```

## 最终评估

```bash
python evaluate.py \
  --csv-test data/test.csv \
  --ckpt best_model.pth \
  --thr best_thresholds.json \
  --temp calibration.json \
  --out eval_metrics.json
```

## 可视化与报告

* 自动生成图表：

  ```bash
  python plot_results.py
  ```
* 或运行：

  ```bash
  python generate_all_reports.py
  ```


```
```
