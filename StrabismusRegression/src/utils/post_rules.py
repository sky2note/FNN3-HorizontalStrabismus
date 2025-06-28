# src/utils/post_rules.py

"""
医疗结果后处理规则库
====================
对模型原始输出（pred_class, pred_reg）应用临床规则：
 1. 双眼对称：如果一侧手术被预测，且对侧未被预测，则镜像对侧预测，并同步剂量。
 2. R&R 互斥：在同一只眼上，内直肌“缩短”(s)与“后徙”(c)术式不可同时单独出现；
               若两者皆被预测，则保留剂量更大的术式，另一术式清零。
 3. 同步回归：任何被清零的分类标签，其对应的回归剂量也应置 0。

接口
----
apply_rules(pred_class, pred_reg) -> (adj_class, adj_reg)

参数
----
pred_class : np.ndarray, shape (n_samples, 8), dtype=int
    原始二值化分类输出，顺序为：
    [MRRsOD_Binary, MRRsOS_Binary, LRRsOD_Binary, LRRsOS_Binary,
     MRRcOD_Binary, MRRcOS_Binary, LRRcOD_Binary, LRRcOS_Binary]
pred_reg : np.ndarray, shape (n_samples, 8), dtype=float
    原始回归预测值，对应上述每个分类标签的手术剂量（mm）。

返回
----
adj_class : np.ndarray, shape (n_samples, 8), dtype=int
    经规则调整后的分类标签。
adj_reg : np.ndarray, shape (n_samples, 8), dtype=float
    经规则调整后的回归预测剂量。

注释解析

输入复制：保护原始 pred_class 和 pred_reg 不被覆盖。

索引映射

idx_pairs_eye：术式在左右眼之间的对称关系，用于镜像对侧的分类标签与剂量。

idx_rr_exclusive：同一只眼上“缩短”(s 或 e)与“后徙”(c 或 r)互斥关系。

双眼对称

若一侧分类为 1 而对侧为 0，则对侧也设为 1，并同步回归剂量：保证双眼对称。

R&R 互斥

若同一眼的两个互斥术式同时被预测，则比较两者的回归剂量，保留较大值对应的术式，清零另一个。

同步回归

最后一步，将所有分类标签为 0 的位置对应的回归剂量一并置零，以防模型在未做手术位置输出非零剂量。

将此文件保存为 src/utils/post_rules.py 后，在推理或后处理阶段即可调用：

python
复制
编辑
from src.utils.post_rules import apply_rules

adj_class, adj_reg = apply_rules(pred_class, pred_reg)




"""

import numpy as np


def apply_rules(
    pred_class: np.ndarray,
    pred_reg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 复制输入以免覆盖原数据
    cls = pred_class.copy()
    reg = pred_reg.copy()

    # 定义索引映射
    # 术式顺序: [sOD, sOS, eOD, eOS, cOD, cOS, rOD, rOS]
    # s = MRRs (shortening 内直肌缩短), c = MRRc (contraction 内直肌后徙)
    # e = LRRs (shortening 外直肌缩短), r = LRRc (recession 外直肌后徙)
    idx_pairs_eye = [
        # 内直肌对侧镜像
        (0, 1),  # MRRsOD vs MRRsOS
        # 外直肌对侧镜像
        (2, 3),  # LRRsOD vs LRRsOS
        # 内直肌后徙对侧镜像
        (4, 5),  # MRRcOD vs MRRcOS
        # 外直肌后徙对侧镜像
        (6, 7),  # LRRcOD vs LRRcOS
    ]
    # R&R 互斥对：每只眼上的 s 与 c 或 e 与 r
    idx_rr_exclusive = [
        (0, 4),  # MRRsOD vs MRRcOD
        (1, 5),  # MRRsOS vs MRRcOS
        (2, 6),  # LRRsOD vs LRRcOD
        (3, 7),  # LRRsOS vs LRRcOS
    ]

    n = cls.shape[0]
    # 1. 双眼对称：镜像单侧预测及剂量
    for i, j in idx_pairs_eye:
        # 对每个样本
        for k in range(n):
            if cls[k, i] == 1 and cls[k, j] == 0:
                cls[k, j] = 1
                reg[k, j] = reg[k, i]
            elif cls[k, j] == 1 and cls[k, i] == 0:
                cls[k, i] = 1
                reg[k, i] = reg[k, j]

    # 2. R&R 互斥：同眼 s/c 或 e/r 同时预测，保留剂量更大者
    for i, j in idx_rr_exclusive:
        for k in range(n):
            if cls[k, i] == 1 and cls[k, j] == 1:
                # 比较回归剂量
                if reg[k, i] >= reg[k, j]:
                    # 保留 i，清零 j
                    cls[k, j] = 0
                    reg[k, j] = 0.0
                else:
                    cls[k, i] = 0
                    reg[k, i] = 0.0

    # 3. 清零未做手术的回归预测
    #    确保任何 cls == 0 的位置，其 reg 也为 0
    mask_bin = cls.astype(bool)
    reg = reg * mask_bin

    return cls, reg
