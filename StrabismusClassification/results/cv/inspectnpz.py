import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def inspect_npz(filepath: Path) -> Dict[str, Tuple[int, ...]]:
    """
    加载 .npz 文件并返回其中所有数组的名称及形状。

    参数
    ----
    filepath : Path
        .npz 文件路径

    返回
    ----
    Dict[str, Tuple[int, ...]]
        键为数组名称，值为对应的 shape 元组
    """
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")
    data = np.load(filepath)
    shapes = {name: data[name].shape for name in data.files}
    return shapes

# 示例调用
shapes = inspect_npz(Path("fold0_val_preds.npz"))
print("包含的数组及其形状：")
for name, shape in shapes.items():
    print(f"  - {name}: {shape}")
