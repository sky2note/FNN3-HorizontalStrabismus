#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tree.py — 打印指定目录（默认当前目录）的树状结构

用法：
    python tree.py [路径]
例：
    python tree.py .
"""

import os
import sys


def print_tree(path: str, prefix: str = "") -> None:
    """
    以树状结构打印给定路径下的所有文件和文件夹。

    参数：
      path   — 要遍历的目录路径
      prefix — 用于递归时的行首前缀，内部自动管理缩进和分支符号
    """
    try:
        entries = sorted(os.listdir(path), key=lambda s: s.lower())
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    for idx, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = "└── " if idx == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(full_path):
            extension = "    " if idx == len(entries) - 1 else "│   "
            print_tree(full_path, prefix + extension)


if __name__ == "__main__":
    # 如果用户指定了路径则使用，否则默认当前目录
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(root)
    print_tree(root)
