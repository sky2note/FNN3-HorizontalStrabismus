# src/utils/logger.py

"""
统一日志模块 —— 基于 Python 标准 logging

功能：
- 在控制台和文件中输出日志
- 支持按模块或全局设置日志级别
- 便于训练、评估等脚本统一记录运行信息

使用示例
--------
from src.utils.logger import setup_logger, logger

# 初始化日志（可在主脚本开头调用）
setup_logger(
    name="strabismus",
    log_file="logs/strabismus.log",
    level="INFO"
)

# 之后即可在任何文件中导入并使用
logger.info("开始第 1 折训练")
logger.error("出现错误：%s", err)
"""

import logging
import sys
from pathlib import Path


# 全局 logger 对象
logger = logging.getLogger("strabismus")


def setup_logger(
    name: str = "strabismus",
    log_file: str | Path = "logs/output.log",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    配置并返回全局 logger。

    参数
    ----
    name : 日志器名称（用于 logger = logging.getLogger(name)）
    log_file : 日志文件路径
    level : 日志级别，DEBUG/INFO/WARNING/ERROR/CRITICAL
    console : 是否在控制台输出
    file : 是否输出到文件
    fmt : 日志消息格式
    datefmt : 时间格式

    返回
    ----
    logger : 配置好的 logging.Logger 实例
    """
    # 创建日志目录
    log_path = Path(log_file)
    if file:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取或创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # 防止重复输出

    # 定义 Formatter
    formatter = logging.Formatter(fmt, datefmt)

    # 控制台 Handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # 文件 Handler
    if file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
