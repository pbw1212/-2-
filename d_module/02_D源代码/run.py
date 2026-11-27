# -*- coding: utf-8 -*-
"""D 模块命令行入口。

用法示例（在 02_D源代码 目录下执行）：
    python run.py eval --out ../03_D实验记录与结果
    python run.py demo --out ../04_D演示
    python run.py replay --case c03_inherit_history --events
    python run.py chat
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dmodule.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
