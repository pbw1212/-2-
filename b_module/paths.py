"""统一的路径与配置入口。

模块 B 只读取 A/C/D/E 的文件，绝不写入。所有写操作都落在 b_module/ 内。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

MODULE_DIR = Path(__file__).resolve().parent               # b_module/
REPO_ROOT = MODULE_DIR.parent                              # 仓库根目录
DEFAULT_A_ROOT = REPO_ROOT / "a_module"
CONFIG_PATH = MODULE_DIR / "config" / "endpoint.json"
DATA_DIR = MODULE_DIR / "data"
RESULTS_DIR = MODULE_DIR / "results"
CACHE_DIR = MODULE_DIR / ".cache"

MANIFEST_PATH = DATA_DIR / "manifest.json"
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """读取 endpoint.json。返回值按路径缓存，避免每帧都读盘。"""
    config_path = Path(path) if path else CONFIG_PATH
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"不支持的配置版本：{config.get('schema_version')}")
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
