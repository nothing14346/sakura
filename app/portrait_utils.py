from __future__ import annotations

import re
from pathlib import Path


def portrait_kind_key(path: Path) -> str:
    """从立绘文件名末尾的 A010/B120/I170 片段提取姿态种类。"""
    match = re.search(r"_([A-Za-z])\d+$", path.stem)
    if match is None:
        return ""
    return match.group(1).upper()


def should_crossfade_portrait(previous_path: Path, next_path: Path) -> bool:
    previous_kind = portrait_kind_key(previous_path)
    next_kind = portrait_kind_key(next_path)
    return bool(previous_kind and next_kind and previous_kind != next_kind)
