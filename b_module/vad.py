"""帧级语音活动判决。

与 A 模块 ``module_a/audio.py::activity()`` **同式同阈值**：

    rms  = sqrt(mean(x**2))          # x 为 [-1, 1) 的 float 采样
    dbfs = 20*log10(max(rms, 1e-6))  # 下限截到 -120 dBFS
    voiced = dbfs >= threshold       # 默认 -45 dBFS

回放模式下 is_speech 由这里算；与原机联调时应改用 A 事件里的 ``voiced``
字段（同样的公式，但多麦克风增益差异由 A 负责）。
"""

from __future__ import annotations

import numpy as np

DBFS_FLOOR = -120.0
MIN_DB = 1e-6


def frame_length(sample_rate: int, frame_ms: float) -> int:
    """一帧的采样点数。16 kHz / 20 ms -> 320。"""
    length = int(round(sample_rate * frame_ms / 1000.0))
    if length < 1:
        raise ValueError(f"帧长非法：sample_rate={sample_rate}, frame_ms={frame_ms}")
    return length


def frame_count(num_samples: int, sample_rate: int, frame_ms: float) -> int:
    """完整帧数（尾部不足一帧的采样被丢弃，与流式接口一致）。"""
    return num_samples // frame_length(sample_rate, frame_ms)


def frame_energies_db(samples: np.ndarray, sample_rate: int, frame_ms: float = 20.0) -> np.ndarray:
    """逐帧 RMS 能量（dBFS），公式与 A 的 activity() 一致。"""
    length = frame_length(sample_rate, frame_ms)
    usable = (len(samples) // length) * length
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    frames = np.asarray(samples[:usable], dtype=np.float64).reshape(-1, length)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, MIN_DB))
    db = np.maximum(db, DBFS_FLOOR)
    return np.round(db, 2)


def voiced_frames(energies_db: np.ndarray, threshold_dbfs: float) -> np.ndarray:
    """布尔数组：该帧是否判为语音。"""
    return np.asarray(energies_db) >= float(threshold_dbfs)


def speech_regions(
    voiced: np.ndarray,
    frame_ms: float = 20.0,
    min_gap_ms: float = 300.0,
    min_speech_ms: float = 0.0,
) -> list[tuple[int, int]]:
    """把连续语音帧聚成语音段，返回闭区间 ``[(start_frame, end_frame), ...]``。

    间隔大于 ``min_gap_ms`` 才算两个语音段；短于 ``min_speech_ms`` 的段被丢弃。
    """
    indices = np.flatnonzero(np.asarray(voiced, dtype=bool))
    if indices.size == 0:
        return []

    min_gap_frames = int(round(min_gap_ms / frame_ms))
    min_speech_frames = int(round(min_speech_ms / frame_ms))

    regions: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > max(min_gap_frames, 1):
            regions.append((start, previous))
            start = index
        previous = index
    regions.append((start, previous))

    return [(s, e) for s, e in regions if (e - s + 1) >= max(min_speech_frames, 1)]


def region_gaps_ms(regions: list[tuple[int, int]], frame_ms: float = 20.0) -> list[float]:
    """相邻语音段之间的间隔（毫秒），即句中停顿。"""
    return [
        float((regions[i + 1][0] - regions[i][1] - 1) * frame_ms)
        for i in range(len(regions) - 1)
    ]


def speech_end_frame(regions: list[tuple[int, int]]) -> int | None:
    """最后一段语音的结束帧（闭区间）。"""
    return regions[-1][1] if regions else None
