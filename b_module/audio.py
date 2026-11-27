"""WAV 读写与整轮音频重建。

只依赖标准库 ``wave`` 与 numpy，不引入 soundfile/torch——B 需要在树莓派上保持轻量。

三条与 A 对齐的约定：
* 内部一律用 ``float32``、单声道、取值 [-1, 1)，与 ``module_a.audio.float_samples`` 一致；
* 帧长 20 ms，帧能量公式见 :mod:`b_module.vad`；
* 重建整轮音频时先按 Detector 同阈值（-45 dBFS）裁掉每个分段首尾静音，
  再插入**恰好** ``pause_ms`` 的静音，否则 TTS 自带的尾静音会让句中停顿虚高。
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import vad


class AudioFormatError(ValueError):
    """WAV 格式不受支持。"""


@dataclass(frozen=True)
class WavInfo:
    path: Path
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration_ms(self) -> float:
        return self.frames * 1000.0 / self.sample_rate


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    """把 PCM 字节解码成 [-1, 1) 的 float32。"""
    if sample_width == 1:  # 8-bit 无符号
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:  # 24-bit，wave 模块不直接支持，手工拼接
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        value = np.where(value >= (1 << 23), value - (1 << 24), value)
        return value.astype(np.float32) / float(1 << 23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioFormatError(f"不支持的采样位宽：{sample_width * 8} bit")


def read_wav(path: str | Path, *, mono: bool = True) -> tuple[np.ndarray, int]:
    """读取 WAV，返回 ``(samples float32, sample_rate)``。多声道默认混成单声道。"""
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1) if mono else samples
    if samples.size == 0:
        raise AudioFormatError(f"空音频：{path}")
    if not np.isfinite(samples).all():
        raise AudioFormatError(f"音频含非法采样值：{path}")
    return samples.astype(np.float32, copy=False), sample_rate


def wav_info(path: str | Path) -> WavInfo:
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        return WavInfo(
            path=path,
            sample_rate=handle.getframerate(),
            channels=handle.getnchannels(),
            sample_width=handle.getsampwidth(),
            frames=handle.getnframes(),
        )


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """写出 16 bit 单声道 WAV（内部缓存用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return path


def duration_ms(num_samples: int, sample_rate: int) -> float:
    return num_samples * 1000.0 / sample_rate


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    return int(round(sample_rate * float(milliseconds) / 1000.0))


def silence(milliseconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(ms_to_samples(max(milliseconds, 0.0), sample_rate), dtype=np.float32)


def trim_to_speech(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
    frame_ms: float = 20.0,
    margin_ms: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """裁掉首尾静音，返回 ``(trimmed, meta)``。

    ``meta`` 记录裁掉的首尾时长，便于标注时还原时间轴。
    """
    energies = vad.frame_energies_db(samples, sample_rate, frame_ms)
    voiced = vad.voiced_frames(energies, threshold_dbfs)
    indices = np.flatnonzero(voiced)
    length = vad.frame_length(sample_rate, frame_ms)

    if indices.size == 0:
        return np.zeros(0, dtype=np.float32), {
            "speech_found": False,
            "lead_trim_ms": duration_ms(len(samples), sample_rate),
            "tail_trim_ms": 0.0,
        }

    margin = ms_to_samples(margin_ms, sample_rate)
    start = max(int(indices[0]) * length - margin, 0)
    end = min((int(indices[-1]) + 1) * length + margin, len(samples))
    trimmed = samples[start:end]
    return trimmed, {
        "speech_found": True,
        "lead_trim_ms": duration_ms(start, sample_rate),
        "tail_trim_ms": duration_ms(len(samples) - end, sample_rate),
    }


def concat_with_pauses(
    segments: list[np.ndarray],
    pauses_ms: list[float],
    sample_rate: int,
) -> tuple[np.ndarray, list[float]]:
    """按给定停顿拼接若干语音段，返回 ``(samples, offsets_ms)``。

    ``offsets_ms[i]`` 是第 i 段在拼接结果中的起始时间。
    """
    if not segments:
        raise ValueError("至少需要一个语音段")
    if len(pauses_ms) != len(segments) - 1:
        raise ValueError("停顿数必须等于段数减一")

    offsets: list[float] = []
    chunks: list[np.ndarray] = []
    cursor = 0
    for index, segment in enumerate(segments):
        if index > 0:
            gap = silence(pauses_ms[index - 1], sample_rate)
            chunks.append(gap)
            cursor += len(gap)
        offsets.append(duration_ms(cursor, sample_rate))
        chunks.append(segment)
        cursor += len(segment)
    return np.concatenate(chunks), offsets


def pad_trailing_silence(samples: np.ndarray, sample_rate: int, tail_ms: float) -> np.ndarray:
    """在末尾补固定长度静音，让所有阈值都能在回放结束前触发。"""
    if tail_ms <= 0:
        return samples
    return np.concatenate([samples, silence(tail_ms, sample_rate)])


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """线性重采样（仅在显式需要统一采样率时使用；VAD 本身与采样率无关）。"""
    if source_rate == target_rate:
        return samples
    if samples.size == 0:
        return samples
    target_length = int(round(len(samples) * target_rate / source_rate))
    source_axis = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    target_axis = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_axis, source_axis, samples).astype(np.float32)


def trailing_silence_ms(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
    frame_ms: float = 20.0,
) -> float:
    """末尾静音时长（按同一 VAD 口径）。"""
    energies = vad.frame_energies_db(samples, sample_rate, frame_ms)
    if energies.size == 0:
        return duration_ms(len(samples), sample_rate)
    voiced = vad.voiced_frames(energies, threshold_dbfs)
    indices = np.flatnonzero(voiced)
    if indices.size == 0:
        return duration_ms(len(samples), sample_rate)
    end = (int(indices[-1]) + 1) * vad.frame_length(sample_rate, frame_ms)
    return duration_ms(len(samples) - end, sample_rate)


def total_ms(samples: np.ndarray, sample_rate: int) -> float:
    return duration_ms(len(samples), sample_rate)


def frames_for_ms(milliseconds: float, frame_ms: float) -> int:
    return int(math.ceil(milliseconds / frame_ms))
