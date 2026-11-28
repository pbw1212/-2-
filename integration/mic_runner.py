"""Microphone capture with ambient noise-floor calibration and half-duplex gating."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
A_PKG_ROOT = PROJECT_ROOT / "a_module"
if str(A_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(A_PKG_ROOT))


class HalfDuplexMicrophone:
    """16 kHz mono PCM frame source wired to EModuleService mic pause/flush callbacks."""

    def __init__(self, *, device: Optional[str | int] = None) -> None:
        self.device = device
        self._stream = None
        self._paused = True

    def open(self) -> None:
        from module_a.audio import FRAME_BYTES, open_microphone

        if self._stream is None:
            self._stream = open_microphone(device=self.device)
            self._stream.start()
        self._frame_samples = FRAME_BYTES // 2

    def pause(self, _turn_id: int) -> None:
        self._paused = True

    def flush_and_open(self, _turn_id: int) -> None:
        """Discard any residual speaker audio buffered during TTS playback before listening."""
        if self._stream is None:
            return
        avail = getattr(self._stream, "read_available", 0)
        while avail and avail > 0:
            to_read = min(avail, self._frame_samples * 8)
            self._stream.read(to_read)
            avail = getattr(self._stream, "read_available", 0)
        self._paused = False

    def calibrate_noise_floor(self, duration_ms: int = 800) -> float:
        """Measure ambient room noise over ~0.8s and recommend a robust VAD dBFS threshold."""
        from module_a.audio import activity

        self.open()
        n_frames = max(5, int(round(duration_ms / 20.0)))
        levels = []
        for _ in range(n_frames):
            raw, _ = self._stream.read(self._frame_samples)
            dbfs, _ = activity(bytes(raw))
            levels.append(dbfs)
        median_floor = float(np.median(levels))
        return round(max(-45.0, min(-25.0, median_floor + 8.0)), 2)

    def iter_frames(self, *, max_frames: int = 1600) -> Iterator[bytes]:
        if self._stream is None:
            self.open()
        for _ in range(max_frames):
            if self._paused:
                break
            raw, _ = self._stream.read(self._frame_samples)
            yield bytes(raw)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def enroll_demo_profiles(db_path: Path) -> Path:
    """Populate CAM++ simulated speaker profiles (p001-p004) from sample audio if empty."""
    from module_a.audio import read_wav
    from module_a.speaker import DEFAULT_MODEL, SpeakerMatcher

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.is_file():
        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
            if count > 0:
                return db_path
        except sqlite3.Error:
            pass
        finally:
            conn.close()

    smoke = A_PKG_ROOT / "examples" / "smoke"
    matcher = SpeakerMatcher(DEFAULT_MODEL, db_path)
    p1_wav = smoke / "speaker1_a_cn_16k.wav"
    p2_wav = smoke / "speaker2_a_cn_16k.wav"
    if p1_wav.is_file():
        matcher.enroll("p001", [read_wav(p1_wav)])
    if p2_wav.is_file():
        matcher.enroll("p002", [read_wav(p2_wav)])
    return db_path
