from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import torch
    import torchaudio
    from torchaudio.compliance import kaldi as _kaldi
except Exception:  # pragma: no cover - optional fallback when torchaudio mismatches or is absent
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
    _kaldi = None

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
FRAME_BYTES = 320 * BYTES_PER_SAMPLE
DEFAULT_VAD_DBFS = -45.0


def read_wav(path: Path) -> bytes:
    """Decode a WAV, mix channels, and resample to the PCM interface format."""
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty or invalid audio: {path}")
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        if torchaudio is not None and torch is not None:
            mono = torchaudio.functional.resample(torch.from_numpy(mono), rate, SAMPLE_RATE).numpy()
        else:
            divisor = gcd(int(rate), SAMPLE_RATE)
            mono = resample_poly(mono, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    return np.clip(np.rint(mono * 32768), -32768, 32767).astype("<i2").tobytes()


def frames(pcm: bytes):
    for offset in range(0, len(pcm), FRAME_BYTES):
        yield pcm[offset : offset + FRAME_BYTES]


def open_microphone(device=None):
    import sounddevice as sd

    return sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_BYTES // 2, device=device
    )


def duration_ms(pcm: bytes) -> float:
    return len(pcm) * 1000 / BYTES_PER_SAMPLE / SAMPLE_RATE


def float_samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def activity(pcm: bytes, threshold: float = DEFAULT_VAD_DBFS) -> tuple[float, bool]:
    samples = float_samples(pcm)
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    dbfs = max(-120.0, 20 * float(np.log10(max(rms, 1e-6))))
    return round(dbfs, 2), dbfs >= threshold


def _numpy_kaldi_fbank(
    waveform: np.ndarray,
    *,
    num_mel_bins: int = 80,
    sample_frequency: int = SAMPLE_RATE,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    low_freq: float = 20.0,
    preemphasis: float = 0.97,
) -> np.ndarray:
    """Kaldi-compatible log Mel filterbank fallback when torchaudio is unavailable."""
    win_len = int(round(sample_frequency * frame_length_ms / 1000.0))
    hop_len = int(round(sample_frequency * frame_shift_ms / 1000.0))
    if waveform.size < win_len:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    n_frames = 1 + (waveform.size - win_len) // hop_len
    n_fft = 1 << (win_len - 1).bit_length()
    idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    framed = waveform[idx].astype(np.float32, copy=True)

    # Remove DC offset, apply preemphasis, and apply Povey window matching Kaldi defaults
    framed -= framed.mean(axis=1, keepdims=True)
    framed[:, 1:] = framed[:, 1:] - preemphasis * framed[:, :-1]
    framed[:, 0] *= 1.0 - preemphasis
    povey = np.power(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_len) / max(1, win_len - 1)), 0.85).astype(np.float32)
    framed *= povey[None, :]

    spec = np.fft.rfft(framed, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    nyquist = 0.5 * sample_frequency
    fft_freqs = np.linspace(0.0, nyquist, n_fft // 2 + 1, dtype=np.float32)
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    mel_points = np.linspace(mel(low_freq), mel(nyquist), num_mel_bins + 2, dtype=np.float32)
    left_mel = mel_points[:-2, None]
    center_mel = mel_points[1:-1, None]
    right_mel = mel_points[2:, None]
    fft_mels = mel(fft_freqs)[None, :]

    up = (fft_mels - left_mel) / np.maximum(center_mel - left_mel, 1e-6)
    down = (right_mel - fft_mels) / np.maximum(right_mel - center_mel, 1e-6)
    fb = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)

    energies = np.maximum(np.einsum("fk,mk->fm", power.astype(np.float64), fb.astype(np.float64)), 1e-10)
    return np.log(energies).astype(np.float32)


class _TensorLike:
    """Minimal wrapper exposing `.numpy()` when PyTorch is absent or avoided."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def numpy(self) -> np.ndarray:
        return self._arr


def speaker_fbank(pcm: bytes):
    samples = float_samples(pcm)
    if _kaldi is not None and torch is not None:
        waveform = torch.from_numpy(samples)
        return _kaldi.fbank(
            waveform.unsqueeze(0),
            num_mel_bins=80,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=SAMPLE_RATE,
            frame_length=25,
            frame_shift=10,
            snip_edges=True,
        )
    arr = _numpy_kaldi_fbank(samples)
    return torch.from_numpy(arr) if torch is not None else _TensorLike(arr)
