from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.compliance import kaldi

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
        mono = torchaudio.functional.resample(torch.from_numpy(mono), rate, SAMPLE_RATE).numpy()
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


def speaker_fbank(pcm: bytes) -> torch.Tensor:
    waveform = torch.from_numpy(float_samples(pcm))
    return kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=80,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=SAMPLE_RATE,
        frame_length=25,
        frame_shift=10,
        snip_edges=True,
    )
