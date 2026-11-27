"""音频与 VAD 测试：用合成信号验证分帧、能量、语音段、裁剪与拼接。"""

from __future__ import annotations

import numpy as np
import pytest

from b_module import audio, vad

SR = 16_000
FRAME_MS = 20.0


def tone(duration_ms: float, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    count = audio.ms_to_samples(duration_ms, SR)
    t = np.arange(count) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(duration_ms: float) -> np.ndarray:
    return np.zeros(audio.ms_to_samples(duration_ms, SR), dtype=np.float32)


def test_frame_length_and_count():
    assert vad.frame_length(SR, 20.0) == 320
    assert vad.frame_length(22_050, 20.0) == 441
    assert vad.frame_count(320 * 5 + 7, SR, 20.0) == 5


def test_energy_levels():
    energy = vad.frame_energies_db(tone(100), SR, FRAME_MS)
    # 0.3 幅值正弦的 RMS 约 0.212 -> 约 -13.5 dBFS
    assert energy.size == 5
    assert np.all(energy > -20) and np.all(energy < -5)

    quiet = vad.frame_energies_db(silence(100), SR, FRAME_MS)
    assert np.all(quiet == -120.0)


def test_voiced_frames_threshold():
    samples = np.concatenate([silence(200), tone(200), silence(200)])
    energies = vad.frame_energies_db(samples, SR, FRAME_MS)
    voiced = vad.voiced_frames(energies, -45.0)
    assert voiced.size == 30
    assert voiced[:10].sum() == 0
    assert voiced[10:20].all()
    assert voiced[20:].sum() == 0


def test_speech_regions_merge_short_gaps():
    voiced = np.zeros(60, dtype=bool)
    voiced[5:10] = True
    voiced[24:29] = True  # 间隔 14 帧 = 280 ms < 300 ms -> 合并
    assert len(vad.speech_regions(voiced, FRAME_MS, min_gap_ms=300)) == 1

    voiced2 = np.zeros(60, dtype=bool)
    voiced2[5:10] = True
    voiced2[30:35] = True  # 间隔 19 帧 = 380 ms > 300 ms -> 两段
    regions = vad.speech_regions(voiced2, FRAME_MS, min_gap_ms=300)
    assert regions == [(5, 9), (30, 34)]
    assert vad.region_gaps_ms(regions, FRAME_MS) == [pytest.approx(400.0)]


def test_trim_to_speech_removes_edges():
    samples = np.concatenate([silence(500), tone(400), silence(500)])
    trimmed, meta = audio.trim_to_speech(samples, SR, threshold_dbfs=-45.0, frame_ms=FRAME_MS)
    assert meta["speech_found"] is True
    assert abs(audio.duration_ms(len(trimmed), SR) - 400.0) < FRAME_MS * 1.5
    assert meta["lead_trim_ms"] == pytest.approx(500.0, abs=FRAME_MS)
    assert meta["tail_trim_ms"] == pytest.approx(500.0, abs=FRAME_MS)


def test_trim_on_pure_silence():
    trimmed, meta = audio.trim_to_speech(silence(300), SR, threshold_dbfs=-45.0)
    assert trimmed.size == 0
    assert meta["speech_found"] is False


def test_concat_with_pauses_offsets():
    segments = [tone(500), tone(500)]
    rebuilt, offsets = audio.concat_with_pauses(segments, [300.0], SR)
    assert offsets == [0.0, pytest.approx(800.0, abs=1.0)]
    assert abs(audio.duration_ms(len(rebuilt), SR) - 1300.0) < 1.0


def test_concat_rejects_mismatched_pauses():
    with pytest.raises(ValueError):
        audio.concat_with_pauses([tone(100), tone(100)], [], SR)


def test_pad_trailing_silence():
    padded = audio.pad_trailing_silence(tone(200), SR, 2000.0)
    assert abs(audio.duration_ms(len(padded), SR) - 2200.0) < 1.0
    assert audio.trailing_silence_ms(padded, SR, threshold_dbfs=-45.0) == pytest.approx(2000.0, abs=FRAME_MS)


def test_wav_roundtrip(tmp_path):
    samples = np.concatenate([silence(100), tone(300)])
    path = audio.write_wav(tmp_path / "probe.wav", samples, SR)
    reloaded, rate = audio.read_wav(path)
    assert rate == SR
    assert reloaded.shape == samples.shape
    assert np.max(np.abs(reloaded - samples)) < 1e-3

    info = audio.wav_info(path)
    assert info.sample_rate == SR
    assert info.channels == 1
    assert info.sample_width == 2


def test_rejects_unsupported_width(tmp_path):
    import wave

    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(b"")
    with pytest.raises(audio.AudioFormatError):
        audio.read_wav(path)
