"""Attention-guided Clear Speech acoustic modulation for elderly clinic TTS.

Implements:
1. Tacotron2 Attention alignment quality metrics (focus & monotonicity).
2. Alignment-guided mapping from pinyin character spans to Mel spectrogram frames.
3. Pitch-preserving local Mel time-stretching on critical medical entities.
4. Energy-based segment silence trimming + calibrated pause insertion + 1.5-4 kHz clarity EQ.
5. Simulated elderly hearing-loss + clinic noise channel for robust intelligibility evaluation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from e_module.config import SAMPLE_RATE

# Exact symbol set matching tacotron2/text/symbols.py + basic_cleaners
_VALID_SYMBOLS = set("-!'(),.:;? abcdefghijklmnopqrstuvwxyz1234567890")


def compute_attention_metrics(alignment: np.ndarray) -> Dict[str, float]:
    """Compute attention focus and monotonicity scores on (T_in, T_dec) matrix."""
    if alignment.size == 0 or alignment.ndim != 2 or alignment.shape[1] == 0:
        return {"focus_score": 0.0, "monotonicity_score": 0.0}

    # Focus: average peak attention weight per decoder step
    max_weights = np.max(alignment, axis=0)
    focus = float(np.mean(max_weights))

    # Monotonicity: fraction of decoder steps whose argmax encoder index >= previous - 1
    peaks = np.argmax(alignment, axis=0)
    if len(peaks) <= 1:
        mono = 1.0
    else:
        diffs = np.diff(peaks)
        # Allow tiny 0/+step jumps; penalize backward jumps (< -1) or huge forward skips
        good_steps = np.logical_and(diffs >= -1, diffs <= 4)
        mono = float(np.mean(good_steps))

    return {
        "focus_score": round(focus, 4),
        "monotonicity_score": round(mono, 4),
    }


def _cleaned_symbol_index_map(raw_pinyin: str) -> List[Optional[int]]:
    """Map each character index in raw_pinyin to its index in Tacotron2's cleaned symbol sequence."""
    # Replicates basic_cleaners (lowercase + collapse whitespace) + _should_keep_symbol + EOS
    cleaned_chars: List[int] = []
    # Map raw char index -> kept symbol index
    char_to_sym: List[Optional[int]] = [None] * len(raw_pinyin)

    prev_space = False
    sym_idx = 0
    for i, ch in enumerate(raw_pinyin.lower()):
        if ch.isspace():
            if not prev_space:
                if " " in _VALID_SYMBOLS:
                    char_to_sym[i] = sym_idx
                    sym_idx += 1
                    cleaned_chars.append(i)
                prev_space = True
        else:
            prev_space = False
            if ch in _VALID_SYMBOLS:
                char_to_sym[i] = sym_idx
                sym_idx += 1
                cleaned_chars.append(i)
    return char_to_sym


def locate_entity_mel_frames(
    alignment: np.ndarray,
    raw_pinyin: str,
    entity_char_spans: List[Tuple[int, int, str]],
    n_frames_per_step: int = 3,
) -> List[Tuple[int, int, str]]:
    """Map character-level entity spans in raw_pinyin to Mel frame intervals [mel_s, mel_e)."""
    if alignment.size == 0 or not entity_char_spans:
        return []

    t_in, t_dec = alignment.shape
    char_to_sym = _cleaned_symbol_index_map(raw_pinyin)
    peaks = np.argmax(alignment, axis=0)  # (T_dec,)

    mel_spans: List[Tuple[int, int, str]] = []
    for c_s, c_e, label in entity_char_spans:
        sym_indices = [
            char_to_sym[idx]
            for idx in range(max(0, c_s), min(len(raw_pinyin), c_e))
            if char_to_sym[idx] is not None
        ]
        if not sym_indices:
            continue
        sym_s = max(0, min(sym_indices))
        sym_e = min(t_in - 1, max(sym_indices))

        # Decoder steps whose attention peak lands inside [sym_s, sym_e]
        matched_steps = np.where(np.logical_and(peaks >= sym_s, peaks <= sym_e))[0]
        if matched_steps.size == 0:
            # Fallback: use cumulative attention mass center
            mass = np.sum(alignment[sym_s : sym_e + 1, :], axis=0)
            matched_steps = np.where(mass > 0.25)[0]
        if matched_steps.size == 0:
            continue

        dec_s = int(matched_steps[0])
        dec_e = int(matched_steps[-1]) + 1
        mel_s = dec_s * n_frames_per_step
        mel_e = dec_e * n_frames_per_step
        if mel_e > mel_s:
            mel_spans.append((mel_s, mel_e, label))

    # Merge overlapping spans
    mel_spans.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int, str]] = []
    for s, e, lbl in mel_spans:
        if merged and s <= merged[-1][1]:
            ps, pe, plbl = merged[-1]
            merged[-1] = (ps, max(pe, e), plbl)
        else:
            merged.append((s, e, lbl))
    return merged


def stretch_mel_local(
    mel: np.ndarray,
    mel_spans: List[Tuple[int, int, str]],
    stretch_factor: float = 1.18,
    ramp_frames: int = 3,
) -> Tuple[np.ndarray, float]:
    """Time-stretch selected Mel frame spans locally while preserving frequency axis (pitch).

    Args:
        mel: Array of shape (80, T_mel).
        mel_spans: List of (mel_start, mel_end, label).
        stretch_factor: >1.0 slows down speech in entity spans (e.g., 1.18 = ~15% slower).
        ramp_frames: Smooth cosine ramp across span boundaries.

    Returns:
        (stretched_mel, realized_entity_stretch_ratio)
    """
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel (n_mels, T), got {mel.shape}")
    n_mels, t_orig = mel.shape
    if t_orig < 4 or not mel_spans or abs(stretch_factor - 1.0) < 1e-3:
        return mel.copy(), 1.0

    # Local expansion rate r[t] = dt_new / dt_orig (>= 1.0 inside entity spans)
    rate = np.ones(t_orig, dtype=np.float64)
    entity_mask = np.zeros(t_orig, dtype=bool)
    for s, e, _ in mel_spans:
        s_c = max(0, s)
        e_c = min(t_orig, e)
        if e_c <= s_c:
            continue
        rate[s_c:e_c] = stretch_factor
        entity_mask[s_c:e_c] = True

        # Cosine transition before start and after end
        if ramp_frames > 0:
            r_pre_s = max(0, s_c - ramp_frames)
            if s_c > r_pre_s:
                w = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, s_c - r_pre_s, endpoint=False)))
                rate[r_pre_s:s_c] = np.maximum(rate[r_pre_s:s_c], 1.0 + (stretch_factor - 1.0) * w)
            r_post_e = min(t_orig, e_c + ramp_frames)
            if r_post_e > e_c:
                w = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0.0, 1.0, r_post_e - e_c, endpoint=False)))
                rate[e_c:r_post_e] = np.maximum(rate[e_c:r_post_e], 1.0 + (stretch_factor - 1.0) * w)

    # Map cumulative output frame axis back to continuous original frame coordinates
    cum_out = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    total_out = float(cum_out[-1] + rate[-1])
    t_new = max(int(round(total_out)), t_orig)
    query_out = np.linspace(0.0, cum_out[-1], t_new)
    src_coords = np.interp(query_out, cum_out, np.arange(t_orig, dtype=np.float64))

    idx0 = np.floor(src_coords).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_orig - 1)
    alpha = (src_coords - idx0).astype(np.float32)[None, :]
    stretched = (1.0 - alpha) * mel[:, idx0] + alpha * mel[:, idx1]

    orig_entity_frames = int(np.sum(entity_mask))
    stretched_entity_frames = float(np.sum(rate[entity_mask]))
    realized_ratio = stretched_entity_frames / max(orig_entity_frames, 1)
    return stretched.astype(np.float32), round(realized_ratio, 4)


def trim_segment_silence(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    top_db: float = 32.0,
    keep_margin_ms: float = 6.0,
) -> np.ndarray:
    """Trim variable leading/trailing silence generated by Tacotron2 so pauses are calibrated."""
    if wav.size == 0:
        return wav
    frame_len = int(round(0.010 * sr))
    hop_len = int(round(0.002 * sr))
    if len(wav) <= frame_len:
        return wav

    frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1) + 1e-12)
    peak_rms = float(np.max(rms))
    if peak_rms < 1e-6:
        return wav

    thresh = peak_rms * (10.0 ** (-top_db / 20.0))
    active = np.where(rms >= thresh)[0]
    if active.size == 0:
        return wav

    half_win = frame_len // 2
    margin_samples = int(round(keep_margin_ms * sr / 1000.0))
    start_sample = max(0, int(active[0]) * hop_len + half_win - margin_samples)
    end_sample = min(len(wav), int(active[-1]) * hop_len + half_win + margin_samples)
    if end_sample <= start_sample:
        return wav
    return wav[start_sample:end_sample]


def _peaking_eq_biquad(fs: float, f0: float, q: float, gain_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """Design a digital peaking EQ filter (RBJ Audio EQ Cookbook)."""
    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / a_amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_amp

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def apply_elderly_clarity_eq(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    center_hz: float = 2600.0,
    q: float = 0.85,
    gain_db: float = 2.8,
) -> np.ndarray:
    """Zero-phase 1.5-4.0 kHz consonant clarity boost + soft-knee DRC for elderly listeners."""
    if wav.size < 32 or abs(gain_db) < 1e-3:
        return wav.astype(np.float32)

    b, a = _peaking_eq_biquad(float(sr), center_hz, q, gain_db)
    sos = signal.tf2sos(b, a)
    filtered = signal.sosfiltfilt(sos, wav.astype(np.float64))

    # Gentle soft-knee limiting + RMS restoration
    in_rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64))) + 1e-12))
    out_rms = float(np.sqrt(np.mean(np.square(filtered)) + 1e-12))
    if out_rms > 1e-6:
        filtered *= in_rms / out_rms

    # Soft ceiling at 0.92 to avoid clipping on transients
    peak = float(np.max(np.abs(filtered)))
    if peak > 0.92:
        filtered = 0.92 * np.tanh(filtered / 0.92)
    return filtered.astype(np.float32)


def measure_clarity_band_ratio_db(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    low_hz: float = 1500.0,
    high_hz: float = 4000.0,
) -> float:
    """Measure energy ratio of the 1.5-4.0 kHz consonant clarity band relative to full speech band (dB)."""
    if wav.size < 512:
        return 0.0
    x = wav.astype(np.float64)
    freqs, psd = signal.welch(x, fs=sr, nperseg=min(2048, len(x)))
    total_mask = np.logical_and(freqs >= 100.0, freqs <= 7500.0)
    clarity_mask = np.logical_and(freqs >= low_hz, freqs <= high_hz)
    total_e = float(np.sum(psd[total_mask])) + 1e-18
    clarity_e = float(np.sum(psd[clarity_mask])) + 1e-18
    return round(10.0 * np.log10(clarity_e / total_e), 3)


def measure_inter_segment_pauses_ms(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    silence_thresh_db: float = -42.0,
    min_pause_ms: float = 100.0,
) -> List[float]:
    """Detect internal silence intervals >= min_pause_ms in a concatenated WAV."""
    if wav.size == 0:
        return []
    x = wav.astype(np.float64)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return []
    x = x / peak

    frame_len = int(round(0.010 * sr))
    hop_len = int(round(0.002 * sr))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    thresh = 10.0 ** (silence_thresh_db / 20.0)

    is_silent = rms < thresh
    pauses_ms: List[float] = []
    in_run = False
    run_start = 0

    # Ignore leading and trailing outer silences; measure only interior pauses
    active = np.where(~is_silent)[0]
    if active.size < 2:
        return []
    first_active, last_active = int(active[0]), int(active[-1])

    for i in range(first_active, last_active + 1):
        if is_silent[i] and not in_run:
            in_run = True
            run_start = i
        elif not is_silent[i] and in_run:
            in_run = False
            dur_ms = (i - run_start) * hop_len / sr * 1000.0
            if dur_ms >= min_pause_ms:
                pauses_ms.append(round(dur_ms, 1))
    return pauses_ms


def simulate_elderly_clinic_channel(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    snr_db: float = 14.0,
    lpf_hz: float = 3400.0,
    seed: int = 1234,
) -> np.ndarray:
    """Simulate mild presbycusis high-frequency rolloff + clinic waiting-room ambient noise."""
    x = wav.astype(np.float64)
    # 1. Presbycusis mild high-frequency rolloff (2nd order Butterworth LPF)
    sos_lpf = signal.butter(2, lpf_hz, btype="lowpass", fs=sr, output="sos")
    x_lpf = signal.sosfiltfilt(sos_lpf, x)

    # 2. Pink-like ambient clinic noise scaled to target SNR
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(x_lpf))
    sos_pink = signal.butter(1, 700.0, btype="lowpass", fs=sr, output="sos")
    ambient = signal.sosfiltfilt(sos_pink, white)

    sig_rms = float(np.sqrt(np.mean(np.square(x_lpf[np.abs(x_lpf) > 0.01 * np.max(np.abs(x_lpf))])) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(np.square(ambient)) + 1e-12))
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noisy = x_lpf + ambient * (target_noise_rms / noise_rms)
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.95:
        noisy = noisy * (0.95 / peak)
    return noisy.astype(np.float32)
