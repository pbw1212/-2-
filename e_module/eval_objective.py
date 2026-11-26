"""Objective acoustic, alignment, Clear Speech, and TTFA evaluations (Exp 2 - Exp 4)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import torch
from scipy.fft import dct
from scipy.io import wavfile

from e_module.audio_player import MockAudioPlayer
from e_module.clear_speech import (
    measure_clarity_band_ratio_db,
    measure_inter_segment_pauses_ms,
    simulate_elderly_clinic_channel,
    trim_segment_silence,
)
from e_module.config import (
    BASELINE_TACOTRON2_DIR,
    CANONICAL_SAMPLES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FAST_GL_ITERS,
    DEFAULT_GL_ITERS,
    DEFAULT_PAUSE_MS,
    DEFAULT_SEED,
    EXTENDED_ABLATION_SAMPLES,
    OUTPUTS_DIR,
    SAMPLE_RATE,
)
from e_module.tacotron_compat import fast_griffin_lim, griffin_lim_compat, prepare_input_sequence
from e_module.tts_engine import Tacotron2Engine, _save_normalized_int16_wav
from common.utils import dc_notch_filter  # noqa: E402

BIAOBEI_WAVS_DIR = BASELINE_TACOTRON2_DIR / "data_mandarin" / "biaobei" / "wavs"
BIAOBEI_METADATA = BASELINE_TACOTRON2_DIR / "data_mandarin" / "biaobei" / "metadata.csv"


def _load_wav_float(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32)


def _prepare_for_mcd(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Remove DC drift, trim studio/TTS outer silence, and normalize peak to 0.8."""
    w = dc_notch_filter(wav.astype(np.float32))
    w = trim_segment_silence(w, sr=sr, top_db=30.0, keep_margin_ms=10.0)
    pk = float(np.max(np.abs(w))) if w.size > 0 else 0.0
    if pk > 1e-6:
        w = w * (0.8 / pk)
    return w.astype(np.float32)


def _extract_mcep(wav: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 25) -> np.ndarray:
    """Extract log-mel cepstral coefficients c1..c24 (excluding c0) for textbook MCD."""
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=50.0,
        fmax=7600.0,
        power=1.0,
    )
    log_mel = np.log(np.maximum(mel_spec, 1e-5))
    cep = dct(log_mel, type=2, axis=0, norm="ortho")
    return cep[1:n_mfcc, :]  # (24, T)


def compute_dtw_mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Compute DTW-aligned Mel-Cepstral Distortion (dB); lower is closer to reference."""
    ref_mcep = _extract_mcep(_prepare_for_mcd(ref_wav, sr=sr), sr=sr)
    syn_mcep = _extract_mcep(_prepare_for_mcd(syn_wav, sr=sr), sr=sr)
    _, wp = librosa.sequence.dtw(X=ref_mcep, Y=syn_mcep, metric="euclidean")
    ref_aligned = ref_mcep[:, wp[:, 0]]
    syn_aligned = syn_mcep[:, wp[:, 1]]
    diff = ref_aligned - syn_aligned
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * float(np.mean(frame_dist))
    return round(mcd, 4)


def _infer_and_vocode_timed(
    engine: Tacotron2Engine,
    pinyin: str,
    vocoder_mode: str,
    gl_iters: int,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, float, bool]:
    """Apples-to-apples timing of model.infer + inv_mel_spectrogram + vocoder."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    seqs, lens, _ = prepare_input_sequence([pinyin], speaker_id=0)
    t0 = time.perf_counter()
    with torch.no_grad():
        _, mels, _, _, mel_lengths = [o.cpu() for o in engine.model.infer(seqs, lens)]
    mel_len = int(mel_lengths[0].item())
    mel = mels[0:1, :, :mel_len]
    hit_max = mel_len >= engine.model.decoder.max_decoder_steps * engine.model.n_frames_per_step
    mag = engine.stft.inv_mel_spectrogram(mel)
    if vocoder_mode == "fast_griffin_lim":
        wav = fast_griffin_lim(mag, engine.stft.stft_fn, n_iters=gl_iters, power=1.5)
    else:
        wav = griffin_lim_compat(mag, engine.stft.stft_fn, n_iters=gl_iters, power=1.5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return wav.astype(np.float32), elapsed_ms, hit_max


def run_exp2a_biaobei_mcd(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    limit: int = 99,
) -> Dict[str, Any]:
    """Evaluate 5 vocoder/quantization configs against up to 99 studio Biaobei recordings."""
    pairs: List[Tuple[str, str, Path]] = []
    with BIAOBEI_METADATA.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            uid = parts[0].strip()
            wav_p = BIAOBEI_WAVS_DIR / f"{uid}.wav"
            if wav_p.exists():
                pairs.append((uid, parts[1].strip(), wav_p))
            if len(pairs) >= limit:
                break

    eng_fp32_fast = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng_fp32_fast.warmup()
    eng_int8_naive = Tacotron2Engine(checkpoint_path, quantize_mode="naive_linear_lstm", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng_int8_naive.warmup()
    eng_int8_lstm = Tacotron2Engine(checkpoint_path, quantize_mode="lstm_converted", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng_int8_lstm.warmup()

    mcds: Dict[str, List[float]] = {
        "fp32_gl50": [],
        "fp32_gl25": [],
        "fp32_fastgl25": [],
        "int8_naive_fastgl25": [],
        "int8_lstm_converted_fastgl25": [],
    }
    times_ms: Dict[str, List[float]] = {k: [] for k in mcds}
    paired_int8_vs_fp32_mcd: List[float] = []
    paired_fp32_seed2_vs_seed1_mcd: List[float] = []
    max_steps_counts: Dict[str, int] = {k: 0 for k in mcds}

    for uid, pinyin, ref_path in pairs:
        ref_wav = _load_wav_float(ref_path)

        # 1. FP32 acoustic pass + inv_mel (timed equally for GL50 / GL25 / Fast-GL25)
        torch.manual_seed(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)
        seqs, lens, _ = prepare_input_sequence([pinyin], speaker_id=0)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, mels, _, _, mel_lengths = [o.cpu() for o in eng_fp32_fast.model.infer(seqs, lens)]
        mel_len = int(mel_lengths[0].item())
        mel = mels[0:1, :, :mel_len]
        hit_max_fp32 = mel_len >= eng_fp32_fast.model.decoder.max_decoder_steps * eng_fp32_fast.model.n_frames_per_step
        mag = eng_fp32_fast.stft.inv_mel_spectrogram(mel)
        t_fp32_ac_inv = (time.perf_counter() - t0) * 1000.0

        # GL-50
        t_v0 = time.perf_counter()
        w_gl50 = griffin_lim_compat(mag, eng_fp32_fast.stft.stft_fn, n_iters=50, power=1.5)
        times_ms["fp32_gl50"].append(t_fp32_ac_inv + (time.perf_counter() - t_v0) * 1000.0)
        mcds["fp32_gl50"].append(compute_dtw_mcd(ref_wav, w_gl50))
        if hit_max_fp32:
            max_steps_counts["fp32_gl50"] += 1

        # GL-25
        t_v1 = time.perf_counter()
        w_gl25 = griffin_lim_compat(mag, eng_fp32_fast.stft.stft_fn, n_iters=25, power=1.5)
        times_ms["fp32_gl25"].append(t_fp32_ac_inv + (time.perf_counter() - t_v1) * 1000.0)
        mcds["fp32_gl25"].append(compute_dtw_mcd(ref_wav, w_gl25))
        if hit_max_fp32:
            max_steps_counts["fp32_gl25"] += 1

        # Fast-GL-25
        t_v2 = time.perf_counter()
        w_fgl25 = fast_griffin_lim(mag, eng_fp32_fast.stft.stft_fn, n_iters=25, power=1.5)
        times_ms["fp32_fastgl25"].append(t_fp32_ac_inv + (time.perf_counter() - t_v2) * 1000.0)
        mcds["fp32_fastgl25"].append(compute_dtw_mcd(ref_wav, w_fgl25))
        if hit_max_fp32:
            max_steps_counts["fp32_fastgl25"] += 1

        # Stochastic self-consistency baseline: same FP32 model with different Prenet seed
        w_fgl25_s2, _, _ = _infer_and_vocode_timed(eng_fp32_fast, pinyin, "fast_griffin_lim", 25, seed=DEFAULT_SEED + 1)
        paired_fp32_seed2_vs_seed1_mcd.append(compute_dtw_mcd(w_fgl25, w_fgl25_s2))

        # 2. Naive INT8 + Fast-GL25 (apples-to-apples timing)
        w_naive, ms_naive, hm_naive = _infer_and_vocode_timed(eng_int8_naive, pinyin, "fast_griffin_lim", 25, seed=DEFAULT_SEED)
        times_ms["int8_naive_fastgl25"].append(ms_naive)
        mcds["int8_naive_fastgl25"].append(compute_dtw_mcd(ref_wav, w_naive))
        if hm_naive:
            max_steps_counts["int8_naive_fastgl25"] += 1

        # 3. LSTM-Converted INT8 + Fast-GL25 (apples-to-apples timing)
        w_lstm, ms_lstm, hm_lstm = _infer_and_vocode_timed(eng_int8_lstm, pinyin, "fast_griffin_lim", 25, seed=DEFAULT_SEED)
        times_ms["int8_lstm_converted_fastgl25"].append(ms_lstm)
        mcds["int8_lstm_converted_fastgl25"].append(compute_dtw_mcd(ref_wav, w_lstm))
        if hm_lstm:
            max_steps_counts["int8_lstm_converted_fastgl25"] += 1

        paired_int8_vs_fp32_mcd.append(compute_dtw_mcd(w_fgl25, w_lstm))

    summary = {}
    sizes = {
        "fp32_gl50": eng_fp32_fast.model_size_mb,
        "fp32_gl25": eng_fp32_fast.model_size_mb,
        "fp32_fastgl25": eng_fp32_fast.model_size_mb,
        "int8_naive_fastgl25": eng_int8_naive.model_size_mb,
        "int8_lstm_converted_fastgl25": eng_int8_lstm.model_size_mb,
    }
    for cfg in mcds:
        summary[cfg] = {
            "utterance_count": len(pairs),
            "model_size_mb": sizes[cfg],
            "mcd_mean_db": round(float(np.mean(mcds[cfg])), 4),
            "mcd_std_db": round(float(np.std(mcds[cfg])), 4),
            "synth_ms_median": round(float(np.median(times_ms[cfg])), 2),
            "synth_ms_p90": round(float(np.percentile(times_ms[cfg], 90)), 2),
            "max_steps_count": max_steps_counts[cfg],
        }
    return {
        "utterance_count": len(pairs),
        "configs": summary,
        "int8_lstm_vs_fp32_fastgl25_paired_mcd_mean_db": round(float(np.mean(paired_int8_vs_fp32_mcd)), 4),
        "fp32_seed2_vs_seed1_paired_mcd_mean_db": round(float(np.mean(paired_fp32_seed2_vs_seed1_mcd)), 4),
    }


def run_exp2b_attention_and_frontend(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Dict[str, Any]:
    """Compare basic_pinyin, tn_only, tn_prosody, and tn_psp_ec on clinic attention stability."""
    eng = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng.warmup()

    modes = ("basic_pinyin", "tn_only", "tn_prosody", "tn_psp_ec")
    align_dir = OUTPUTS_DIR / "alignments"
    out_dir = OUTPUTS_DIR / "exp2b_wavs"
    records: List[Dict[str, Any]] = []

    for sample in EXTENDED_ABLATION_SAMPLES:
        for mode in modes:
            wav_p = out_dir / mode / f"{sample.sample_id}.wav"
            save_align = align_dir / f"{sample.sample_id}_{mode}.json" if sample.sample_id in ("short_prompt", "multi_field_confirm") else None
            res = eng.synthesize(
                text=sample.text,
                output_wav=wav_p,
                mode=mode,
                action=sample.action,
                pause_ms=DEFAULT_PAUSE_MS,
                use_cache=False,
                save_alignment_path=save_align,
            )
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "mode": mode,
                    "audio_duration_s": res.audio_duration_s,
                    "total_ms": res.total_ms,
                    "rtf": res.rtf,
                    "hit_max_steps": res.hit_max_steps,
                    "attention_focus": res.attention_focus,
                    "attention_monotonicity": res.attention_monotonicity,
                    "segment_count": len(res.trace["segments"]),
                }
            )

    by_mode: Dict[str, Any] = {}
    for mode in modes:
        sub = [r for r in records if r["mode"] == mode]
        by_mode[mode] = {
            "max_steps_rate": round(sum(1 for r in sub if r["hit_max_steps"]) / len(sub), 4),
            "attention_focus_mean": round(float(np.mean([r["attention_focus"] for r in sub])), 4),
            "attention_monotonicity_mean": round(float(np.mean([r["attention_monotonicity"] for r in sub])), 4),
            "duration_s_mean": round(float(np.mean([r["audio_duration_s"] for r in sub])), 2),
        }
    return {"by_mode": by_mode, "records": records}


def run_exp3_clear_speech(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Dict[str, Any]:
    """Evaluate Clear Speech acoustic modulation, pause precision, and simulated elderly channel demos."""
    eng = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng.warmup()

    demo_dir = OUTPUTS_DIR / "clear_speech_demo"
    clean_dir = demo_dir / "clean"
    sim_dir = demo_dir / "elderly_channel_sim"
    clean_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    treatments = [
        ("standard", False, False, 1.0, 0.0),
        ("trim_pause", False, True, 1.0, 0.0),
        ("clear_speech_full", True, True, 1.18, 2.8),
    ]

    records: List[Dict[str, Any]] = []
    for sample in EXTENDED_ABLATION_SAMPLES:
        for t_name, cs_flag, trim_flag, stretch, eq_gain in treatments:
            wav_p = clean_dir / t_name / f"{sample.sample_id}.wav"
            res = eng.synthesize(
                text=sample.text,
                output_wav=wav_p,
                mode="tn_psp_ec",
                action=sample.action,
                pause_ms=280,
                use_cache=False,
                clear_speech=cs_flag,
                trim_silence=trim_flag,
                entity_stretch=stretch,
                clarity_eq_gain_db=eq_gain,
            )
            # Generate elderly presbycusis + clinic ambient noise channel version for subjective comparison
            sr, pcm = wavfile.read(str(wav_p))
            y_float = pcm.astype(np.float32) / 32768.0
            sim_wav = simulate_elderly_clinic_channel(y_float, sr=sr, snr_db=14.0, lpf_hz=3400.0)
            _save_normalized_int16_wav(sim_wav, sim_dir / t_name / f"{sample.sample_id}.wav", sr=sr)

            records.append(
                {
                    "sample_id": sample.sample_id,
                    "treatment": t_name,
                    "audio_duration_s": res.audio_duration_s,
                    "entity_stretch_ratio": res.entity_stretch_ratio,
                    "clarity_band_db": res.clarity_band_db,
                    "hit_max_steps": res.hit_max_steps,
                    "wav_path": str(wav_p),
                }
            )

    # Calibrated pause precision test on multi_field_confirm (200 / 280 / 350 ms)
    confirm_sample = next(s for s in EXTENDED_ABLATION_SAMPLES if s.sample_id == "multi_field_confirm")
    pause_calibration: List[Dict[str, Any]] = []
    for target_ms in (200, 280, 350):
        wav_p = demo_dir / "pause_calibrated" / f"confirm_p{target_ms}.wav"
        _ = eng.synthesize(
            text=confirm_sample.text,
            output_wav=wav_p,
            mode="tn_psp_ec",
            action=confirm_sample.action,
            pause_ms=target_ms,
            use_cache=False,
            clear_speech=True,
            trim_silence=True,
            entity_stretch=1.18,
            clarity_eq_gain_db=2.8,
        )
        sr, pcm = wavfile.read(str(wav_p))
        detected = measure_inter_segment_pauses_ms(pcm.astype(np.float32) / 32768.0, sr=sr, min_pause_ms=120.0)
        errors = [round(abs(d - target_ms), 1) for d in detected]
        pause_calibration.append(
            {
                "target_pause_ms": target_ms,
                "detected_pauses_ms": detected,
                "abs_errors_ms": errors,
                "max_abs_error_ms": max(errors) if errors else None,
                "wav_path": str(wav_p),
            }
        )

    std_band = float(np.mean([r["clarity_band_db"] for r in records if r["treatment"] == "standard"]))
    cs_band = float(np.mean([r["clarity_band_db"] for r in records if r["treatment"] == "clear_speech_full"]))
    stretch_vals = [r["entity_stretch_ratio"] for r in records if r["treatment"] == "clear_speech_full" and r["entity_stretch_ratio"] > 1.0]

    return {
        "summary": {
            "standard_clarity_band_db": round(std_band, 3),
            "clear_speech_clarity_band_db": round(cs_band, 3),
            "clarity_band_gain_db": round(cs_band - std_band, 3),
            "entity_stretch_ratio_mean": round(float(np.mean(stretch_vals)), 4) if stretch_vals else 1.0,
            "pause_calibration_max_error_ms": max(
                p["max_abs_error_ms"] for p in pause_calibration if p["max_abs_error_ms"] is not None
            ),
        },
        "pause_calibration": pause_calibration,
        "records": records,
    }


def run_exp4_pipelined_ttfa(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> Dict[str, Any]:
    """Compare whole-utterance vs chunk-pipelined synthesis TTFA across canonical utterances."""
    eng_fp32 = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng_fp32.warmup()
    eng_int8 = Tacotron2Engine(checkpoint_path, quantize_mode="lstm_converted", vocoder_mode="fast_griffin_lim", gl_iters=25)
    eng_int8.warmup()

    pipe_dir = OUTPUTS_DIR / "exp4_pipelined"
    rows: List[Dict[str, Any]] = []

    for sample in CANONICAL_SAMPLES:
        # Whole-utterance FP32 Fast-GL25
        res_whole_fp32 = eng_fp32.synthesize(
            text=sample.text,
            output_wav=pipe_dir / "whole_fp32" / f"{sample.sample_id}.wav",
            mode="tn_psp_ec",
            action=sample.action,
            use_cache=False,
            clear_speech=True,
            trim_silence=True,
        )
        # Whole-utterance INT8 LSTM-Converted Fast-GL25
        res_whole_int8 = eng_int8.synthesize(
            text=sample.text,
            output_wav=pipe_dir / "whole_int8" / f"{sample.sample_id}.wav",
            mode="tn_psp_ec",
            action=sample.action,
            use_cache=False,
            clear_speech=True,
            trim_silence=True,
        )
        # Chunk-pipelined INT8 playback (1.0x real-time playback simulation to accurately verify zero underruns)
        player_rt = MockAudioPlayer(speed_factor=1.0)
        res_pipe_int8 = eng_int8.synthesize_and_play_pipelined(
            text=sample.text,
            output_dir=pipe_dir / "pipe_int8",
            stem=sample.sample_id,
            player=player_rt,
            mode="tn_psp_ec",
            action=sample.action,
            use_cache=False,
            clear_speech=True,
            trim_silence=True,
        )

        ttfa_drop_pct = round((res_whole_int8.total_ms - res_pipe_int8.ttfa_ms) / max(res_whole_int8.total_ms, 1e-6) * 100.0, 2)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "segment_count": res_pipe_int8.segment_count,
                "whole_fp32_ttfa_ms": res_whole_fp32.total_ms,
                "whole_int8_ttfa_ms": res_whole_int8.total_ms,
                "pipelined_int8_ttfa_ms": res_pipe_int8.ttfa_ms,
                "ttfa_reduction_pct_vs_whole_int8": ttfa_drop_pct,
                "pipelined_underrun_count": res_pipe_int8.underrun_count,
                "audio_duration_s": res_pipe_int8.audio_duration_s,
            }
        )

    return {"samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run objective Exp 2 - Exp 4 evaluations.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--mcd-limit", type=int, default=99)
    parser.add_argument("--skip-mcd", action="store_true")
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    if not args.skip_mcd:
        print(f"Running Exp 2A: DTW-MCD over {args.mcd_limit} Biaobei studio recordings...")
        report["exp2a_biaobei_mcd"] = run_exp2a_biaobei_mcd(args.checkpoint, limit=args.mcd_limit)
    print("Running Exp 2B: Clinic frontend & attention stability...")
    report["exp2b_attention_frontend"] = run_exp2b_attention_and_frontend(args.checkpoint)
    print("Running Exp 3: Clear Speech acoustic modulation & pause calibration...")
    report["exp3_clear_speech"] = run_exp3_clear_speech(args.checkpoint)
    print("Running Exp 4: Whole vs chunk-pipelined TTFA...")
    report["exp4_pipelined_ttfa"] = run_exp4_pipelined_ttfa(args.checkpoint)

    out_path = OUTPUTS_DIR / "exp2_4_objective_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved objective report to {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "exp2b_attention_frontend"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
