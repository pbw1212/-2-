"""Run E-module frontend ablation experiments and CPU speed/quantization benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from e_module.baseline_runner import run_canonical_baseline, validate_wav_file
from e_module.config import (
    ABLATION_OUTPUTS_DIR,
    BASELINE_OUTPUTS_DIR,
    CANONICAL_SAMPLES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FAST_GL_ITERS,
    DEFAULT_GL_ITERS,
    DEFAULT_PAUSE_MS,
    EXTENDED_ABLATION_SAMPLES,
    OUTPUTS_DIR,
)
from e_module.tts_engine import Tacotron2Engine


def _stats(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "median": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def run_ablation_experiment(
    engine: Tacotron2Engine,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = ABLATION_OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Generate ablation WAVs across clinic samples + pause length sweep.

    Reuses the 3 canonical baseline_raw WAVs from outputs/baseline/ so we preserve
    exact buggy-baseline evidence without wasting 5+ minutes re-running max_steps noise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    baseline_metrics_path = BASELINE_OUTPUTS_DIR / "baseline_metrics.json"
    if not baseline_metrics_path.exists():
        run_canonical_baseline(checkpoint_path=checkpoint_path, output_dir=BASELINE_OUTPUTS_DIR)
    baseline_data = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))

    baseline_raw_dir = output_dir / "baseline_raw"
    baseline_raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in baseline_data["records"]:
        src_wav = Path(rec["wav_path"])
        dst_wav = baseline_raw_dir / f"{rec['sample_id']}.wav"
        if src_wav.exists():
            shutil.copyfile(src_wav, dst_wav)
        records.append(
            {
                "sample_id": rec["sample_id"],
                "category": rec["category"],
                "action": "baseline",
                "mode": "baseline_raw",
                "raw_text": rec["raw_text"],
                "normalized_text": rec["raw_text"],
                "segment_count": len(rec["split_sentences"]),
                "segments": [{"text": s, "pinyin": s, "pause_after_ms": 0} for s in rec["split_sentences"]],
                "wav_path": str(dst_wav),
                "audio_duration_s": rec["audio_duration_s"],
                "total_ms": rec["total_synth_ms"],
                "tacotron2_ms": rec["tacotron2_ms"],
                "vocoder_ms": rec["vocoder_ms"],
                "rtf": rec["rtf"],
                "hit_max_steps": rec["hit_max_steps"],
                "wav_valid": rec["wav_valid"],
            }
        )

    modes = ("basic_pinyin", "tn_only", "tn_prosody")
    for sample in EXTENDED_ABLATION_SAMPLES:
        for mode in modes:
            wav_path = output_dir / mode / f"{sample.sample_id}.wav"
            res = engine.synthesize(
                text=sample.text,
                output_wav=wav_path,
                mode=mode,
                action=sample.action,
                pause_ms=DEFAULT_PAUSE_MS,
                cacheable=False,
                use_cache=False,
            )
            validity = validate_wav_file(wav_path)
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "action": sample.action,
                    "mode": mode,
                    "raw_text": sample.text,
                    "normalized_text": res.trace["normalized_text"],
                    "segment_count": len(res.trace["segments"]),
                    "segments": res.trace["segments"],
                    "wav_path": str(wav_path),
                    "audio_duration_s": res.audio_duration_s,
                    "total_ms": res.total_ms,
                    "tacotron2_ms": res.tacotron2_ms,
                    "vocoder_ms": res.vocoder_ms,
                    "rtf": res.rtf,
                    "hit_max_steps": res.hit_max_steps,
                    "wav_valid": validity["wav_valid"],
                }
            )

    confirm_sample = next(s for s in EXTENDED_ABLATION_SAMPLES if s.sample_id == "multi_field_confirm")
    pause_records: List[Dict[str, Any]] = []
    for pause_ms in (200, 260, 350):
        wav_path = output_dir / "pause_sweep" / f"{confirm_sample.sample_id}_p{pause_ms}.wav"
        res = engine.synthesize(
            text=confirm_sample.text,
            output_wav=wav_path,
            mode="tn_prosody",
            action=confirm_sample.action,
            pause_ms=pause_ms,
            cacheable=False,
            use_cache=False,
        )
        pause_records.append(
            {
                "sample_id": confirm_sample.sample_id,
                "pause_ms": pause_ms,
                "wav_path": str(wav_path),
                "audio_duration_s": res.audio_duration_s,
                "total_ms": res.total_ms,
                "hit_max_steps": res.hit_max_steps,
            }
        )

    manifest = {
        "pause_ms_default": DEFAULT_PAUSE_MS,
        "vocoder_mode": engine.vocoder_mode,
        "gl_iters": engine.gl_iters,
        "records": records,
        "pause_sweep": pause_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_csv = output_dir / "listening_eval_template.csv"
    rng = random.Random(42)
    with eval_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "blind_item_id",
                "sample_id",
                "hidden_version",
                "wav_path",
                "listener_id",
                "recalled_fields",
                "required_fields_count",
                "correct_fields_count",
                "naturalness_1_to_5",
                "notes",
            ]
        )
        items = [r for r in records if r["mode"] in ("baseline_raw", "basic_pinyin", "tn_only", "tn_prosody")]
        rng.shuffle(items)
        for idx, rec in enumerate(items, start=1):
            writer.writerow(
                [
                    f"L{idx:02d}",
                    rec["sample_id"],
                    rec["mode"],
                    rec["wav_path"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    return manifest


def run_speed_compression_benchmark(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = OUTPUTS_DIR / "benchmark_wavs",
) -> Dict[str, Any]:
    """Benchmark 6 engine configurations across canonical prompts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    def _run_config(
        cfg_name: str,
        engine: Tacotron2Engine,
        warmup_first: bool,
        use_cache: bool,
        cacheable_override: bool,
        include_load_in_first: bool = False,
    ) -> Dict[str, Any]:
        warmup_ms = engine.warmup() if warmup_first else 0.0
        per_sample: List[Dict[str, Any]] = []
        total_ms_list: List[float] = []
        t2_ms_list: List[float] = []
        voc_ms_list: List[float] = []
        rtfs: List[float] = []

        for idx, sample in enumerate(CANONICAL_SAMPLES):
            wav_path = output_dir / cfg_name / f"{sample.sample_id}.wav"
            res = engine.synthesize(
                text=sample.text,
                output_wav=wav_path,
                mode="tn_prosody",
                action=sample.action,
                pause_ms=DEFAULT_PAUSE_MS,
                cacheable=cacheable_override,
                use_cache=use_cache,
            )
            validity = validate_wav_file(wav_path)
            effective_ms = res.total_ms + (engine.cold_load_ms if (include_load_in_first and idx == 0) else 0.0)
            eff_rtf = round((effective_ms / 1000.0) / max(res.audio_duration_s, 1e-6), 4)
            total_ms_list.append(effective_ms)
            t2_ms_list.append(res.tacotron2_ms)
            voc_ms_list.append(res.vocoder_ms)
            rtfs.append(eff_rtf)
            per_sample.append(
                {
                    "sample_id": sample.sample_id,
                    "audio_duration_s": res.audio_duration_s,
                    "total_ms": round(effective_ms, 2),
                    "tacotron2_ms": res.tacotron2_ms,
                    "vocoder_ms": res.vocoder_ms,
                    "rtf": eff_rtf,
                    "cache_hit": res.cache_hit,
                    "hit_max_steps": res.hit_max_steps,
                    "wav_valid": validity["wav_valid"],
                }
            )

        return {
            "config_name": cfg_name,
            "cold_load_ms": engine.cold_load_ms,
            "warmup_ms": warmup_ms,
            "model_size_mb": engine.model_size_mb,
            "quantized": engine.quantize_int8,
            "quantize_mode": engine.quantize_mode,
            "quant_engine": engine.quant_engine,
            "vocoder_mode": engine.vocoder_mode,
            "gl_iters": engine.gl_iters,
            "total_ms_stats": _stats(total_ms_list),
            "tacotron2_ms_stats": _stats(t2_ms_list),
            "vocoder_ms_stats": _stats(voc_ms_list),
            "rtf_median": round(float(np.median(rtfs)), 4),
            "rtf_p90": round(float(np.percentile(rtfs, 90)), 4),
            "any_hit_max_steps": any(s["hit_max_steps"] for s in per_sample),
            "all_wav_valid": all(s["wav_valid"] for s in per_sample),
            "samples": per_sample,
        }

    # 1. Cold start FP32 + GL50
    eng_cold = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="griffin_lim", gl_iters=DEFAULT_GL_ITERS)
    results["cold_start_fp32_gl50"] = _run_config(
        "cold_start_fp32_gl50", eng_cold, warmup_first=False, use_cache=False, cacheable_override=False, include_load_in_first=True
    )

    # 2. Warm FP32 + GL50
    results["warm_fp32_gl50"] = _run_config(
        "warm_fp32_gl50", eng_cold, warmup_first=True, use_cache=False, cacheable_override=False
    )

    # 3. Warm FP32 + Fast-GL25
    eng_fast = Tacotron2Engine(checkpoint_path, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=DEFAULT_FAST_GL_ITERS)
    results["warm_fp32_fastgl25"] = _run_config(
        "warm_fp32_fastgl25", eng_fast, warmup_first=True, use_cache=False, cacheable_override=False
    )

    # 4. Warm INT8 Naive (Linear+LSTM only, LSTMCell unquantized -> negative ablation control)
    eng_int8_naive = Tacotron2Engine(
        checkpoint_path, quantize_mode="naive_linear_lstm", vocoder_mode="fast_griffin_lim", gl_iters=DEFAULT_FAST_GL_ITERS
    )
    results["warm_int8_naive_fastgl25"] = _run_config(
        "warm_int8_naive_fastgl25", eng_int8_naive, warmup_first=True, use_cache=False, cacheable_override=False
    )

    # 5. Warm INT8 LSTM-Converted (LSTMCell -> LSTM quantized, Linear stays FP32)
    eng_int8_lstm = Tacotron2Engine(
        checkpoint_path, quantize_mode="lstm_converted", vocoder_mode="fast_griffin_lim", gl_iters=DEFAULT_FAST_GL_ITERS
    )
    results["warm_int8_lstm_converted_fastgl25"] = _run_config(
        "warm_int8_lstm_converted_fastgl25", eng_int8_lstm, warmup_first=True, use_cache=False, cacheable_override=False
    )

    # 6. Fixed prompt cache hit
    for sample in CANONICAL_SAMPLES:
        _ = eng_fast.synthesize(
            text=sample.text,
            output_wav=output_dir / "cache_warmup" / f"{sample.sample_id}.wav",
            mode="tn_prosody",
            action=sample.action,
            pause_ms=DEFAULT_PAUSE_MS,
            cacheable=True,
            use_cache=True,
        )
    results["fixed_prompt_cache_hit"] = _run_config(
        "fixed_prompt_cache_hit", eng_fast, warmup_first=False, use_cache=True, cacheable_override=True
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E-module ablation and speed/compression benchmarks.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()

    summary: Dict[str, Any] = {}
    if not args.skip_ablation:
        eng = Tacotron2Engine(args.checkpoint, quantize_mode="none", vocoder_mode="fast_griffin_lim", gl_iters=DEFAULT_FAST_GL_ITERS)
        eng.warmup()
        summary["ablation"] = run_ablation_experiment(eng, args.checkpoint, ABLATION_OUTPUTS_DIR)

    if not args.skip_speed:
        summary["speed_compression"] = run_speed_compression_benchmark(args.checkpoint)

    report_path = OUTPUTS_DIR / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if "speed_compression" in summary:
        print("=== SPEED & COMPRESSION BENCHMARK SUMMARY ===")
        for cfg_name, info in summary["speed_compression"].items():
            print(
                f"{cfg_name:34s} | size={info['model_size_mb']:6.2f} MB | "
                f"median={info['total_ms_stats']['median']:7.2f} ms | "
                f"p90={info['total_ms_stats']['p90']:7.2f} ms | "
                f"RTF_med={info['rtf_median']:.4f} | "
                f"max_steps={info['any_hit_max_steps']} | valid={info['all_wav_valid']}"
            )
    if "ablation" in summary:
        print(f"Ablation manifest written to {ABLATION_OUTPUTS_DIR / 'manifest.json'}")
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    main()
