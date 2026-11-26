"""Course baseline Tacotron2 runner (preserves original inference.py frontend logic)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from pypinyin import Style, lazy_pinyin
from scipy.io import wavfile

from e_module.config import (
    BASELINE_OUTPUTS_DIR,
    CANONICAL_SAMPLES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_GL_ITERS,
    DEFAULT_SEED,
    SAMPLE_RATE,
)
from e_module.tacotron_compat import (
    CompatTacotronSTFT,
    griffin_lim_compat,
    load_tacotron2_model,
    prepare_input_sequence,
)
from common.utils import save_wav  # noqa: E402


@dataclass
class BaselineRunRecord:
    sample_id: str
    category: str
    raw_text: str
    split_sentences: List[str]
    wav_path: str
    sample_rate: int
    num_samples: int
    audio_duration_s: float
    peak_abs: int
    wav_valid: bool
    tacotron2_ms: float
    vocoder_ms: float
    total_synth_ms: float
    rtf: float
    hit_max_steps: bool


def validate_wav_file(wav_path: Path, expected_sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Read back a generated WAV file and verify basic audio integrity."""
    sr, data = wavfile.read(str(wav_path))
    arr = np.asarray(data)
    non_empty = arr.size > 0
    finite = bool(np.isfinite(arr).all())
    peak = int(np.max(np.abs(arr))) if non_empty else 0
    duration_s = float(arr.shape[0] / sr) if sr > 0 else 0.0
    valid = (sr == expected_sr) and non_empty and finite and (peak > 0) and (duration_s > 0.1)
    return {
        "sample_rate": int(sr),
        "num_samples": int(arr.shape[0]),
        "audio_duration_s": round(duration_s, 4),
        "peak_abs": peak,
        "wav_valid": valid,
    }


def synthesize_baseline_lines(
    model: torch.nn.Module,
    stft: CompatTacotronSTFT,
    lines: List[str],
    output_wav: Path,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Replicate course inference.py synthesis on raw lines."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Exact baseline line from inference.py:
    # sentences = list(map(lambda s: s.strip(), lazy_pinyin(f.readlines(), Style.TONE3)))
    raw_tokens = lazy_pinyin(lines, Style.TONE3)
    sentences = [s.strip() for s in raw_tokens if s.strip()]
    if not sentences:
        sentences = ["."]

    sequences, text_lengths, ids_sorted_decreasing = prepare_input_sequence(sentences, speaker_id=0)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.infer(sequences, text_lengths)
        _, mels, _, _, mel_lengths = [output.cpu() for output in outputs]
    t1 = time.perf_counter()

    ids_list = ids_sorted_decreasing.numpy().tolist()
    mel_list = [mel[:, :length] for mel, length in zip(mels, mel_lengths)]
    mel_list = [mel_list[ids_list.index(i)] for i in range(len(ids_list))]
    hit_max_steps = any(int(length) >= model.decoder.max_decoder_steps * model.n_frames_per_step for length in mel_lengths.tolist())

    t2 = time.perf_counter()
    magnitudes = stft.inv_mel_spectrogram(torch.cat(mel_list, dim=-1))
    wav = griffin_lim_compat(magnitudes, stft.stft_fn, n_iters=DEFAULT_GL_ITERS, power=1.5)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    save_wav(wav, str(output_wav))
    t3 = time.perf_counter()

    info = validate_wav_file(output_wav)
    tacotron2_ms = (t1 - t0) * 1000.0
    vocoder_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0
    rtf = (total_ms / 1000.0) / max(info["audio_duration_s"], 1e-6)
    return {
        "sentences": sentences,
        "tacotron2_ms": round(tacotron2_ms, 2),
        "vocoder_ms": round(vocoder_ms, 2),
        "total_synth_ms": round(total_ms, 2),
        "rtf": round(rtf, 4),
        "hit_max_steps": hit_max_steps,
        **info,
    }


def run_canonical_baseline(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = BASELINE_OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Run the course baseline on the three canonical clinic test utterances."""
    output_dir.mkdir(parents=True, exist_ok=True)

    t_load_0 = time.perf_counter()
    model = load_tacotron2_model(checkpoint_path)
    stft = CompatTacotronSTFT(cache_window_sum=False)
    cold_load_ms = round((time.perf_counter() - t_load_0) * 1000.0, 2)

    records: List[BaselineRunRecord] = []
    for sample in CANONICAL_SAMPLES:
        wav_path = output_dir / f"{sample.sample_id}.wav"
        res = synthesize_baseline_lines(
            model=model,
            stft=stft,
            lines=[sample.text + "\n"],
            output_wav=wav_path,
        )
        records.append(
            BaselineRunRecord(
                sample_id=sample.sample_id,
                category=sample.category,
                raw_text=sample.text,
                split_sentences=res["sentences"],
                wav_path=str(wav_path),
                sample_rate=res["sample_rate"],
                num_samples=res["num_samples"],
                audio_duration_s=res["audio_duration_s"],
                peak_abs=res["peak_abs"],
                wav_valid=res["wav_valid"],
                tacotron2_ms=res["tacotron2_ms"],
                vocoder_ms=res["vocoder_ms"],
                total_synth_ms=res["total_synth_ms"],
                rtf=res["rtf"],
                hit_max_steps=res["hit_max_steps"],
            )
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "cold_load_ms": cold_load_ms,
        "records": [asdict(r) for r in records],
    }
    report_path = output_dir / "baseline_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E-module Tacotron2 course baseline.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_OUTPUTS_DIR)
    parser.add_argument("--text", type=str, default=None, help="Optional single utterance to synthesize")
    args = parser.parse_args()

    if args.text:
        model = load_tacotron2_model(args.checkpoint)
        stft = CompatTacotronSTFT(cache_window_sum=False)
        out = args.output_dir / "custom_baseline.wav"
        res = synthesize_baseline_lines(model, stft, [args.text], out)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        report = run_canonical_baseline(args.checkpoint, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
