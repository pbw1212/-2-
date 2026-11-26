"""Resident Tacotron2 TTS engine with Fast Griffin-Lim, Clear Speech modulation, pipelined TTFA, caching, and INT8 quantization."""

from __future__ import annotations

import hashlib
import json
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile

from e_module.audio_player import AudioPlayer, PlaybackOutcome
from e_module.clear_speech import (
    apply_elderly_clarity_eq,
    compute_attention_metrics,
    locate_entity_mel_frames,
    measure_clarity_band_ratio_db,
    stretch_mel_local,
    trim_segment_silence,
)
from e_module.config import (
    CACHE_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FAST_GL_ITERS,
    DEFAULT_FAST_GL_MOMENTUM,
    DEFAULT_GL_ITERS,
    DEFAULT_PAUSE_MS,
    DEFAULT_SEED,
    DEFAULT_SHORT_PAUSE_MS,
    SAMPLE_RATE,
    TARGET_SEGMENT_RMS,
)
from e_module.tacotron_compat import (
    CompatTacotronSTFT,
    fast_griffin_lim,
    griffin_lim_compat,
    load_tacotron2_model,
    prepare_input_sequence,
)
from e_module.tts_frontend import FrontendResult, process_text
from common.utils import dc_notch_filter  # noqa: E402

FIXED_CACHEABLE_PROMPTS = {
    "您好，我是离线复诊登记助手，请问有什么可以帮您？",
    "请问您想约哪天？",
    "请问您想挂哪个科室？",
    "请问您是上午还是下午来就诊？",
    "请您再说一遍，我没听清。",
    "抱歉，系统遇到了一点小问题，请再试一次。",
    "好的，本次复诊登记已取消，如需重新预约请随时告诉我。",
    "好的，已为您登记完成，请按预约时间前往医院复诊。",
}


@dataclass
class SynthResult:
    """Structured synthesis metrics and audit metadata."""

    wav_path: str
    sample_rate: int
    num_samples: int
    audio_duration_s: float
    frontend_ms: float
    tacotron2_ms: float
    vocoder_ms: float
    postprocess_ms: float
    total_ms: float
    rtf: float
    cache_hit: bool
    hit_max_steps: bool
    quantized: bool
    quantize_mode: str
    vocoder_mode: str
    mode: str
    trace: Dict[str, Any]
    attention_focus: float = 0.0
    attention_monotonicity: float = 0.0
    entity_stretch_ratio: float = 1.0
    clarity_band_db: float = 0.0
    clear_speech: bool = False
    alignment_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelinedPlaybackResult:
    """Metrics from chunk-pipelined synthesis + playback (minimizing TTFA)."""

    success: bool
    ttfa_ms: float
    total_synth_ms: float
    total_wall_ms: float
    audio_duration_s: float
    segment_count: int
    underrun_count: int
    cache_hit: bool
    hit_max_steps: bool
    playback_start_mono_ms: Optional[int]
    playback_end_mono_ms: Optional[int]
    error_code: Optional[str] = None
    backend: str = "unknown"
    segment_wavs: List[str] = field(default_factory=list)


class LSTMCellAsLSTM(torch.nn.Module):
    """Lossless inference wrapper converting nn.LSTMCell into single-layer nn.LSTM for INT8 quantization."""

    def __init__(self, cell: torch.nn.LSTMCell) -> None:
        super().__init__()
        has_bias = cell.bias_ih is not None
        self.lstm = torch.nn.LSTM(
            input_size=cell.input_size,
            hidden_size=cell.hidden_size,
            num_layers=1,
            bias=has_bias,
            batch_first=True,
        )
        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(cell.weight_ih)
            self.lstm.weight_hh_l0.copy_(cell.weight_hh)
            if has_bias:
                self.lstm.bias_ih_l0.copy_(cell.bias_ih)
                self.lstm.bias_hh_l0.copy_(cell.bias_hh)

    def forward(
        self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hx
        _, (h_n, c_n) = self.lstm(
            x.unsqueeze(1),
            (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous()),
        )
        return h_n.squeeze(0), c_n.squeeze(0)


def convert_decoder_lstm_cells(model: torch.nn.Module) -> torch.nn.Module:
    """Replace decoder LSTMCell modules with mathematically equivalent LSTMCellAsLSTM wrappers."""
    model.decoder.lstm0 = LSTMCellAsLSTM(model.decoder.lstm0)
    model.decoder.lstm1 = LSTMCellAsLSTM(model.decoder.lstm1)
    return model


def _apply_dynamic_int8(model: torch.nn.Module, quantize_mode: str) -> Tuple[torch.nn.Module, str]:
    """Apply PyTorch dynamic INT8 quantization compatible with ARM64 (qnnpack)."""
    engines = getattr(torch.backends.quantized, "supported_engines", [])
    if "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
        engine_used = "qnnpack"
    elif engines:
        torch.backends.quantized.engine = engines[0]
        engine_used = engines[0]
    else:
        engine_used = "default"

    quant_api = getattr(torch, "ao", None)
    quant_fn = (
        quant_api.quantization.quantize_dynamic
        if quant_api and hasattr(quant_api, "quantization")
        else torch.quantization.quantize_dynamic
    )
    if quantize_mode == "naive_linear_lstm":
        q_model = quant_fn(model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)
    elif quantize_mode == "lstm_converted":
        convert_decoder_lstm_cells(model)
        q_model = quant_fn(model, {torch.nn.LSTM}, dtype=torch.qint8)
    else:
        raise ValueError(f"Unknown quantize_mode: {quantize_mode}")
    return q_model, engine_used


def _fade_edges(audio: np.ndarray, sr: int = SAMPLE_RATE, fade_ms: float = 10.0) -> np.ndarray:
    """Apply smooth linear fade-in/fade-out at segment edges to avoid concatenation clicks."""
    out = audio.astype(np.float32, copy=True)
    n_fade = min(int(round(sr * fade_ms / 1000.0)), len(out) // 4)
    if n_fade > 1:
        ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        out[:n_fade] *= ramp
        out[-n_fade:] *= ramp[::-1]
    return out


def _normalize_segment(audio: np.ndarray, target_rms: float = TARGET_SEGMENT_RMS) -> np.ndarray:
    """Remove DC offset, fade edges, and equalize segment RMS before concatenation."""
    cleaned = dc_notch_filter(audio.astype(np.float32))
    cleaned = _fade_edges(cleaned)
    rms = float(np.sqrt(np.mean(np.square(cleaned)))) if cleaned.size > 0 else 0.0
    if rms > 1e-4:
        cleaned = cleaned * (target_rms / rms)
    peak = float(np.max(np.abs(cleaned))) if cleaned.size > 0 else 0.0
    if peak > 0.95:
        cleaned = cleaned * (0.95 / peak)
    return cleaned.astype(np.float32)


def _save_normalized_int16_wav(
    wav: np.ndarray, path: Path, sr: int = SAMPLE_RATE, preserve_level: bool = False
) -> Tuple[int, int]:
    """Save 16-bit PCM WAV with DC notch and safe clipping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_clean = dc_notch_filter(wav.astype(np.float32))
    if preserve_level:
        scaled = wav_clean * 32767.0
    else:
        peak = float(np.max(np.abs(wav_clean))) if wav_clean.size > 0 else 0.0
        if peak > 1e-5:
            wav_clean = wav_clean * (0.8 * 32767.0 / peak)
        scaled = wav_clean
    pcm = np.clip(np.round(scaled), -32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
    return sr, int(pcm.shape[0])


def estimate_state_dict_size_mb(model: torch.nn.Module) -> float:
    """Measure serialized model checkpoint size in MB."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / (1024 * 1024), 2)


class Tacotron2Engine:
    """Resident offline Tacotron2 synthesis engine for E-module service & benchmarking."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        quantize_int8: bool = False,
        quantize_mode: Optional[str] = None,
        vocoder_mode: str = "fast_griffin_lim",
        gl_iters: Optional[int] = None,
        cache_dir: Path = CACHE_DIR,
        seed: int = DEFAULT_SEED,
        max_decoder_steps: Optional[int] = None,
    ) -> None:
        if vocoder_mode not in ("griffin_lim", "fast_griffin_lim"):
            raise ValueError(f"Unsupported vocoder_mode: {vocoder_mode}")

        if quantize_mode is None:
            quantize_mode = "lstm_converted" if quantize_int8 else "none"
        if quantize_mode not in ("none", "naive_linear_lstm", "lstm_converted"):
            raise ValueError(f"Unsupported quantize_mode: {quantize_mode}")

        self.checkpoint_path = Path(checkpoint_path)
        self.quantize_mode = quantize_mode
        self.quantize_int8 = quantize_mode != "none"
        self.vocoder_mode = vocoder_mode
        self.gl_iters = gl_iters if gl_iters is not None else (
            DEFAULT_FAST_GL_ITERS if vocoder_mode == "fast_griffin_lim" else DEFAULT_GL_ITERS
        )
        self.cache_dir = Path(cache_dir)
        self.seed = seed

        t0 = time.perf_counter()
        model = load_tacotron2_model(self.checkpoint_path, max_decoder_steps=max_decoder_steps)
        self.quant_engine = "none"
        if self.quantize_int8:
            model, self.quant_engine = _apply_dynamic_int8(model, self.quantize_mode)
        self.model = model
        self.stft = CompatTacotronSTFT(cache_window_sum=True)
        self.cold_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.model_size_mb = estimate_state_dict_size_mb(self.model)

    def warmup(self) -> float:
        """Run a short one-time warmup pass so JIT/caches/LSTM kernels settle."""
        t0 = time.perf_counter()
        _ = self._synthesize_single_pinyin("ni2 hao3 #3 .")
        return round((time.perf_counter() - t0) * 1000.0, 2)

    def _synthesize_single_pinyin(
        self,
        pinyin_text: str,
        entity_char_spans: Optional[List[Tuple[int, int, str]]] = None,
        clear_speech: bool = False,
        trim_silence: bool = False,
        entity_stretch: float = 1.18,
        compute_mel_spans: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        sequences, text_lengths, _ = prepare_input_sequence([pinyin_text], speaker_id=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.infer(sequences, text_lengths)
            _, mels, _, alignments, mel_lengths = [output.cpu() for output in outputs]
        t1 = time.perf_counter()

        seq_len = int(text_lengths[0].item())
        mel_len = int(mel_lengths[0].item())
        max_possible = self.model.decoder.max_decoder_steps * self.model.n_frames_per_step
        hit_max_steps = mel_len >= max_possible
        mel = mels[0:1, :, :mel_len]

        dec_steps = min(alignments.size(2), max(1, mel_len // self.model.n_frames_per_step))
        align_np = alignments[0, :seq_len, :dec_steps].numpy()
        attn_metrics = compute_attention_metrics(align_np)

        mel_spans: List[Tuple[int, int, str]] = []
        stretch_ratio = 1.0
        if (clear_speech or compute_mel_spans) and entity_char_spans:
            mel_spans = locate_entity_mel_frames(
                align_np,
                pinyin_text,
                entity_char_spans,
                n_frames_per_step=self.model.n_frames_per_step,
            )
            if clear_speech and mel_spans:
                stretched_np, stretch_ratio = stretch_mel_local(
                    mel[0].numpy(),
                    mel_spans,
                    stretch_factor=entity_stretch,
                )
                mel = torch.from_numpy(stretched_np).unsqueeze(0)

        t2 = time.perf_counter()
        magnitudes = self.stft.inv_mel_spectrogram(mel)
        if self.vocoder_mode == "fast_griffin_lim":
            wav = fast_griffin_lim(
                magnitudes,
                self.stft.stft_fn,
                n_iters=self.gl_iters,
                power=1.5,
                momentum=DEFAULT_FAST_GL_MOMENTUM,
            )
        else:
            wav = griffin_lim_compat(
                magnitudes,
                self.stft.stft_fn,
                n_iters=self.gl_iters,
                power=1.5,
            )
        wav = wav.astype(np.float32)
        if trim_silence or clear_speech:
            wav = trim_segment_silence(wav, sr=SAMPLE_RATE)
        t3 = time.perf_counter()

        return {
            "wav": wav,
            "t2_ms": (t1 - t0) * 1000.0,
            "voc_ms": (t3 - t2) * 1000.0,
            "hit_max_steps": hit_max_steps,
            "attn_metrics": attn_metrics,
            "stretch_ratio": stretch_ratio,
            "alignment": align_np,
            "mel_spans": mel_spans,
        }

    def _cache_key(
        self,
        frontend_res: FrontendResult,
        pause_ms: int,
        short_pause_ms: int,
        clear_speech: bool,
        trim_silence: bool,
        entity_stretch: float,
        clarity_eq_gain_db: float,
    ) -> str:
        payload = json.dumps(
            {
                "mode": frontend_res.mode,
                "segments": [(s.pinyin, s.pause_after_ms, s.entity_char_spans) for s in frontend_res.segments],
                "pause_ms": pause_ms,
                "short_pause_ms": short_pause_ms,
                "vocoder_mode": self.vocoder_mode,
                "gl_iters": self.gl_iters,
                "quantize_mode": self.quantize_mode,
                "seed": self.seed,
                "clear_speech": clear_speech,
                "trim_silence": trim_silence,
                "entity_stretch": entity_stretch if clear_speech else 1.0,
                "clarity_eq_gain_db": clarity_eq_gain_db if clear_speech else 0.0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        mode: str = "tn_prosody",
        action: Optional[str] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        short_pause_ms: int = DEFAULT_SHORT_PAUSE_MS,
        cacheable: Optional[bool] = None,
        use_cache: bool = True,
        clear_speech: bool = False,
        trim_silence: bool = False,
        entity_stretch: float = 1.18,
        clarity_eq_gain_db: float = 2.8,
        save_alignment_path: Optional[Path] = None,
    ) -> SynthResult:
        """Synthesize Chinese reply text to a 22050 Hz mono 16-bit PCM WAV file."""
        t_total_0 = time.perf_counter()

        t_fe_0 = time.perf_counter()
        fe_res = process_text(
            text,
            mode=mode,
            action=action,
            pause_ms=pause_ms,
            short_pause_ms=short_pause_ms,
        )
        frontend_ms = (time.perf_counter() - t_fe_0) * 1000.0

        is_cacheable = cacheable if cacheable is not None else (text.strip() in FIXED_CACHEABLE_PROMPTS)
        cache_key = self._cache_key(
            fe_res,
            pause_ms,
            short_pause_ms,
            clear_speech,
            trim_silence,
            entity_stretch,
            clarity_eq_gain_db,
        )
        cache_wav = self.cache_dir / f"{cache_key}.wav"
        cache_meta = self.cache_dir / f"{cache_key}.json"

        if use_cache and is_cacheable and cache_wav.exists() and cache_meta.exists() and save_alignment_path is None:
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_wav, output_wav)
            sr, data = wavfile.read(str(output_wav))
            num_samples = int(data.shape[0])
            dur_s = float(num_samples / sr)
            total_ms = (time.perf_counter() - t_total_0) * 1000.0
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            return SynthResult(
                wav_path=str(output_wav),
                sample_rate=int(sr),
                num_samples=num_samples,
                audio_duration_s=round(dur_s, 4),
                frontend_ms=round(frontend_ms, 2),
                tacotron2_ms=0.0,
                vocoder_ms=0.0,
                postprocess_ms=round(total_ms - frontend_ms, 2),
                total_ms=round(total_ms, 2),
                rtf=round((total_ms / 1000.0) / max(dur_s, 1e-6), 4),
                cache_hit=True,
                hit_max_steps=bool(meta.get("hit_max_steps", False)),
                quantized=self.quantize_int8,
                quantize_mode=self.quantize_mode,
                vocoder_mode=self.vocoder_mode,
                mode=mode,
                trace=fe_res.to_dict(),
                attention_focus=float(meta.get("attention_focus", 0.0)),
                attention_monotonicity=float(meta.get("attention_monotonicity", 0.0)),
                entity_stretch_ratio=float(meta.get("entity_stretch_ratio", 1.0)),
                clarity_band_db=float(meta.get("clarity_band_db", 0.0)),
                clear_speech=clear_speech,
            )

        t2_total_ms = 0.0
        voc_total_ms = 0.0
        any_max_steps = False
        audio_chunks: List[np.ndarray] = []
        focus_list: List[float] = []
        mono_list: List[float] = []
        stretch_list: List[float] = []
        saved_alignments: List[Dict[str, Any]] = []

        if mode == "baseline_raw":
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            pinyins = [s.pinyin for s in fe_res.segments]
            sequences, text_lengths, ids_sorted_decreasing = prepare_input_sequence(pinyins, speaker_id=0)
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.infer(sequences, text_lengths)
                _, mels, _, alignments, mel_lengths = [o.cpu() for o in outputs]
            t1 = time.perf_counter()
            t2_total_ms = (t1 - t0) * 1000.0
            ids_list = ids_sorted_decreasing.numpy().tolist()
            mel_list = [mel[:, :length] for mel, length in zip(mels, mel_lengths)]
            mel_list = [mel_list[ids_list.index(i)] for i in range(len(ids_list))]
            max_possible = self.model.decoder.max_decoder_steps * self.model.n_frames_per_step
            any_max_steps = any(int(l) >= max_possible for l in mel_lengths.tolist())

            for idx_orig in range(len(ids_list)):
                pos = ids_list.index(idx_orig)
                sl = int(text_lengths[pos].item())
                ml = int(mel_lengths[pos].item())
                ds = min(alignments.size(2), max(1, ml // self.model.n_frames_per_step))
                am = compute_attention_metrics(alignments[pos, :sl, :ds].numpy())
                focus_list.append(am["focus_score"])
                mono_list.append(am["monotonicity_score"])

            t2 = time.perf_counter()
            magnitudes = self.stft.inv_mel_spectrogram(torch.cat(mel_list, dim=-1))
            if self.vocoder_mode == "fast_griffin_lim":
                wav = fast_griffin_lim(magnitudes, self.stft.stft_fn, n_iters=self.gl_iters)
            else:
                wav = griffin_lim_compat(magnitudes, self.stft.stft_fn, n_iters=self.gl_iters)
            voc_total_ms = (time.perf_counter() - t2) * 1000.0
            audio_chunks.append(wav.astype(np.float32))
        else:
            for idx, seg in enumerate(fe_res.segments):
                seg_out = self._synthesize_single_pinyin(
                    seg.pinyin,
                    entity_char_spans=seg.entity_char_spans,
                    clear_speech=clear_speech,
                    trim_silence=trim_silence,
                    entity_stretch=entity_stretch,
                    compute_mel_spans=(save_alignment_path is not None),
                )
                t2_total_ms += seg_out["t2_ms"]
                voc_total_ms += seg_out["voc_ms"]
                any_max_steps = any_max_steps or seg_out["hit_max_steps"]
                focus_list.append(seg_out["attn_metrics"]["focus_score"])
                mono_list.append(seg_out["attn_metrics"]["monotonicity_score"])
                if seg_out["mel_spans"]:
                    stretch_list.append(seg_out["stretch_ratio"])

                if save_alignment_path is not None:
                    saved_alignments.append(
                        {
                            "text": seg.text,
                            "pinyin": seg.pinyin,
                            "alignment": seg_out["alignment"].tolist(),
                            "mel_spans": seg_out["mel_spans"],
                        }
                    )

                seg_wav = seg_out["wav"]
                if len(fe_res.segments) > 1 or clear_speech:
                    seg_wav = _normalize_segment(seg_wav)
                audio_chunks.append(seg_wav)

                if seg.pause_after_ms > 0 and idx < len(fe_res.segments) - 1:
                    silence_samples = int(round(seg.pause_after_ms * SAMPLE_RATE / 1000.0))
                    if silence_samples > 0:
                        audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))

        t_post_0 = time.perf_counter()
        full_wav = np.concatenate(audio_chunks) if audio_chunks else np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        if clear_speech and clarity_eq_gain_db > 0:
            full_wav = apply_elderly_clarity_eq(full_wav, sr=SAMPLE_RATE, gain_db=clarity_eq_gain_db)

        clarity_band_db = measure_clarity_band_ratio_db(full_wav, sr=SAMPLE_RATE)
        sr, num_samples = _save_normalized_int16_wav(full_wav, output_wav, SAMPLE_RATE)
        postprocess_ms = (time.perf_counter() - t_post_0) * 1000.0

        avg_focus = round(float(np.mean(focus_list)), 4) if focus_list else 0.0
        avg_mono = round(float(np.mean(mono_list)), 4) if mono_list else 0.0
        avg_stretch = round(float(np.mean(stretch_list)), 4) if stretch_list else 1.0

        align_saved_str: Optional[str] = None
        if save_alignment_path is not None and saved_alignments:
            save_alignment_path.parent.mkdir(parents=True, exist_ok=True)
            save_alignment_path.write_text(json.dumps(saved_alignments, ensure_ascii=False), encoding="utf-8")
            align_saved_str = str(save_alignment_path)

        if use_cache and is_cacheable and not any_max_steps:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output_wav, cache_wav)
            cache_meta.write_text(
                json.dumps(
                    {
                        "raw_text": text,
                        "hit_max_steps": any_max_steps,
                        "attention_focus": avg_focus,
                        "attention_monotonicity": avg_mono,
                        "entity_stretch_ratio": avg_stretch,
                        "clarity_band_db": clarity_band_db,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        dur_s = float(num_samples / sr)
        total_ms = (time.perf_counter() - t_total_0) * 1000.0
        return SynthResult(
            wav_path=str(output_wav),
            sample_rate=sr,
            num_samples=num_samples,
            audio_duration_s=round(dur_s, 4),
            frontend_ms=round(frontend_ms, 2),
            tacotron2_ms=round(t2_total_ms, 2),
            vocoder_ms=round(voc_total_ms, 2),
            postprocess_ms=round(postprocess_ms, 2),
            total_ms=round(total_ms, 2),
            rtf=round((total_ms / 1000.0) / max(dur_s, 1e-6), 4),
            cache_hit=False,
            hit_max_steps=any_max_steps,
            quantized=self.quantize_int8,
            quantize_mode=self.quantize_mode,
            vocoder_mode=self.vocoder_mode,
            mode=mode,
            trace=fe_res.to_dict(),
            attention_focus=avg_focus,
            attention_monotonicity=avg_mono,
            entity_stretch_ratio=avg_stretch,
            clarity_band_db=clarity_band_db,
            clear_speech=clear_speech,
            alignment_path=align_saved_str,
        )

    def synthesize_and_play_pipelined(
        self,
        text: str,
        output_dir: Path,
        stem: str,
        player: AudioPlayer,
        mode: str = "tn_psp_ec",
        action: Optional[str] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        short_pause_ms: int = DEFAULT_SHORT_PAUSE_MS,
        cacheable: Optional[bool] = None,
        use_cache: bool = True,
        clear_speech: bool = True,
        trim_silence: bool = True,
        entity_stretch: float = 1.18,
        clarity_eq_gain_db: float = 2.8,
    ) -> PipelinedPlaybackResult:
        """Synthesize segment 0 and play immediately while synthesizing remaining segments concurrently."""
        t_wall_0 = time.perf_counter()
        fe_res = process_text(
            text,
            mode=mode,
            action=action,
            pause_ms=pause_ms,
            short_pause_ms=short_pause_ms,
        )
        is_cacheable = cacheable if cacheable is not None else (text.strip() in FIXED_CACHEABLE_PROMPTS)

        # Fast path for single segment or cache hit
        if use_cache and is_cacheable:
            whole_wav = output_dir / f"{stem}.wav"
            res = self.synthesize(
                text=text,
                output_wav=whole_wav,
                mode=mode,
                action=action,
                pause_ms=pause_ms,
                short_pause_ms=short_pause_ms,
                cacheable=True,
                use_cache=True,
                clear_speech=clear_speech,
                trim_silence=trim_silence,
                entity_stretch=entity_stretch,
                clarity_eq_gain_db=clarity_eq_gain_db,
            )
            if res.cache_hit:
                ttfa_ms = (time.perf_counter() - t_wall_0) * 1000.0
                outcome = player.play(whole_wav)
                player.stop()
                return PipelinedPlaybackResult(
                    success=outcome.success,
                    ttfa_ms=round(ttfa_ms, 2),
                    total_synth_ms=res.total_ms,
                    total_wall_ms=round((time.perf_counter() - t_wall_0) * 1000.0, 2),
                    audio_duration_s=res.audio_duration_s,
                    segment_count=len(fe_res.segments),
                    underrun_count=0,
                    cache_hit=True,
                    hit_max_steps=res.hit_max_steps,
                    playback_start_mono_ms=outcome.playback_start_mono_ms,
                    playback_end_mono_ms=outcome.playback_end_mono_ms,
                    error_code=outcome.error_code,
                    backend=outcome.backend,
                    segment_wavs=[str(whole_wav)],
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        seg_queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=8)
        producer_errors: List[Exception] = []

        def _producer() -> None:
            try:
                for idx, seg in enumerate(fe_res.segments):
                    t_s0 = time.perf_counter()
                    seg_out = self._synthesize_single_pinyin(
                        seg.pinyin,
                        entity_char_spans=seg.entity_char_spans,
                        clear_speech=clear_speech,
                        trim_silence=trim_silence,
                        entity_stretch=entity_stretch,
                    )
                    seg_wav = _normalize_segment(seg_out["wav"])
                    if clear_speech and clarity_eq_gain_db > 0:
                        seg_wav = apply_elderly_clarity_eq(seg_wav, sr=SAMPLE_RATE, gain_db=clarity_eq_gain_db)
                    if seg.pause_after_ms > 0 and idx < len(fe_res.segments) - 1:
                        sil = np.zeros(int(round(seg.pause_after_ms * SAMPLE_RATE / 1000.0)), dtype=np.float32)
                        seg_wav = np.concatenate([seg_wav, sil])
                    seg_path = output_dir / f"{stem}_seg{idx}.wav"
                    sr, n_samp = _save_normalized_int16_wav(seg_wav, seg_path, SAMPLE_RATE, preserve_level=True)
                    seg_synth_ms = (time.perf_counter() - t_s0) * 1000.0
                    seg_queue.put(
                        {
                            "idx": idx,
                            "wav_path": seg_path,
                            "synth_ms": seg_synth_ms,
                            "duration_s": n_samp / sr,
                            "hit_max_steps": seg_out["hit_max_steps"],
                        }
                    )
            except Exception as exc:
                producer_errors.append(exc)
            finally:
                seg_queue.put(None)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        ttfa_ms = 0.0
        total_synth_ms = 0.0
        total_dur_s = 0.0
        underrun_count = 0
        any_max_steps = False
        seg_paths: List[str] = []
        first_play_start: Optional[int] = None
        last_play_end: Optional[int] = None
        last_backend = "unknown"
        err_code: Optional[str] = None
        success = True

        while True:
            t_wait0 = time.perf_counter()
            was_empty = seg_queue.empty()
            item = seg_queue.get()
            wait_ms = (time.perf_counter() - t_wait0) * 1000.0
            if item is None:
                break

            idx = item["idx"]
            if idx == 0:
                ttfa_ms = (time.perf_counter() - t_wall_0) * 1000.0
            elif was_empty and wait_ms > 5.0:
                underrun_count += 1

            total_synth_ms += item["synth_ms"]
            total_dur_s += item["duration_s"]
            any_max_steps = any_max_steps or item["hit_max_steps"]
            seg_paths.append(str(item["wav_path"]))

            outcome: PlaybackOutcome = player.play(item["wav_path"])
            last_backend = outcome.backend
            if first_play_start is None:
                first_play_start = outcome.playback_start_mono_ms
            last_play_end = outcome.playback_end_mono_ms
            if not outcome.success:
                success = False
                err_code = outcome.error_code or "device_error"
                player.stop()
                break

        thread.join(timeout=30.0)
        player.stop()

        if producer_errors and success:
            success = False
            err_code = f"synth_error:{type(producer_errors[0]).__name__}"

        return PipelinedPlaybackResult(
            success=success,
            ttfa_ms=round(ttfa_ms, 2),
            total_synth_ms=round(total_synth_ms, 2),
            total_wall_ms=round((time.perf_counter() - t_wall_0) * 1000.0, 2),
            audio_duration_s=round(total_dur_s, 4),
            segment_count=len(fe_res.segments),
            underrun_count=underrun_count,
            cache_hit=False,
            hit_max_steps=any_max_steps,
            playback_start_mono_ms=first_play_start,
            playback_end_mono_ms=last_play_end,
            error_code=err_code,
            backend=last_backend,
            segment_wavs=seg_paths,
        )
