"""Unit tests for BiGRU-CRF prosody predictor, EC-Viterbi, Clear Speech, and pipelined TTFA."""

from pathlib import Path

import numpy as np
import pytest

from e_module.audio_player import MockAudioPlayer
from e_module.clear_speech import (
    apply_elderly_clarity_eq,
    compute_attention_metrics,
    locate_entity_mel_frames,
    measure_clarity_band_ratio_db,
    measure_inter_segment_pauses_ms,
    stretch_mel_local,
    trim_segment_silence,
)
from e_module.config import SAMPLE_RATE
from e_module.prosody_model import DEFAULT_PSP_CHECKPOINT, ProsodyPredictor
from e_module.tts_frontend import extract_clinic_entity_spans, process_text


def test_clinic_entity_span_extraction():
    text = "康复科张医生,十二月五日 下午两点半,对吗?"
    spans = extract_clinic_entity_spans(text)
    labels = [lbl for _, _, lbl in spans]
    assert "department_doctor" in labels
    assert "date_time" in labels


def test_ec_viterbi_never_splits_entity_with_phrase_break():
    if not DEFAULT_PSP_CHECKPOINT.exists():
        pytest.skip("Trained BiGRU-CRF checkpoint not generated yet")
    predictor = ProsodyPredictor(DEFAULT_PSP_CHECKPOINT)
    res = process_text(
        "已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
        mode="tn_psp_ec",
        action="confirm",
        prosody_predictor=predictor,
    )
    assert len(res.segments) >= 3
    assert any("二零二六年十二月四日下午" in seg.text for seg in res.segments)
    assert any(seg.entity_char_spans for seg in res.segments)


def test_attention_metrics_and_mel_local_stretch():
    # Synthetic clean diagonal alignment (10 encoder tokens, 20 decoder steps)
    align = np.zeros((10, 20), dtype=np.float32)
    for t in range(20):
        idx = min(9, t // 2)
        prev_idx = max(0, idx - 1)
        if prev_idx != idx:
            align[idx, t] = 0.95
            align[prev_idx, t] = 0.05
        else:
            align[idx, t] = 1.0

    m = compute_attention_metrics(align)
    assert m["focus_score"] > 0.9
    assert m["monotonicity_score"] > 0.95

    raw_py = "kang1 fu4 #1 ke1 #3 ."
    spans = locate_entity_mel_frames(align, raw_py, [(0, 9, "department_doctor")], n_frames_per_step=3)
    assert len(spans) == 1
    mel = np.ones((80, 60), dtype=np.float32)
    stretched, ratio = stretch_mel_local(mel, [(spans[0][0], spans[0][1], "department_doctor")], stretch_factor=1.20)
    assert stretched.shape[0] == 80
    assert stretched.shape[1] > mel.shape[1]
    assert 1.12 <= ratio <= 1.25


def test_trim_silence_calibrated_pause_and_clarity_eq():
    sr = SAMPLE_RATE
    t = np.linspace(0, 0.4, int(0.4 * sr), endpoint=False, dtype=np.float32)
    # Mix fundamental + 2.5 kHz consonant component
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 2600 * t)
    padded = np.concatenate([np.zeros(int(0.15 * sr), dtype=np.float32), tone, np.zeros(int(0.25 * sr), dtype=np.float32)])

    trimmed = trim_segment_silence(padded, sr=sr, top_db=32.0, keep_margin_ms=6.0)
    assert len(trimmed) < len(padded) * 0.7

    pause = np.zeros(int(0.28 * sr), dtype=np.float32)
    joined = np.concatenate([trimmed, pause, trimmed])
    detected = measure_inter_segment_pauses_ms(joined, sr=sr, min_pause_ms=120.0)
    assert len(detected) == 1
    assert abs(detected[0] - 280.0) <= 15.0

    before_db = measure_clarity_band_ratio_db(joined, sr=sr)
    eq_wav = apply_elderly_clarity_eq(joined, sr=sr, gain_db=2.8)
    after_db = measure_clarity_band_ratio_db(eq_wav, sr=sr)
    assert after_db - before_db >= 1.5
    assert float(np.max(np.abs(eq_wav))) <= 1.0


def test_pipelined_playback_zero_underrun_on_stub(tmp_path: Path):
    from e_module.tts_engine import Tacotron2Engine
    from e_module.tts_frontend import FrontendSegment, FrontendResult

    # Monkeypatch process_text and _synthesize_single_pinyin on a dummy instance without loading checkpoint
    eng = object.__new__(Tacotron2Engine)
    eng.vocoder_mode = "fast_griffin_lim"
    eng.gl_iters = 25
    eng.quantize_int8 = False
    eng.quantize_mode = "none"
    eng.cache_dir = tmp_path / "cache"
    eng.seed = 1234

    def _fake_single(pinyin_text, **kwargs):
        # 0.5s audio synthesized in ~0ms
        return {
            "wav": 0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 11025, dtype=np.float32)),
            "t2_ms": 5.0,
            "voc_ms": 5.0,
            "hit_max_steps": False,
            "attn_metrics": {"focus_score": 0.9, "monotonicity_score": 1.0},
            "stretch_ratio": 1.0,
            "alignment": np.eye(3, dtype=np.float32),
            "mel_spans": [],
        }

    eng._synthesize_single_pinyin = _fake_single  # type: ignore[method-assign]
    import e_module.tts_engine as eng_mod

    orig_process = eng_mod.process_text

    def _fake_process(text, **kwargs):
        return FrontendResult(
            raw_text=text,
            mode="tn_psp_ec",
            action="faq",
            normalized_text=text,
            segments=[
                FrontendSegment("段一,", "duan4 yi1 #3 ,", pause_after_ms=200),
                FrontendSegment("段二.", "duan4 er4 #3 .", pause_after_ms=0),
            ],
        )

    eng_mod.process_text = _fake_process
    try:
        player = MockAudioPlayer(speed_factor=0.05)
        res = eng.synthesize_and_play_pipelined(
            text="测试流水线",
            output_dir=tmp_path / "pipe",
            stem="test",
            player=player,
            use_cache=False,
        )
        assert res.success is True
        assert res.segment_count == 2
        assert res.underrun_count == 0
        assert len(player.played_paths) == 2
    finally:
        eng_mod.process_text = orig_process
