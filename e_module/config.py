"""Configuration constants and paths for the E module."""

import os
from dataclasses import dataclass
from pathlib import Path

E_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = E_MODULE_DIR.parent


def _resolve_baseline_tacotron2_dir() -> Path:
    """Resolve the course Tacotron2 directory across Mac, teammate hosts, and Pi."""
    env_dir = os.environ.get("SPEECH_TACOTRON2_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()

    candidates = [
        PROJECT_ROOT / "语音信息处理综合实践网盘" / "语音合成" / "tacotron2",
        PROJECT_ROOT / "speech" / "语音合成" / "tacotron2",
        PROJECT_ROOT / "speech" / "tacotron2",
        Path.home() / "speech" / "语音合成" / "tacotron2",
        Path.home() / "speech" / "tacotron2",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


BASELINE_TACOTRON2_DIR = _resolve_baseline_tacotron2_dir()
DEFAULT_CHECKPOINT_PATH = (
    Path(os.environ["SPEECH_TACOTRON2_CKPT"]).expanduser()
    if os.environ.get("SPEECH_TACOTRON2_CKPT", "").strip()
    else BASELINE_TACOTRON2_DIR / "logs" / "checkpoint_latest.pt"
)

OUTPUTS_DIR = E_MODULE_DIR / "outputs"
BASELINE_OUTPUTS_DIR = OUTPUTS_DIR / "baseline"
ABLATION_OUTPUTS_DIR = OUTPUTS_DIR / "ablation"
CACHE_DIR = OUTPUTS_DIR / "prompt_cache"
DOCS_DIR = PROJECT_ROOT / "docs"

# Fixed model architecture parameters matching checkpoint_latest.pt
MODEL_CONFIG = {
    "mask_padding": False,
    "n_mel_channels": 80,
    "n_symbols": 75,
    "symbols_embedding_dim": 512,
    "encoder_kernel_size": 5,
    "encoder_n_convolutions": 3,
    "encoder_embedding_dim": 512,
    "attention_dim": 128,
    "attention_location_n_filters": 32,
    "attention_location_kernel_size": 31,
    "n_frames_per_step": 3,
    "prenet_dim": 256,
    "decoder_rnn_dim": 1024,
    "max_decoder_steps": 1000,
    "gate_threshold": 0.5,
    "decoder_n_lstms": 2,
    "p_decoder_dropout": 0.1,
    "postnet_embedding_dim": 512,
    "postnet_kernel_size": 5,
    "postnet_n_convolutions": 5,
}

# Fixed STFT / audio parameters matching training
AUDIO_CONFIG = {
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "sampling_rate": 22050,
    "mel_fmin": 50.0,
    "mel_fmax": 7600.0,
}

SAMPLE_RATE = 22050
DEFAULT_SEED = 1234
DEFAULT_PAUSE_MS = 260  # Tunable within 200-350 ms per E-module spec
DEFAULT_SHORT_PAUSE_MS = 160
DEFAULT_GL_ITERS = 50
DEFAULT_FAST_GL_ITERS = 25
DEFAULT_FAST_GL_MOMENTUM = 0.99
TARGET_SEGMENT_RMS = 0.16


@dataclass(frozen=True)
class BaselineSample:
    sample_id: str
    category: str
    action: str
    text: str


# Three canonical sample types required by E-module plan + extended clinic set
CANONICAL_SAMPLES = [
    BaselineSample(
        sample_id="short_prompt",
        category="short_prompt",
        action="ask_slot",
        text="请问您想约哪天？",
    ),
    BaselineSample(
        sample_id="multi_field_confirm",
        category="multi_field_confirm",
        action="confirm",
        text="康复科张医生，12月5日 14:30，对吗？",
    ),
    BaselineSample(
        sample_id="long_faq",
        category="long_faq",
        action="faq",
        text="复诊请带好医保卡、身份证和上次的病历本，提前十五分钟到一楼大厅自助机或窗口取号候诊。",
    ),
]

EXTENDED_ABLATION_SAMPLES = CANONICAL_SAMPLES + [
    BaselineSample(
        sample_id="reschedule_confirm",
        category="reschedule_confirm",
        action="confirm",
        text="已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
    ),
    BaselineSample(
        sample_id="faq_with_time",
        category="faq_with_time",
        action="faq",
        text="门诊取号时间是上午08:05到11:30，下午13:30到16:30，请在3号诊室门口候诊。",
    ),
    BaselineSample(
        sample_id="cancel_done",
        category="cancel_done",
        action="done",
        text="好的，本次复诊登记已取消，如需重新预约请随时告诉我。",
    ),
]
