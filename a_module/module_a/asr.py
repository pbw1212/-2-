from __future__ import annotations

import json
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import sherpa_onnx

from .audio import BYTES_PER_SAMPLE, SAMPLE_RATE, float_samples

MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL = MODULE_DIR / "models" / "zipformer"
DEFAULT_HOTWORDS = MODULE_DIR / "configs" / "clinic_hotwords.json"
SAMPLE_WAV = MODULE_DIR / "examples" / "clinic" / "demo_pause.wav"
MODEL_ID = "x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05"
DEFAULT_BIAS_WEIGHT = 1.5
_TAIL = np.zeros(int(0.64 * SAMPLE_RATE), dtype=np.float32)


def load_hotwords(path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("词表需要把每组词写成一个列表")
    contexts: dict[str, tuple[str, ...]] = {"general": ()}
    for name, terms in raw.items():
        if not name.strip() or name in ("general", "registration") or not isinstance(terms, list):
            raise ValueError(f"词表分组格式不对 {name}")
        words = []
        for term in terms:
            if not isinstance(term, str) or not term.strip() or any(
                char in term for char in ":/\r\n"
            ):
                raise ValueError(f"无法使用这个热词 {term!r}")
            words.append(term.strip())
        contexts[name] = tuple(dict.fromkeys(words))
    contexts["registration"] = tuple(
        dict.fromkeys(word for words in contexts.values() for word in words)
    )
    return contexts


@contextmanager
def native_vocab_path(path: Path):
    # The Windows BPE reader opens a narrow path. The ONNX paths support Unicode.
    if os.name != "nt":
        yield str(path)
        return
    try:
        encoded = str(path).encode("mbcs")
    except UnicodeEncodeError:
        with TemporaryDirectory(prefix="asr-vocab-") as directory:
            temporary = Path(directory) / "bpe.vocab"
            shutil.copyfile(path, temporary)
            yield str(temporary).encode("mbcs")
    else:
        yield encoded


class Zipformer:
    """One local model with a separate decoding state for each turn."""

    def __init__(self, model_dir: Path = DEFAULT_ASR_MODEL, *, threads: int = 2):
        model_dir = model_dir.resolve()
        assets = [
            model_dir / name for name in
            ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt", "bpe.vocab")
        ]
        for asset in assets:
            if not asset.is_file():
                raise FileNotFoundError(f"缺少识别模型文件 {asset}")
        encoder, decoder, joiner, tokens, vocab = assets
        with native_vocab_path(vocab) as native_vocab:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                bpe_vocab=native_vocab,
                modeling_unit="bpe",
                num_threads=threads,
                sample_rate=SAMPLE_RATE,
                enable_endpoint_detection=False,
                decoding_method="modified_beam_search",
                max_active_paths=4,
                provider="cpu",
            )

    def stream(
        self, words: tuple[str, ...] = (), *, weight: float = DEFAULT_BIAS_WEIGHT,
    ) -> ZipformerStream:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("热词权重需要是非负数")
        hints = "/".join(f"{word} :{weight:g}" for word in words) if weight else ""
        return ZipformerStream(self.recognizer, hints)


class ZipformerStream:
    def __init__(self, recognizer: sherpa_onnx.OnlineRecognizer, hints: str):
        self._recognizer = recognizer
        self._stream = recognizer.create_stream(hints or None)
        self._text = ""
        self._has_audio = False
        self._finished = False

    def _decode_ready(self) -> str:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        self._text = self._recognizer.get_result(self._stream)
        return self._text

    def push(self, pcm: bytes) -> str:
        if self._finished:
            raise ValueError("本轮录音已结束")
        if len(pcm) % BYTES_PER_SAMPLE:
            raise ValueError("PCM 需要完整的 16 位采样点")
        if not pcm:
            return self._text
        self._stream.accept_waveform(SAMPLE_RATE, float_samples(pcm))
        self._has_audio = True
        return self._decode_ready()

    def finish(self) -> str:
        if not self._finished:
            self._finished = True
            if self._has_audio:
                # Let the final chunk use its right-hand context before closing.
                self._stream.accept_waveform(SAMPLE_RATE, _TAIL)
                self._stream.input_finished()
                self._decode_ready()
        return self._text
