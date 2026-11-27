"""数据集：把 A 的分段录音重建成“整轮音频”，并给出可复现的回放清单。

数据来源（只读，不改动 A 的目录）：

    a_module/examples/clinic/cases.json      每条的 parts / pause_ms / context / targets
    a_module/examples/clinic/audio/*.wav     parts 对应的分段合成录音（22.05 kHz）
    a_module/examples/smoke/short.wav        短答代理样本（yes_no 轮）

重建规则（关键，见 README 的“为什么不能直接拼接”）：

1. 逐段读入，按 Detector 同阈值 -45 dBFS **裁掉首尾静音**（余量默认 0 ms）；
2. 段间插入**恰好** ``pause_ms`` 的静音；
3. 末尾统一补 ``tail_pad_ms``（默认 2000 ms）静音，保证 1500 ms 档也能触发。

分组与 question_type 是按 D 模块交付稿的语义**推定**的，联调后以 D 的
``TURN_CONTEXT`` 为准；推定结果写进 ``data/manifest.json`` 便于复核。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from . import audio, paths

CLINIC_RELATIVE = Path("examples") / "clinic"
SMOKE_RELATIVE = Path("examples") / "smoke"


class DatasetError(RuntimeError):
    """数据缺失或格式不符。"""


@dataclass
class CaseSpec:
    """一条可回放的整轮样本。"""

    case_id: str
    group: str
    question_type: str
    expected_slots: List[str]
    parts: List[str]            # 相对 a_module/ 的路径
    pause_ms: float
    context: str
    targets: List[str]
    split: str
    voice: str
    rate: int

    @property
    def expected_slot(self) -> str | None:
        return self.expected_slots[0] if len(self.expected_slots) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["expected_slot"] = self.expected_slot
        return data


@dataclass
class CaseAudio:
    """重建后的整轮音频。"""

    samples: np.ndarray
    sample_rate: int
    part_offsets_ms: List[float]
    from_cache: bool
    source: str

    @property
    def duration_ms(self) -> float:
        return audio.duration_ms(len(self.samples), self.sample_rate)


def _classify(context: str, targets: Iterable[str], pause_ms: float, case_id: str) -> tuple[str, str, List[str]]:
    """返回 ``(group, question_type, expected_slots)``。"""
    targets = list(targets)
    if case_id.startswith("smoke_"):
        return "yes_no_proxy", "yes_no", []
    if pause_ms > 0 and context == "registration":
        return "open_pause", "open", []
    if context in ("department", "date"):
        return "targeted", "targeted", [context]
    if context == "registration" and not targets:
        return "open_long", "open", []
    return "open_full", "open", []


def load_cases(a_root: str | Path = paths.DEFAULT_A_ROOT, *, include_smoke: bool = True) -> List[CaseSpec]:
    """从 A 的 cases.json 构造回放清单。"""
    a_root = Path(a_root)
    cases_file = a_root / CLINIC_RELATIVE / "cases.json"
    if not cases_file.is_file():
        raise DatasetError(f"找不到 A 的样例清单：{cases_file}")

    raw_cases = json.loads(cases_file.read_text(encoding="utf-8-sig"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetError(f"样例清单格式不对：{cases_file}")

    specs: List[CaseSpec] = []
    for raw in raw_cases:
        case_id = str(raw["id"])
        parts_raw = list(raw.get("parts") or [])
        if not parts_raw:
            raise DatasetError(f"{case_id} 没有 parts")
        # A 的命名规则：<case_id>-01.wav, <case_id>-02.wav, ...
        # 路径统一用正斜杠，保证同一份 manifest 在树莓派（Linux）上也能读
        part_files = [
            f"{CLINIC_RELATIVE.as_posix()}/audio/{case_id}-{index:02d}.wav"
            for index in range(1, len(parts_raw) + 1)
        ]
        for relative in part_files:
            if not (a_root / relative).is_file():
                raise DatasetError(f"{case_id} 缺少分段音频：{a_root / relative}")

        pause_ms = float(raw.get("pause_ms") or 0.0)
        context = str(raw.get("context") or "registration")
        targets = [str(item) for item in (raw.get("targets") or [])]
        group, question_type, expected = _classify(context, targets, pause_ms, case_id)
        specs.append(
            CaseSpec(
                case_id=case_id,
                group=group,
                question_type=question_type,
                expected_slots=expected,
                parts=part_files,
                pause_ms=pause_ms,
                context=context,
                targets=targets,
                split=str(raw.get("split") or ""),
                voice=str(raw.get("voice") or ""),
                rate=int(raw.get("rate") or 0),
            )
        )

    if include_smoke:
        short_wav = f"{SMOKE_RELATIVE.as_posix()}/short.wav"
        if (a_root / short_wav).is_file():
            specs.append(
                CaseSpec(
                    case_id="smoke_short",
                    group="yes_no_proxy",
                    question_type="yes_no",
                    expected_slots=[],
                    parts=[short_wav],
                    pause_ms=0.0,
                    context="confirm",
                    targets=["confirm"],
                    split="proxy",
                    voice="(见 A 的 README：Windows 语音合成)",
                    rate=0,
                )
            )
    return specs


def _fingerprint(spec: CaseSpec, a_root: Path, config: Dict[str, Any]) -> str:
    """内容指纹：源文件 mtime/size + 影响重建结果的配置项。"""
    chunks = [spec.case_id, f"{spec.pause_ms:.1f}"]
    for relative in spec.parts:
        info = (a_root / relative).stat()
        chunks.append(f"{relative}:{info.st_size}:{int(info.st_mtime_ns)}")
    audio_cfg = config["audio"]
    for key in ("frame_ms", "reconstruct_trim_dbfs", "reconstruct_trim_margin_ms", "tail_pad_ms"):
        chunks.append(f"{key}={audio_cfg[key]}")
    return hashlib.sha1("|".join(chunks).encode("utf-8")).hexdigest()[:12]


def build_case_audio(
    spec: CaseSpec,
    *,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    config: Dict[str, Any] | None = None,
    use_cache: bool = True,
) -> CaseAudio:
    """按重建规则生成整轮音频（可缓存到 b_module/.cache/wav/）。"""
    a_root = Path(a_root)
    config = config or paths.load_config()
    audio_cfg = config["audio"]
    frame_ms = float(audio_cfg["frame_ms"])
    trim_dbfs = float(audio_cfg["reconstruct_trim_dbfs"])
    margin_ms = float(audio_cfg["reconstruct_trim_margin_ms"])
    tail_pad_ms = float(audio_cfg["tail_pad_ms"])

    fingerprint = _fingerprint(spec, a_root, config)
    cache_dir = paths.ensure_dir(paths.CACHE_DIR / "wav")
    cache_wav = cache_dir / f"{spec.case_id}__{fingerprint}.wav"
    cache_meta = cache_dir / f"{spec.case_id}__{fingerprint}.json"

    if use_cache and cache_wav.is_file() and cache_meta.is_file():
        samples, sample_rate = audio.read_wav(cache_wav)
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        return CaseAudio(
            samples=samples,
            sample_rate=sample_rate,
            part_offsets_ms=[float(item) for item in meta["part_offsets_ms"]],
            from_cache=True,
            source=str(cache_wav),
        )

    segments: List[np.ndarray] = []
    sample_rate: int | None = None
    for relative in spec.parts:
        samples, rate = audio.read_wav(a_root / relative)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise DatasetError(f"{spec.case_id} 的分段采样率不一致：{sample_rate} vs {rate}")
        trimmed, _ = audio.trim_to_speech(
            samples,
            rate,
            threshold_dbfs=trim_dbfs,
            frame_ms=frame_ms,
            margin_ms=margin_ms,
        )
        if trimmed.size == 0:
            raise DatasetError(f"{spec.case_id} 的分段没有检测到语音：{relative}")
        segments.append(trimmed)

    assert sample_rate is not None
    pauses = [spec.pause_ms] * (len(segments) - 1)
    rebuilt, offsets = audio.concat_with_pauses(segments, pauses, sample_rate)
    rebuilt = audio.pad_trailing_silence(rebuilt, sample_rate, tail_pad_ms)

    if use_cache:
        audio.write_wav(cache_wav, rebuilt, sample_rate)
        cache_meta.write_text(
            json.dumps(
                {
                    "case_id": spec.case_id,
                    "sample_rate": sample_rate,
                    "part_offsets_ms": offsets,
                    "duration_ms": audio.duration_ms(len(rebuilt), sample_rate),
                    "pause_ms": spec.pause_ms,
                    "tail_pad_ms": tail_pad_ms,
                    "fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return CaseAudio(
        samples=rebuilt,
        sample_rate=sample_rate,
        part_offsets_ms=offsets,
        from_cache=False,
        source="rebuilt",
    )


def build_manifest(
    specs: List[CaseSpec],
    *,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    config: Dict[str, Any] | None = None,
    output: str | Path | None = None,
) -> Dict[str, Any]:
    """写出 data/manifest.json。"""
    config = config or paths.load_config()
    output = Path(output) if output else paths.MANIFEST_PATH
    paths.ensure_dir(output.parent)

    group_counts: Dict[str, int] = {}
    for spec in specs:
        group_counts[spec.group] = group_counts.get(spec.group, 0) + 1

    payload = {
        "schema_version": 1,
        "generated_by": "b_module.dataset.build_manifest",
        "a_module_root": "a_module",
        "reconstruction": {
            "trim_dbfs": config["audio"]["reconstruct_trim_dbfs"],
            "trim_margin_ms": config["audio"]["reconstruct_trim_margin_ms"],
            "tail_pad_ms": config["audio"]["tail_pad_ms"],
            "frame_ms": config["audio"]["frame_ms"],
        },
        "group_counts": dict(sorted(group_counts.items())),
        "case_count": len(specs),
        "cases": [spec.to_dict() for spec in specs],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
