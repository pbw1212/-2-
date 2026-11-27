"""轮末标注（真值）生成。

对每条重建后的整轮音频，用**比 Detector 更保守**的口径找语音末端：

* 阈值 -50 dBFS（Detector 用 -45），且至少连续 3 帧（60 ms）才算语音；
* 间隔 ≥ 300 ms 才算两个语音段；
* ``turn_end_ms`` 取最后一段语音的结束时刻（帧右边界）。

真值口径比检测口径宽松，因此不会人为制造“提前截断”。标注结果可由
``--verify`` / ``--set-end`` 人工覆盖，重复运行会保留人工字段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import audio, dataset, paths, vad

ANNOTATION_METHOD = {
    "kind": "auto_energy_vad",
    "note": "自动标注 + 人工抽检；真值比检测口径更保守（阈值更低、要求连续帧）",
}


def _load_previous(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(item["case_id"]): item for item in data.get("cases", [])}


def annotate_case(
    spec: dataset.CaseSpec,
    case_audio: dataset.CaseAudio,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    audio_cfg = config["audio"]
    eval_cfg = config["evaluation"]
    frame_ms = float(audio_cfg["frame_ms"])
    threshold = float(audio_cfg["annotate_dbfs"])
    min_speech_frames = int(audio_cfg["annotate_min_speech_frames"])
    min_gap_ms = float(eval_cfg["min_region_gap_ms"])

    samples = case_audio.samples
    sample_rate = case_audio.sample_rate
    energies = vad.frame_energies_db(samples, sample_rate, frame_ms)
    voiced = vad.voiced_frames(energies, threshold)
    regions = vad.speech_regions(
        voiced,
        frame_ms=frame_ms,
        min_gap_ms=min_gap_ms,
        min_speech_ms=min_speech_frames * frame_ms,
    )
    duration = case_audio.duration_ms

    # Detector 看到的静音：用检测阈值（-45）再分一次段。
    # 它可能比“设计停顿”长，因为语音从 -45 衰减到 -60 的尾音在检测器眼里已经是静音。
    detect_regions = vad.speech_regions(
        vad.voiced_frames(energies, float(audio_cfg["detect_dbfs"])),
        frame_ms=frame_ms,
        min_gap_ms=min_gap_ms,
        min_speech_ms=min_speech_frames * frame_ms,
    )
    detect_gaps = vad.region_gaps_ms(detect_regions, frame_ms)

    if not regions:
        speech_end_ms = 0.0
        mid_pause_ms = 0.0
        region_count = 0
    else:
        speech_end_ms = float((regions[-1][1] + 1) * frame_ms)
        gaps = vad.region_gaps_ms(regions, frame_ms)
        mid_pause_ms = float(max(gaps)) if gaps else 0.0
        region_count = len(regions)

    return {
        "case_id": spec.case_id,
        "group": spec.group,
        "question_type": spec.question_type,
        "expected_slot": spec.expected_slot,
        "split": spec.split,
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "duration_ms": round(duration, 2),
        "speech_end_ms": round(speech_end_ms, 2),
        "turn_end_ms": round(speech_end_ms, 2),
        "tail_silence_ms": round(duration - speech_end_ms, 2),
        "region_count": region_count,
        "mid_pause_design_ms": round(float(spec.pause_ms), 2),
        "mid_pause_ms": round(mid_pause_ms, 2),
        "detect_gap_ms": round(max(detect_gaps), 2) if detect_gaps else 0.0,
        "detect_region_count": len(detect_regions),
        "has_mid_pause": bool(spec.pause_ms > 0),
        "part_offsets_ms": [round(value, 2) for value in case_audio.part_offsets_ms],
        "source_parts": list(spec.parts),
        "verified": False,
        "verified_by": None,
        "override_end_ms": None,
    }


def plot_case(
    record: Dict[str, Any],
    case_audio: dataset.CaseAudio,
    config: Dict[str, Any],
    output: Path,
) -> Path:
    """画一条能量曲线：语音段、两档阈值、标注轮末。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio_cfg = config["audio"]
    frame_ms = float(audio_cfg["frame_ms"])
    energies = vad.frame_energies_db(case_audio.samples, case_audio.sample_rate, frame_ms)
    times = np.arange(energies.size) * frame_ms / 1000.0

    fig, axis = plt.subplots(figsize=(8.0, 2.6), dpi=100)
    axis.plot(times, energies, linewidth=0.9, color="#1f77b4")
    axis.axhline(float(audio_cfg["detect_dbfs"]), color="#d62728", linestyle="--",
                 linewidth=0.9, label=f"detect {audio_cfg['detect_dbfs']} dBFS")
    axis.axhline(float(audio_cfg["annotate_dbfs"]), color="#2ca02c", linestyle=":",
                 linewidth=0.9, label=f"annotate {audio_cfg['annotate_dbfs']} dBFS")

    voiced = vad.voiced_frames(energies, float(audio_cfg["annotate_dbfs"]))
    regions = vad.speech_regions(
        voiced,
        frame_ms=frame_ms,
        min_gap_ms=float(config["evaluation"]["min_region_gap_ms"]),
        min_speech_ms=int(audio_cfg["annotate_min_speech_frames"]) * frame_ms,
    )
    for start, end in regions:
        axis.axvspan(start * frame_ms / 1000.0, (end + 1) * frame_ms / 1000.0,
                     color="#2ca02c", alpha=0.12)

    end_s = record["turn_end_ms"] / 1000.0
    axis.axvline(end_s, color="#9467bd", linewidth=1.2,
                 label=f"turn_end {record['turn_end_ms']:.0f} ms")
    for offset in record["part_offsets_ms"][1:]:
        axis.axvline(offset / 1000.0, color="#7f7f7f", linestyle="-.", linewidth=0.8)

    axis.set_xlim(0, max(times[-1] if times.size else 1.0, end_s + 0.2))
    axis.set_ylim(-90, 5)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("dBFS")
    axis.set_title(
        f"{record['case_id']}  group={record['group']}  qtype={record['question_type']}"
        f"  pause={record['mid_pause_design_ms']:.0f}ms",
        fontsize=9,
    )
    axis.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def build_annotations(
    specs: List[dataset.CaseSpec],
    *,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    config: Dict[str, Any] | None = None,
    output: str | Path | None = None,
    plot_dir: str | Path | None = None,
    make_plots: bool = True,
    use_cache: bool = True,
    reset_verified: bool = False,
) -> Dict[str, Any]:
    config = config or paths.load_config()
    output = Path(output) if output else paths.ANNOTATIONS_PATH
    plot_dir = Path(plot_dir) if plot_dir else (paths.RESULTS_DIR / "annotation")
    previous = {} if reset_verified else _load_previous(output)

    records: List[Dict[str, Any]] = []
    for spec in specs:
        case_audio = dataset.build_case_audio(
            spec, a_root=a_root, config=config, use_cache=use_cache
        )
        record = annotate_case(spec, case_audio, config)

        old = previous.get(spec.case_id)
        if old:
            if old.get("verified"):
                record["verified"] = True
                record["verified_by"] = old.get("verified_by")
            if old.get("override_end_ms") is not None:
                record["override_end_ms"] = float(old["override_end_ms"])
                record["turn_end_ms"] = float(old["override_end_ms"])
        note = old.get("note") if old else None
        if note:
            record["note"] = note

        if make_plots:
            path = plot_case(record, case_audio, config,
                             plot_dir / f"{spec.case_id}.png")
            try:
                record["plot"] = str(path.relative_to(paths.REPO_ROOT)).replace("\\", "/")
            except ValueError:
                record["plot"] = str(path)
        records.append(record)

    payload = {
        "schema_version": 1,
        "generated_by": "b_module.annotate.build_annotations",
        "annotation_method": {
            **ANNOTATION_METHOD,
            "threshold_dbfs": config["audio"]["annotate_dbfs"],
            "min_speech_frames": config["audio"]["annotate_min_speech_frames"],
            "min_region_gap_ms": config["evaluation"]["min_region_gap_ms"],
            "frame_ms": config["audio"]["frame_ms"],
            "tolerance_ms": config["evaluation"]["tolerance_ms"],
        },
        "case_count": len(records),
        "verified_count": sum(1 for item in records if item.get("verified")),
        "cases": records,
    }
    paths.ensure_dir(output.parent)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def apply_manual_updates(
    output: str | Path,
    *,
    verify: List[str] = (),
    set_end: List[str] = (),
    note: str | None = None,
) -> Dict[str, Any]:
    """人工抽检：标记 verified、覆盖 turn_end_ms。

    ``set_end`` 的元素形如 ``"dev_01=1540"``（毫秒）。
    """
    output = Path(output)
    if not output.is_file():
        raise FileNotFoundError(f"尚未生成标注文件：{output}")
    data = json.loads(output.read_text(encoding="utf-8"))
    overrides: Dict[str, float] = {}
    for item in set_end:
        if "=" not in item:
            raise ValueError(f"--set-end 需要 CASE=MS 形式：{item!r}")
        case_id, raw_value = item.split("=", 1)
        overrides[case_id.strip()] = float(raw_value)

    known = {item["case_id"] for item in data["cases"]}
    for case_id in list(verify) + list(overrides):
        if case_id not in known:
            raise ValueError(f"标注里没有这个 case：{case_id}")

    for item in data["cases"]:
        if item["case_id"] in verify:
            item["verified"] = True
            item["verified_by"] = note or "manual"
        if item["case_id"] in overrides:
            item["override_end_ms"] = overrides[item["case_id"]]
            item["turn_end_ms"] = overrides[item["case_id"]]
            item["verified"] = True
            item["verified_by"] = note or "manual"
        if note and item["case_id"] in verify:
            item["note"] = note

    data["verified_count"] = sum(1 for item in data["cases"] if item.get("verified"))
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def summary_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """给终端打印用的摘要表。"""
    rows = []
    for item in data["cases"]:
        rows.append(
            {
                "case": item["case_id"],
                "group": item["group"],
                "qtype": item["question_type"],
                "sr": item["sample_rate"],
                "dur_ms": item["duration_ms"],
                "end_ms": item["turn_end_ms"],
                "pause_design": item["mid_pause_design_ms"],
                "pause_meas": item["mid_pause_ms"],
                "regions": item["region_count"],
                "verified": item["verified"],
            }
        )
    return rows
