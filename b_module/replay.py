"""回放驱动（S10）。

把一条重建后的整轮音频按 20 ms 帧推进，合成 ``ASR_TICK`` 喂给 Detector，并把
``TURN_STOP`` 落成组协议事件。两种模式：

* 默认：尽快跑完（批量评测、单元测试用）；
* ``realtime=True``：按真实时间等待，用于现场演示；同时打印语音/文本的关键变化点。

trace 用**原始信封 JSONL** 落盘，E 的 ``latency_analyzer.py`` 可以直接读
（它按 ``TURN_STOP`` 事件名取 ``emitted_mono_ms``，按 ``kind`` 读时序点）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from . import annotate, audio, dataset, detector as detector_mod, paths, protocol, strategies, textsim, vad
from .context import TurnContext
from .detector import Detector, Tick, TurnStop


@dataclass
class ReplayResult:
    case_id: str
    group: str
    question_type: str
    strategy: str
    truth_end_ms: float
    duration_ms: float
    stop: TurnStop | None
    tick_count: int
    dropped_ticks: int
    text_change_count: int
    trace_path: Path | None = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def decision_ms(self) -> float | None:
        return None if self.stop is None else float(self.stop.decision_audio_ms)


def spec_context(spec: dataset.CaseSpec, session_id: str = "s01") -> TurnContext:
    """由 manifest 里的推定字段构造一个 TURN_CONTEXT（阶段一默认上下文）。"""
    return TurnContext(
        session_id=session_id,
        turn_id=1,
        question_type=spec.question_type,
        expected_slots=list(spec.expected_slots),
    )


def iter_ticks(
    case_audio: dataset.CaseAudio,
    timeline: textsim.TextTimeline,
    config: Mapping[str, Any],
    turn_id: int = 1,
) -> Iterator[Tick]:
    """按 20 ms 帧产出 ASR_TICK 序列。"""
    frame_ms = float(config["audio"]["frame_ms"])
    threshold = float(config["audio"]["detect_dbfs"])
    energies = vad.frame_energies_db(case_audio.samples, case_audio.sample_rate, frame_ms)
    voiced = vad.voiced_frames(energies, threshold)
    for index in range(voiced.size):
        audio_ms = (index + 1) * frame_ms
        yield Tick(
            audio_ms=audio_ms,
            is_speech=bool(voiced[index]),
            partial_text=timeline.text_at(audio_ms),
            partial_updated_ms=timeline.changed_at(audio_ms),
            turn_id=turn_id,
        )


def _tick_event(tick: Tick, session_id: str) -> Dict[str, Any]:
    return protocol.make_event(
        "ASR_TICK",
        session_id,
        int(tick.turn_id if tick.turn_id is not None else 0),
        {
            "audio_ms": round(float(tick.audio_ms), 2),
            "is_speech": bool(tick.is_speech),
            "partial_text": tick.partial_text or None,
            "partial_updated_ms": None if tick.partial_updated_ms is None
            else round(float(tick.partial_updated_ms), 2),
        },
    )


def replay_turn(
    case_id: str,
    strategy_spec: str,
    *,
    config: Mapping[str, Any] | None = None,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    context: TurnContext | None = None,
    session_id: str = "s01",
    turn_id: int = 1,
    realtime: bool = False,
    trace_path: str | Path | None = None,
    trace_ticks: bool = False,
    log: Any = print,
    quiet: bool = False,
) -> ReplayResult:
    """回放一条样本，返回决策与统计。"""
    config = config or paths.load_config()
    specs = {spec.case_id: spec for spec in dataset.load_cases(a_root)}
    if case_id not in specs:
        raise KeyError(f"manifest 里没有 {case_id}；可用：{sorted(specs)[:5]}...")
    spec = specs[case_id]

    case_audio = dataset.build_case_audio(spec, a_root=a_root, config=config, use_cache=True)
    record = annotate.annotate_case(spec, case_audio, config)
    timeline = textsim.timeline_for(case_id)
    strategy = strategies.build_strategy(strategy_spec, config)
    ctx = context or spec_context(spec, session_id)

    if not quiet:
        log(f"[{case_id}] strategy={strategy_spec}  question_type={spec.question_type}  "
            f"truth_end={record['turn_end_ms']:.0f}ms  duration={record['duration_ms']:.0f}ms")

    trace_handle = None
    if trace_path:
        trace_file = Path(trace_path)
        paths.ensure_dir(trace_file.parent)
        trace_handle = trace_file.open("w", encoding="utf-8")

        def emit(obj: Dict[str, Any]) -> None:
            trace_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

        emit({
            "kind": "B_REPLAY_START",
            "session_id": session_id,
            "turn_id": turn_id,
            "mono_ms": protocol.monotonic_ms(),
            "detail": {
                "case_id": case_id,
                "group": spec.group,
                "strategy": strategy_spec,
                "question_type": spec.question_type,
                "expected_slots": spec.expected_slots,
                "truth_end_ms": record["turn_end_ms"],
                "mid_pause_ms": record["mid_pause_ms"],
                "source_parts": spec.parts,
            },
        })
    else:
        def emit(obj: Dict[str, Any]) -> None:  # type: ignore[no-redef]
            return

    det = Detector(session_id, turn_id, config, strategy, ctx)
    started = time.perf_counter()
    previous_text = ""
    stop: TurnStop | None = None

    # 语音状态只在“真的有变化”时打印：起音需连续 3 帧语音抗抖动，
    # 停顿需累计 200 ms 才认为是一段静音开始，避免字间掉帧刷屏。
    frame_ms = float(config["audio"]["frame_ms"])
    debounce_frames = max(int(round(60.0 / frame_ms)), 1)
    pause_frames = max(int(round(200.0 / frame_ms)), 1)
    silence_frames = 10**9
    speech_active = False

    try:
        for tick in iter_ticks(case_audio, timeline, config, turn_id):
            if realtime:
                target = started + tick.audio_ms / 1000.0
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            if trace_ticks:
                emit(_tick_event(tick, session_id))

            if tick.is_speech:
                if not quiet and not speech_active and silence_frames >= debounce_frames:
                    log(f"  {tick.audio_ms/1000:6.2f}s  speech on")
                silence_frames = 0
                speech_active = True
            else:
                silence_frames += 1
                if not quiet and speech_active and silence_frames == pause_frames:
                    log(f"  {tick.audio_ms/1000:6.2f}s  speech off"
                        f"（已静音 {pause_frames * frame_ms:.0f}ms）")
                    speech_active = False

            if not quiet and tick.partial_text and tick.partial_text != previous_text:
                log(f"  {tick.audio_ms/1000:6.2f}s  partial   \"{tick.partial_text}\"")
                previous_text = tick.partial_text

            result = det.on_tick(tick)
            if result is not None:
                stop = result
                break

        if trace_handle is not None:
            if stop is not None:
                emit(protocol.make_turn_stop(
                    stop.session_id, stop.turn_id, stop.decision_audio_ms, stop.reason))
                emit({
                    "kind": "B_DECISION",
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "mono_ms": protocol.monotonic_ms(),
                    "detail": {
                        **stop.to_dict(),
                        "truth_end_ms": record["turn_end_ms"],
                        "early_cutoff": stop.decision_audio_ms < record["turn_end_ms"] - float(
                            config["evaluation"]["tolerance_ms"]),
                        "tick_count": det.tick_count,
                        "text_change_count": det.text_change_count,
                    },
                })
            emit({
                "kind": "B_REPLAY_END",
                "session_id": session_id,
                "turn_id": turn_id,
                "mono_ms": protocol.monotonic_ms(),
                "detail": {
                    "case_id": case_id,
                    "strategy": strategy_spec,
                    "stopped": stop is not None,
                    "tick_count": det.tick_count,
                    "dropped_ticks": det.dropped_ticks,
                    "text_change_count": det.text_change_count,
                },
            })
    finally:
        if trace_handle is not None:
            trace_handle.close()

    if not quiet:
        if stop is None:
            log("  未触发 TURN_STOP（回放结束前没有达到任何阈值）")
        else:
            early = stop.decision_audio_ms < record["turn_end_ms"] - float(
                config["evaluation"]["tolerance_ms"])
            verdict = "提前截断" if early else "正常/偏晚"
            log(f"  {stop.decision_audio_ms/1000:6.2f}s  TURN_STOP reason={stop.reason} "
                f"basis={stop.basis} silence={stop.silence_ms:.0f}ms -> {verdict}")

    return ReplayResult(
        case_id=case_id,
        group=spec.group,
        question_type=spec.question_type,
        strategy=strategy_spec,
        truth_end_ms=float(record["turn_end_ms"]),
        duration_ms=float(record["duration_ms"]),
        stop=stop,
        tick_count=det.tick_count,
        dropped_ticks=det.dropped_ticks,
        text_change_count=det.text_change_count,
        trace_path=Path(trace_path) if trace_path else None,
    )


def replay_many(
    case_ids: Sequence[str],
    strategies_spec: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
) -> List[ReplayResult]:
    """批量回放（供 scan 使用，静默输出）。"""
    config = config or paths.load_config()
    results: List[ReplayResult] = []
    for case_id in case_ids:
        for strategy_spec in strategies_spec:
            results.append(replay_turn(
                case_id, strategy_spec, config=config, a_root=a_root, quiet=True,
            ))
    return results
