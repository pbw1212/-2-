"""S8 端点状态机测试：单次触发、四类 reason、文本稳定计时、整机时序。"""

from __future__ import annotations

import copy

import pytest

from b_module import annotate, dataset, paths, strategies, textsim, vad
from b_module.context import TurnContext
from b_module.detector import Detector, Tick
from b_module.protocol import check_against_group_validator, validate_event

A_ROOT = paths.DEFAULT_A_ROOT
needs_audio = pytest.mark.skipif(
    not (A_ROOT / "examples" / "clinic" / "cases.json").is_file(),
    reason="缺少 A 模块样例，跳过音频驱动的用例",
)


@pytest.fixture(scope="module")
def config():
    return paths.load_config()


def make_context(spec, turn_id: int = 1) -> TurnContext:
    return TurnContext(
        session_id="s01", turn_id=turn_id, question_type=spec.question_type,
        expected_slots=list(spec.expected_slots),
    )


def drive(case_id: str, strategy_spec: str, config):
    """把一条整轮音频按 20 ms 帧喂给 Detector，返回 (stop, detector, 标注)。"""
    specs = {spec.case_id: spec for spec in dataset.load_cases(A_ROOT)}
    spec = specs[case_id]
    case_audio = dataset.build_case_audio(spec, a_root=A_ROOT, config=config, use_cache=True)
    record = annotate.annotate_case(spec, case_audio, config)
    timeline = textsim.timeline_for(case_id)

    frame_ms = float(config["audio"]["frame_ms"])
    energies = vad.frame_energies_db(case_audio.samples, case_audio.sample_rate, frame_ms)
    voiced = vad.voiced_frames(energies, float(config["audio"]["detect_dbfs"]))

    strategy = strategies.build_strategy(strategy_spec, config)
    detector = Detector("s01", 1, config, strategy, make_context(spec))
    stop = None
    for index in range(voiced.size):
        audio_ms = (index + 1) * frame_ms
        tick = Tick(
            audio_ms=audio_ms,
            is_speech=bool(voiced[index]),
            partial_text=timeline.text_at(audio_ms),
            partial_updated_ms=timeline.changed_at(audio_ms),
            turn_id=1,
        )
        stop = detector.on_tick(tick)
        if stop is not None:
            break
    return stop, detector, record


# --------------------------------------------------------------------- 单元
def synthetic_config(config, **overrides):
    cfg = copy.deepcopy(config)
    cfg["limits"].update(overrides)
    return cfg


def test_only_one_stop_per_turn(config):
    detector = Detector("s01", 1, config, strategies.build_strategy("fixed:300", config))
    ticks = [
        Tick(100, True, turn_id=1), Tick(200, True, turn_id=1),
        Tick(500, False, turn_id=1), Tick(600, False, turn_id=1),
        Tick(2000, False, turn_id=1), Tick(5000, False, turn_id=1),
    ]
    stops = [detector.on_tick(tick) for tick in ticks]
    assert sum(1 for item in stops if item is not None) == 1
    assert stops[2] is not None
    assert stops[2].decision_audio_ms == 500
    assert detector.stopped is True
    assert detector.dropped_ticks == 3


def test_no_speech_path(config):
    cfg = synthetic_config(config, no_speech_ms=1000)
    detector = Detector("s01", 1, cfg, strategies.build_strategy("fixed:600", cfg))
    stop = None
    for index in range(80):
        stop = detector.on_tick(Tick((index + 1) * 20, False, turn_id=1))
        if stop:
            break
    assert stop is not None
    assert stop.reason == "no_speech"
    assert stop.decision_audio_ms == 1000
    assert stop.basis == "limit:no_speech"


def test_hard_cap_timeout(config):
    cfg = synthetic_config(config, hard_cap_ms=1000)
    detector = Detector("s01", 1, cfg, strategies.build_strategy("fixed:600", cfg))
    stop = None
    for index in range(100):
        stop = detector.on_tick(Tick((index + 1) * 20, True, turn_id=1))  # 一直在说话
        if stop:
            break
    assert stop is not None
    assert stop.reason == "timeout"
    assert stop.decision_audio_ms == 1000


def test_old_turn_ticks_are_dropped(config):
    detector = Detector("s01", 2, config, strategies.build_strategy("fixed:300", config))
    assert detector.on_tick(Tick(100, True, turn_id=1)) is None
    assert detector.dropped_ticks == 1
    assert detector.last_audio_ms is None


def test_audio_ms_must_be_monotonic(config):
    detector = Detector("s01", 1, config, strategies.build_strategy("fixed:300", config))
    detector.on_tick(Tick(200, True, turn_id=1))
    with pytest.raises(ValueError):
        detector.on_tick(Tick(100, True, turn_id=1))


def test_text_change_resets_stability(config):
    detector = Detector("s01", 1, config, strategies.build_strategy("adaptive_text", config),
                        TurnContext("s01", 1, "yes_no", []))
    detector.on_tick(Tick(200, True, "对", 200, 1))
    assert detector.text_state(400).stable_ms == 200
    detector.on_tick(Tick(420, True, "对不起", 400, 1))  # 回改，稳定计时重置
    assert detector.text_state(500).stable_ms == 100
    assert detector.text_change_count == 2


def test_duplicate_text_does_not_reset_stability(config):
    detector = Detector("s01", 1, config, strategies.build_strategy("adaptive_text", config),
                        TurnContext("s01", 1, "yes_no", []))
    detector.on_tick(Tick(200, True, "对", 200, 1))
    detector.on_tick(Tick(400, True, "对", None, 1))
    assert detector.text_state(600).stable_ms == 400
    assert detector.text_change_count == 1


def test_turn_stop_event_is_protocol_valid(config):
    cfg = synthetic_config(config, hard_cap_ms=1000)
    detector = Detector("s01", 3, cfg, strategies.build_strategy("fixed:300", cfg))
    detector.on_tick(Tick(100, True, turn_id=3))
    stop = detector.on_tick(Tick(500, False, turn_id=3))
    assert stop is not None
    event = stop.to_event()
    validate_event(event)
    assert event["turn_id"] == 3
    assert event["payload"]["decision_audio_ms"] == 500
    problem = check_against_group_validator(event)
    if problem:
        pytest.skip(problem)


# ------------------------------------------------------------------ 音频驱动
@needs_audio
def test_fixed_600_truncates_mid_pause_turn(config):
    """dev_01 在句中停顿 1000 ms：600 ms 固定阈值一定提前截断。"""
    stop, _, record = drive("dev_01", "fixed:600", config)
    assert stop is not None
    assert stop.reason == "fixed_silence"
    assert stop.decision_audio_ms == pytest.approx(2000, abs=40)
    assert stop.decision_audio_ms < record["turn_end_ms"]


@needs_audio
def test_fixed_1200_survives_mid_pause(config):
    stop, _, record = drive("dev_01", "fixed:1200", config)
    assert stop is not None
    assert stop.decision_audio_ms == pytest.approx(record["turn_end_ms"] + 1200, abs=40)
    assert stop.decision_audio_ms > record["turn_end_ms"]


@needs_audio
def test_adaptive_question_does_not_truncate_mid_pause(config):
    """open 档取 1500 ms 后，1000 ms 停顿的样本不再被截断。

    这一条锁定了"open 档必须 >= 可观测静音上界"的设计结论；早期把 open 定成
    900 ms 时这里会失败（那正是发现的第一个设计错误）。
    """
    stop, _, record = drive("dev_01", "adaptive_question", config)
    assert stop is not None
    assert stop.reason == "adaptive_silence"
    assert stop.basis == "question:open"
    assert stop.decision_audio_ms > record["turn_end_ms"]


@needs_audio
def test_adaptive_text_waits_for_continuation_cue(config):
    stop, _, record = drive("dev_01", "adaptive_text", config)
    assert stop is not None
    assert stop.decision_audio_ms > record["turn_end_ms"]
    assert "text" in stop.basis


@needs_audio
def test_600_truncates_650ms_pause_but_900_does_not(config):
    early, _, record = drive("test_02", "fixed:600", config)
    late, _, _ = drive("test_02", "fixed:900", config)
    assert early is not None and late is not None
    assert early.decision_audio_ms < record["turn_end_ms"]
    assert late.decision_audio_ms > record["turn_end_ms"]


@needs_audio
def test_short_answer_latency_scales_with_threshold(config):
    """定向短答：阈值越大，等待越久，决策点随之后移。"""
    decisions = {}
    for threshold in ("fixed:300", "fixed:600", "fixed:1200"):
        stop, _, record = drive("smoke_short", threshold, config)
        assert stop is not None
        decisions[threshold] = stop.decision_audio_ms - record["turn_end_ms"]
    assert decisions["fixed:300"] == pytest.approx(300, abs=40)
    assert decisions["fixed:600"] == pytest.approx(600, abs=40)
    assert decisions["fixed:1200"] == pytest.approx(1200, abs=40)


@needs_audio
def test_adaptive_text_shortens_yes_no_turn(config):
    stop, _, record = drive("smoke_short", "adaptive_text", config)
    assert stop is not None
    assert stop.reason == "adaptive_silence"
    assert stop.basis == "text:short_answer"
    assert stop.decision_audio_ms - record["turn_end_ms"] < 600
