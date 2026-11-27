"""S10-S12 测试：回放驱动、指标口径、报告与主图输出。"""

from __future__ import annotations

import json

import pytest

from b_module import dataset, metrics, paths, protocol, replay
from b_module.detector import TurnStop

A_ROOT = paths.DEFAULT_A_ROOT
needs_audio = pytest.mark.skipif(
    not (A_ROOT / "examples" / "clinic" / "cases.json").is_file(),
    reason="缺少 A 模块样例，跳过音频驱动的用例",
)


@pytest.fixture(scope="module")
def config():
    return paths.load_config()


def fake_stop(decision_ms: float, reason: str = "fixed_silence") -> TurnStop:
    return TurnStop(
        session_id="s01", turn_id=1, decision_audio_ms=decision_ms, reason=reason,
        silence_ms=600.0, wait_ms=600.0, basis="fixed:600", strategy="fixed:600",
        state_at_stop="in_speech",
    )


def fake_result(case_id: str, group: str, truth: float, decision: float | None,
                strategy: str = "fixed:600", reason: str = "fixed_silence") -> replay.ReplayResult:
    return replay.ReplayResult(
        case_id=case_id, group=group, question_type="open", strategy=strategy,
        truth_end_ms=truth, duration_ms=truth + 2000,
        stop=None if decision is None else fake_stop(decision, reason),
        tick_count=100, dropped_ticks=0, text_change_count=1,
    )


# --------------------------------------------------------------- 指标口径
def test_evaluate_early_and_latency():
    early = metrics.evaluate(fake_result("a", "open_pause", 3000.0, 2900.0), 20.0)
    assert early.early is True
    assert early.latency_ms is None  # 截断轮不参与延迟统计

    normal = metrics.evaluate(fake_result("b", "open_pause", 3000.0, 3600.0), 20.0)
    assert normal.early is False
    assert normal.latency_ms == pytest.approx(600.0)

    boundary = metrics.evaluate(fake_result("c", "open_pause", 3000.0, 2980.0), 20.0)
    assert boundary.early is False  # 容差内不算截断


def test_evaluate_untriggered():
    outcome = metrics.evaluate(fake_result("d", "open_pause", 3000.0, None), 20.0)
    assert outcome.untriggered is True
    assert outcome.early is False
    assert outcome.decision_ms is None


def test_summarize_counts_and_rates():
    outcomes = [
        metrics.evaluate(fake_result("a", "open_pause", 3000.0, 2000.0), 20.0),
        metrics.evaluate(fake_result("b", "open_pause", 3000.0, 4000.0), 20.0),
        metrics.evaluate(fake_result("c", "open_pause", 3000.0, 4000.0), 20.0),
        metrics.evaluate(fake_result("d", "open_pause", 3000.0, None), 20.0),
    ]
    summary = metrics.summarize(outcomes)
    overall = summary["strategies"]["fixed:600"]["overall"]
    assert overall["turns"] == 4
    assert overall["triggered"] == 3
    assert overall["untriggered"] == 1
    assert overall["early"] == 1
    assert overall["early_denominator"] == 3
    assert overall["early_rate"] == pytest.approx(1 / 3, rel=1e-3)
    assert overall["latency_median_ms"] == pytest.approx(1000.0, abs=1.0)
    assert overall["latency_n"] == 2


def test_summarize_by_group():
    outcomes = [
        metrics.evaluate(fake_result("a", "open_pause", 3000.0, 2000.0), 20.0),
        metrics.evaluate(fake_result("b", "targeted", 1000.0, 1600.0), 20.0),
    ]
    summary = metrics.summarize(outcomes)
    by_group = summary["strategies"]["fixed:600"]["by_group"]
    assert by_group["open_pause"]["early_rate"] == 1.0
    assert by_group["targeted"]["early_rate"] == 0.0
    assert by_group["targeted"]["latency_median_ms"] == pytest.approx(600.0)


# ------------------------------------------------------------ 报告与输出
def test_csv_and_markdown(tmp_path):
    outcomes = [
        metrics.evaluate(fake_result("a", "open_pause", 3000.0, 2000.0), 20.0),
        metrics.evaluate(fake_result("a", "open_pause", 3000.0, 3600.0, "fixed:1200"), 20.0),
    ]
    csv_path = metrics.write_csv(outcomes, tmp_path / "details.csv")
    assert csv_path.is_file()
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("strategy")

    summary = metrics.summarize(outcomes)
    md = metrics.to_markdown(summary, {"case_count": 2, "group_text": "open_pause 2", "tolerance_ms": 20})
    assert "## 1. 总体" in md
    assert "fixed:1200" in md


def test_plot_written(tmp_path):
    pytest.importorskip("matplotlib")
    outcomes = [
        metrics.evaluate(fake_result("a", "open_pause", 3000.0, 2000.0, "fixed:600"), 20.0),
        metrics.evaluate(fake_result("b", "open_pause", 3000.0, 4200.0, "fixed:1200"), 20.0),
    ]
    path = metrics.plot_summary(metrics.summarize(outcomes), tmp_path / "fig.png")
    assert path.is_file() and path.stat().st_size > 1000


# ---------------------------------------------------------------- 回放驱动
@needs_audio
def test_iter_ticks_monotonic(config):
    specs = {spec.case_id: spec for spec in dataset.load_cases(A_ROOT)}
    spec = specs["dev_02"]
    case_audio = dataset.build_case_audio(spec, config=config, use_cache=True)
    ticks = list(replay.iter_ticks(case_audio, replay.textsim.TextTimeline(), config))
    assert len(ticks) == int(case_audio.duration_ms // float(config["audio"]["frame_ms"]))
    assert ticks[0].audio_ms == pytest.approx(float(config["audio"]["frame_ms"]))
    assert all(b.audio_ms > a.audio_ms for a, b in zip(ticks, ticks[1:]))


@needs_audio
def test_replay_fixed_threshold(config):
    result = replay.replay_turn("dev_01", "fixed:600", config=config, quiet=True)
    assert result.stop is not None
    assert result.stop.reason == "fixed_silence"
    assert result.decision_ms == pytest.approx(2000, abs=40)
    assert result.tick_count > 0


@needs_audio
def test_replay_unknown_case(config):
    with pytest.raises(KeyError):
        replay.replay_turn("no_such_case", "fixed:600", config=config, quiet=True)


@needs_audio
def test_trace_contains_protocol_turn_stop(config, tmp_path):
    trace = tmp_path / "trace.jsonl"
    replay.replay_turn("dev_01", "adaptive_text", config=config, quiet=True, trace_path=trace)
    lines = [json.loads(item) for item in trace.read_text(encoding="utf-8").splitlines() if item.strip()]
    stops = [item for item in lines if item.get("event") == "TURN_STOP"]
    assert len(stops) == 1
    protocol.validate_event(stops[0])
    assert stops[0]["payload"]["reason"] == "adaptive_silence"
    kinds = [item.get("kind") for item in lines if "kind" in item]
    assert kinds[0] == "B_REPLAY_START" and kinds[-1] == "B_REPLAY_END"


@needs_audio
def test_replay_many_covers_all(config):
    results = replay.replay_many(["dev_01", "smoke_short"], ["fixed:600", "adaptive_text"],
                                 config=config)
    assert len(results) == 4
    assert {item.strategy for item in results} == {"fixed:600", "adaptive_text"}
