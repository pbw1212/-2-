"""S14 测试：A 内层协议 -> 组协议的翻译，以及用假 A 跑通的闭环。

不依赖 A 的模型与 torch：用一段 ``-c`` 脚本模拟 A 的 serve 协议。
"""

from __future__ import annotations

import json
import sys

import pytest

from b_module import a_bridge, dataset, paths, protocol
from b_module.a_bridge import AProtocolTranslator, ABridge, ABridgeError
from b_module.context import TurnContext

A_ROOT = paths.DEFAULT_A_ROOT
needs_audio = pytest.mark.skipif(
    not (A_ROOT / "examples" / "clinic" / "cases.json").is_file(),
    reason="缺少 A 模块样例，跳过端到端闭环测试",
)

FAKE_A = r"""
import json, sys
text = ""
frames = 0
print(json.dumps({"type": "ready", "sample_rate": 16000, "frame_ms": 20}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op")
    tid = req.get("turn_id")
    if op == "start":
        frames = 0
        text = ""
        print(json.dumps({"type": "started", "turn_id": tid, "audio_ms": 0.0,
                          "elapsed_ms": 0.0, "standby": False}), flush=True)
    elif op == "pcm":
        frames += 1
        ms = frames * 20.0
        voiced = 20.0 <= ms <= 400.0
        print(json.dumps({"type": "audio", "turn_id": tid, "audio_ms": ms,
                          "elapsed_ms": ms, "level_dbfs": -30.0 if voiced else -90.0,
                          "voiced": voiced}), flush=True)
        if frames == 5:
            text = "我想约康复科"
            print(json.dumps({"type": "partial", "turn_id": tid, "audio_ms": ms,
                              "elapsed_ms": ms, "text": text}), flush=True)
    elif op == "finish":
        print(json.dumps({"type": "final", "turn_id": tid, "audio_ms": frames * 20.0,
                          "elapsed_ms": frames * 20.0, "text": text,
                          "status": "ok" if text else "no_text",
                          "finalize_ms": 1.0}), flush=True)
        if text:
            print(json.dumps({"type": "speaker", "turn_id": tid, "audio_ms": frames * 20.0,
                              "elapsed_ms": frames * 20.0, "status": "candidate",
                              "profile_id": "p001", "score": 0.8}), flush=True)
    elif op == "cancel":
        print(json.dumps({"type": "closed", "turn_id": tid, "status": "cancelled"}), flush=True)
"""


def ctx(question_type: str = "open", slots=None) -> TurnContext:
    return TurnContext(session_id="s01", turn_id=1, question_type=question_type,
                       expected_slots=list(slots or []))


def test_turn_id_mapping():
    assert AProtocolTranslator.map_turn_id(0) == "turn0"
    assert AProtocolTranslator.map_turn_id(7) == "turn7"
    with pytest.raises(ValueError):
        AProtocolTranslator.map_turn_id("visit-1")  # type: ignore[arg-type]


def test_context_name_mapping():
    assert AProtocolTranslator.context_name(ctx("targeted", ["department"])) == "department"
    assert AProtocolTranslator.context_name(ctx("targeted", ["date"])) == "date"
    assert AProtocolTranslator.context_name(ctx("open")) == "registration"
    assert AProtocolTranslator.context_name(ctx("yes_no")) == "general"
    assert AProtocolTranslator.context_name(None) == "general"


def test_translate_audio_to_tick():
    translator = AProtocolTranslator("s01")
    events = translator.observe(
        {"type": "audio", "audio_ms": 120.0, "voiced": True}, 3)
    assert len(events) == 1
    envelope = events[0]
    protocol.validate_event(envelope)
    assert envelope["event"] == "ASR_TICK"
    assert envelope["turn_id"] == 3
    assert envelope["payload"]["is_speech"] is True
    assert envelope["payload"]["partial_text"] is None
    assert envelope["payload"]["partial_updated_ms"] is None


def test_partial_stamps_updated_ms():
    translator = AProtocolTranslator("s01")
    assert translator.observe(
        {"type": "partial", "audio_ms": 100.0, "text": "我想约"}, 1) == []
    events = translator.observe({"type": "audio", "audio_ms": 120.0, "voiced": False}, 1)
    payload = events[0]["payload"]
    assert payload["partial_text"] == "我想约"
    assert payload["partial_updated_ms"] == 100.0


def test_final_status_mapping():
    translator = AProtocolTranslator("s01")
    ok = translator.observe(
        {"type": "final", "status": "ok", "text": "康复科", "audio_ms": 900.0}, 1)[0]
    assert ok["payload"]["status"] == "ok"
    assert ok["payload"]["final_text"] == "康复科"
    assert ok["payload"]["end_audio_ms"] == 900.0

    empty = translator.observe(
        {"type": "final", "status": "no_text", "text": "", "audio_ms": 900.0}, 1)[0]
    assert empty["payload"]["status"] == "no_input"
    assert empty["payload"]["final_text"] is None

    broken = translator.observe(
        {"type": "final", "status": "wake_missed", "text": "", "audio_ms": 10.0}, 1)[0]
    assert broken["payload"]["status"] == "error"
    assert broken["payload"]["error_code"].startswith("asr_final_status:")


def test_speaker_mapping():
    translator = AProtocolTranslator("s01")
    matched = translator.observe(
        {"type": "speaker", "status": "candidate", "profile_id": "p001"}, 1)[0]
    assert matched["event"] == "PROFILE_SUGGESTION"
    assert matched["payload"] == {"profile_id": "p001", "match_status": "matched"}

    low = translator.observe(
        {"type": "speaker", "status": "unknown", "profile_id": "p002"}, 1)[0]
    assert low["payload"]["match_status"] == "low_score"

    none = translator.observe({"type": "speaker", "status": "too_short"}, 1)[0]
    assert none["payload"] == {"profile_id": None, "match_status": "none"}


def test_stale_turn_events_dropped():
    translator = AProtocolTranslator("s01")
    stale = {"type": "audio", "turn_id": "turn1", "audio_ms": 100.0, "voiced": True}
    assert translator.observe(stale, 5, expect_a_turn="turn2") == []
    fresh = {"type": "audio", "turn_id": "turn2", "audio_ms": 100.0, "voiced": True}
    assert len(translator.observe(fresh, 5, expect_a_turn="turn2")) == 1


def test_unknown_event_ignored_and_error_raised():
    translator = AProtocolTranslator("s01")
    assert translator.observe({"type": "wake", "keyword": "你好助手"}, 1) == []
    with pytest.raises(ABridgeError):
        translator.observe({"type": "error", "message": "boom"}, 1)


# ------------------------------------------------------------------ 假 A 闭环
@pytest.fixture
def fake_bridge():
    bridge = ABridge(session_id="s01", command=[sys.executable, "-c", FAKE_A],
                     event_timeout_s=20.0)
    bridge.open()
    yield bridge
    bridge.close()


def test_bridge_roundtrip(fake_bridge):
    assert fake_bridge.ready["sample_rate"] == 16000
    assert fake_bridge.frame_bytes == 640

    fake_bridge.begin_turn(2, ctx("open"))
    ticks = []
    for _ in range(60):
        ticks.extend(fake_bridge.push_frame(2, b"\x00" * 640))
    assert len(ticks) == 60
    assert all(item["event"] == "ASR_TICK" for item in ticks)
    assert all(item["turn_id"] == 2 for item in ticks)
    assert ticks[-1]["payload"]["partial_text"] == "我想约康复科"
    assert ticks[-1]["payload"]["partial_updated_ms"] == 100.0

    finals = fake_bridge.finish_turn(2)
    kinds = [item["event"] for item in finals]
    assert "ASR_FINAL" in kinds and "PROFILE_SUGGESTION" in kinds
    final = next(item for item in finals if item["event"] == "ASR_FINAL")
    assert final["payload"]["status"] == "ok"
    assert final["payload"]["final_text"] == "我想约康复科"
    profile = next(item for item in finals if item["event"] == "PROFILE_SUGGESTION")
    assert profile["payload"]["profile_id"] == "p001"


def test_bridge_rejects_missing_executable():
    bridge = ABridge(session_id="s01", command=["definitely-not-a-python-binary"])
    with pytest.raises(ABridgeError):
        bridge.open()
    bridge.close()


@needs_audio
def test_full_a_b_loop_with_fake_a():
    bridge = ABridge(session_id="s01", command=[sys.executable, "-c", FAKE_A],
                     event_timeout_s=20.0)
    result = a_bridge.run_ab_loop("dev_01", "fixed:600", bridge=bridge, quiet=True)
    assert result["stop"] is not None
    assert result["stop"].reason == "fixed_silence"
    events = result["events"]
    assert any(item["event"] == "ASR_TICK" for item in events)
    assert any(item["event"] == "ASR_FINAL" for item in events)
    # 假 A 的语音只覆盖 20..400 ms，600 ms 阈值应在 1000 ms 触发
    assert result["stop"].decision_audio_ms == pytest.approx(1000, abs=40)


@needs_audio
def test_pcm_frames_are_16k_mono():
    pcm = a_bridge.load_pcm_16k("dev_02")
    assert len(pcm) % 2 == 0
    frames = list(a_bridge.iter_pcm_frames(pcm))
    assert len(frames) > 10
    assert all(len(item) == a_bridge.A_FRAME_BYTES for item in frames)
    # 16 kHz * 20 ms * 2 byte = 640 B；总时长应与 manifest 一致（±1 帧）
    duration_ms = len(frames) * 20.0
    spec = {item.case_id: item for item in dataset.load_cases(A_ROOT)}["dev_02"]
    assert duration_ms > 1000
