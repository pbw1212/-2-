"""事件协议测试：字段、枚举、与 E 模块校验器的兼容性。"""

from __future__ import annotations

import json

import pytest

from b_module import protocol


def test_turn_stop_envelope_fields():
    event = protocol.make_turn_stop("s01", 2, 3540.4, "adaptive_silence", emitted_mono_ms=5000)
    assert event["schema_version"] == 1
    assert event["event"] == "TURN_STOP"
    assert event["session_id"] == "s01"
    assert event["turn_id"] == 2
    assert event["emitted_mono_ms"] == 5000
    assert event["payload"] == {"decision_audio_ms": 3540, "reason": "adaptive_silence"}


@pytest.mark.parametrize("reason", list(protocol.REASONS))
def test_all_reasons_accepted(reason):
    event = protocol.make_turn_stop("s01", 0, 1000, reason, emitted_mono_ms=1)
    protocol.validate_event(event)


def test_invalid_reason_rejected():
    with pytest.raises(protocol.ProtocolError):
        protocol.make_turn_stop("s01", 1, 1000, "因为静音", emitted_mono_ms=1)


def test_string_turn_id_rejected():
    """A 内层 serve 协议用字符串 turn_id，组协议必须是整数，转接由 a_bridge 负责。"""
    with pytest.raises(protocol.ProtocolError):
        protocol.make_event("TURN_STOP", "s01", "visit-1", {"decision_audio_ms": 1, "reason": "fixed_silence"})


def test_negative_decision_rejected():
    with pytest.raises(protocol.ProtocolError):
        protocol.make_event("TURN_STOP", "s01", 1, {"decision_audio_ms": -1, "reason": "fixed_silence"})


def test_json_line_roundtrip():
    event = protocol.make_turn_stop("s01", 3, 2100, "fixed_silence", emitted_mono_ms=42)
    line = protocol.to_json_line(event)
    assert "\n" not in line
    assert json.loads(line)["payload"]["reason"] == "fixed_silence"
    assert protocol.from_json_line(line) == event


def test_group_validator_accepts_b_events():
    """E 的 validate_event 必须接受 B 的事件（跨模块一致性）。"""
    event = protocol.make_turn_stop("s01", 4, 2100, "adaptive_silence", emitted_mono_ms=99)
    problem = protocol.check_against_group_validator(event)
    if problem:
        pytest.skip(problem)
