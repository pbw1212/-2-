"""S6 上下文解析测试。"""

from __future__ import annotations

import json

import pytest

from b_module import context


def event(payload, *, turn_id=2, session_id="s01", name="TURN_CONTEXT"):
    return {
        "schema_version": 1,
        "event": name,
        "session_id": session_id,
        "turn_id": turn_id,
        "emitted_mono_ms": 1000,
        "payload": payload,
    }


def test_open_context_with_empty_slots():
    ctx = context.from_event(event({
        "dialog_phase": "idle", "question_type": "open",
        "expected_slots": [], "expected_slot": None, "reference_date": "2026-11-24",
    }))
    assert ctx.question_type == "open"
    assert ctx.expected_slots == []
    assert ctx.expected_slot is None
    assert ctx.reference_date == "2026-11-24"


def test_yes_no_confirmation_has_empty_slots():
    """D 在确认轮发的 expected_slots 是空数组——B 不能靠它判短答。"""
    ctx = context.from_event(event({
        "dialog_phase": "confirming", "question_type": "yes_no",
        "expected_slots": [], "expected_slot": None, "reference_date": "2026-11-24",
    }))
    assert ctx.question_type == "yes_no"
    assert ctx.expected_slots == []
    assert ctx.expected_slot is None
    assert ctx.is_confirmation is True


def test_targeted_derives_single_slot():
    ctx = context.from_event(event({
        "dialog_phase": "collecting", "question_type": "targeted",
        "expected_slots": ["department"], "reference_date": "2026-11-24",
    }))
    assert ctx.expected_slots == ["department"]
    assert ctx.expected_slot == "department"


def test_multiple_slots_do_not_derive_single():
    ctx = context.from_event(event({
        "dialog_phase": "collecting", "question_type": "open",
        "expected_slots": ["department", "date"], "reference_date": "2026-11-24",
    }))
    assert ctx.expected_slot is None


def test_invalid_question_type_degrades_to_unknown():
    ctx = context.from_event(event({
        "dialog_phase": "idle", "question_type": "闲聊",
        "expected_slots": [], "reference_date": "2026-11-24",
    }))
    assert ctx.question_type == "unknown"


def test_legacy_single_slot_field_is_accepted():
    ctx = context.from_event(event({
        "dialog_phase": "collecting", "question_type": "targeted",
        "expected_slot": "date", "reference_date": "2026-11-24",
    }))
    assert ctx.expected_slots == ["date"]


def test_rejects_other_event_names():
    with pytest.raises(ValueError):
        context.from_event(event({"expected_slots": []}, name="ASR_FINAL"))


def test_rejects_bad_turn_id():
    with pytest.raises(ValueError):
        context.from_event(event({"question_type": "open", "expected_slots": []}, turn_id="visit-1"))


def test_nested_d_log_format(tmp_path):
    """D 的回放日志是 {"event": {...}, "note": null} 的嵌套格式。"""
    line = {
        "event": event({"dialog_phase": "confirming", "question_type": "yes_no",
                        "expected_slots": [], "expected_slot": None,
                        "reference_date": "2026-11-24"}, turn_id=2),
        "note": None,
    }
    path = tmp_path / "d_log.jsonl"
    path.write_text(json.dumps(line, ensure_ascii=False) + "\n" +
                    json.dumps({"event": event({"question_type": "open", "expected_slots": []},
                                               turn_id=1)}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    contexts = context.load_contexts_from_jsonl(path)
    assert [item.turn_id for item in contexts] == [2, 1]
    assert contexts[0].question_type == "yes_no"
    assert context.first_context(contexts, 1).question_type == "open"
    assert context.first_context(contexts, 9) is None
