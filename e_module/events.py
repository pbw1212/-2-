"""Unified UTF-8 JSON event protocol (schema_version=1) for modules A-E."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

EVENT_NAMES = {
    "TURN_CONTEXT",
    "LISTEN_START",
    "ASR_TICK",
    "TURN_STOP",
    "ASR_FINAL",
    "NLU_RESULT",
    "PROFILE_SUGGESTION",
    "REPLY_REQUEST",
    "PLAYBACK_RESULT",
}

VALID_DIALOG_PHASES = {"idle", "collecting", "confirming"}
VALID_QUESTION_TYPES = {"open", "targeted", "yes_no"}
VALID_PLAYBACK_STATUSES = {"completed", "failed"}


def monotonic_ms() -> int:
    """Return current host monotonic clock in integer milliseconds."""
    return int(round(time.monotonic() * 1000))


def make_event(
    event: str,
    session_id: str,
    turn_id: int,
    payload: Dict[str, Any],
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Construct and validate an event envelope matching schema_version=1."""
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "session_id": session_id,
        "turn_id": turn_id,
        "emitted_mono_ms": monotonic_ms() if emitted_mono_ms is None else emitted_mono_ms,
        "payload": payload,
    }
    validate_event(envelope)
    return envelope


def make_turn_context(
    session_id: str,
    turn_id: int,
    dialog_phase: str,
    question_type: str,
    expected_slots: List[str],
    reference_date: str,
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Create TURN_CONTEXT with expected_slots primary and derived expected_slot."""
    derived_slot = expected_slots[0] if len(expected_slots) == 1 else None
    payload = {
        "dialog_phase": dialog_phase,
        "question_type": question_type,
        "expected_slots": list(expected_slots),
        "expected_slot": derived_slot,
        "reference_date": reference_date,
    }
    return make_event("TURN_CONTEXT", session_id, turn_id, payload, emitted_mono_ms)


def make_listen_start(
    session_id: str,
    turn_id: int,
    mic_buffer_flushed: bool = True,
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Create LISTEN_START emitted by E once playback ends and context is ready."""
    payload = {
        "mic_buffer_flushed": mic_buffer_flushed,
    }
    return make_event("LISTEN_START", session_id, turn_id, payload, emitted_mono_ms)


def make_reply_request(
    session_id: str,
    turn_id: int,
    reply_id: str,
    reply_text: str,
    action: str = "ask_slot",
    cacheable: Optional[bool] = None,
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Create REPLY_REQUEST sent from D to E."""
    payload: Dict[str, Any] = {
        "reply_id": reply_id,
        "reply_text": reply_text,
        "action": action,
    }
    if cacheable is not None:
        payload["cacheable"] = cacheable
    return make_event("REPLY_REQUEST", session_id, turn_id, payload, emitted_mono_ms)


def make_playback_result(
    session_id: str,
    turn_id: int,
    reply_id: str,
    status: str,
    error_code: Optional[str] = None,
    timing: Optional[Dict[str, Any]] = None,
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Create PLAYBACK_RESULT emitted by E back to D."""
    payload: Dict[str, Any] = {
        "reply_id": reply_id,
        "status": status,
    }
    if error_code is not None:
        payload["error_code"] = error_code
    if timing is not None:
        payload["timing"] = timing
    return make_event("PLAYBACK_RESULT", session_id, turn_id, payload, emitted_mono_ms)


def validate_event(envelope: Dict[str, Any]) -> None:
    """Validate common envelope fields and E-adjacent event payload invariants."""
    required_common = ("schema_version", "event", "session_id", "turn_id", "emitted_mono_ms", "payload")
    for key in required_common:
        if key not in envelope:
            raise ValueError(f"Missing common event field: '{key}'")

    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {envelope['schema_version']}")
    if envelope["event"] not in EVENT_NAMES:
        raise ValueError(f"Unknown event name: {envelope['event']}")
    if not isinstance(envelope["session_id"], str) or not envelope["session_id"]:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(envelope["turn_id"], int) or envelope["turn_id"] < 0:
        raise ValueError("turn_id must be a non-negative integer")
    if not isinstance(envelope["emitted_mono_ms"], int) or envelope["emitted_mono_ms"] < 0:
        raise ValueError("emitted_mono_ms must be a non-negative integer")
    if not isinstance(envelope["payload"], dict):
        raise ValueError("payload must be a dict")

    event = envelope["event"]
    payload = envelope["payload"]

    if event == "TURN_CONTEXT":
        for field in ("dialog_phase", "question_type", "expected_slots", "reference_date"):
            if field not in payload:
                raise ValueError(f"TURN_CONTEXT missing '{field}'")
        if payload["dialog_phase"] not in VALID_DIALOG_PHASES:
            raise ValueError(f"Invalid dialog_phase: {payload['dialog_phase']}")
        if payload["question_type"] not in VALID_QUESTION_TYPES:
            raise ValueError(f"Invalid question_type: {payload['question_type']}")
        if not isinstance(payload["expected_slots"], list):
            raise ValueError("expected_slots must be a list")
        expected_derived = payload["expected_slots"][0] if len(payload["expected_slots"]) == 1 else None
        if "expected_slot" in payload and payload["expected_slot"] != expected_derived:
            raise ValueError(
                f"expected_slot={payload['expected_slot']!r} inconsistent with expected_slots={payload['expected_slots']!r}"
            )
    elif event == "REPLY_REQUEST":
        for field in ("reply_id", "reply_text"):
            if not isinstance(payload.get(field), str) or not payload.get(field):
                raise ValueError(f"REPLY_REQUEST requires non-empty string '{field}'")
    elif event == "PLAYBACK_RESULT":
        if not isinstance(payload.get("reply_id"), str) or not payload.get("reply_id"):
            raise ValueError("PLAYBACK_RESULT requires non-empty string 'reply_id'")
        status = payload.get("status")
        if status not in VALID_PLAYBACK_STATUSES:
            raise ValueError(f"Invalid PLAYBACK_RESULT status: {status}")
        if status == "failed" and not payload.get("error_code"):
            raise ValueError("PLAYBACK_RESULT with status='failed' must specify error_code")


def to_json_line(envelope: Dict[str, Any]) -> str:
    """Serialize validated event as single-line UTF-8 JSON."""
    validate_event(envelope)
    return json.dumps(envelope, ensure_ascii=False)


def from_json_line(raw_line: str) -> Dict[str, Any]:
    """Parse and validate a UTF-8 JSON line into an event dict."""
    obj = json.loads(raw_line)
    validate_event(obj)
    return obj
