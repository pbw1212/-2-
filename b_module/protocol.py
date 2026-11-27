"""组内联调事件协议（schema_version = 1）的构造与校验。

B 只发出一种事件：``TURN_STOP``。字段命名严格取自 docs/interface_spec_v1.md
与全组联调总方案，不做任何扩展——额外的调试信息写进 trace 的外层包装，
不混进事件 payload，避免下游校验失败。

``REASONS`` 的语义（与 docs 中 ``timeout -> error`` 一致）：

    fixed_silence     固定阈值触发，正常轮末
    adaptive_silence  自适应策略触发（含 1500ms 等待上限），正常轮末
    no_speech         整轮未检测到语音，下游按 no_input 处理
    timeout           整轮超过 hard_cap，下游按 error 处理
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

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

REASONS = ("fixed_silence", "adaptive_silence", "no_speech", "timeout")


class ProtocolError(ValueError):
    """事件不符合 schema_version = 1 的公共约定。"""


def monotonic_ms() -> int:
    """树莓派本地单调时钟，整数毫秒。"""
    return int(round(time.monotonic() * 1000))


def make_event(
    event: str,
    session_id: str,
    turn_id: int,
    payload: Dict[str, Any],
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """构造并校验一条事件信封。"""
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "session_id": session_id,
        "turn_id": turn_id,
        "emitted_mono_ms": monotonic_ms() if emitted_mono_ms is None else int(emitted_mono_ms),
        "payload": dict(payload),
    }
    validate_event(envelope)
    return envelope


def make_turn_stop(
    session_id: str,
    turn_id: int,
    decision_audio_ms: float,
    reason: str,
    emitted_mono_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """B -> A：本轮到此结束。

    ``decision_audio_ms`` 是决策时最新可见的音频位置，取整到毫秒。
    """
    payload = {
        "decision_audio_ms": int(round(float(decision_audio_ms))),
        "reason": reason,
    }
    return make_event("TURN_STOP", session_id, turn_id, payload, emitted_mono_ms)


def validate_event(envelope: Dict[str, Any]) -> None:
    """校验公共字段与 TURN_STOP 的 payload。"""
    for key in ("schema_version", "event", "session_id", "turn_id", "emitted_mono_ms", "payload"):
        if key not in envelope:
            raise ProtocolError(f"缺少公共字段 {key!r}")

    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ProtocolError(f"不支持的 schema_version：{envelope['schema_version']}")
    if envelope["event"] not in EVENT_NAMES:
        raise ProtocolError(f"未知事件名：{envelope['event']}")
    if not isinstance(envelope["session_id"], str) or not envelope["session_id"]:
        raise ProtocolError("session_id 必须是非空字符串")
    # 组协议要求整数 turn_id；A 内层 serve 协议用的是字符串，转接由 a_bridge 负责。
    if not isinstance(envelope["turn_id"], int) or isinstance(envelope["turn_id"], bool):
        raise ProtocolError("turn_id 必须是非负整数")
    if envelope["turn_id"] < 0:
        raise ProtocolError("turn_id 不能为负")
    if not isinstance(envelope["emitted_mono_ms"], int) or isinstance(envelope["emitted_mono_ms"], bool):
        raise ProtocolError("emitted_mono_ms 必须是整数毫秒")
    if envelope["emitted_mono_ms"] < 0:
        raise ProtocolError("emitted_mono_ms 不能为负")
    if not isinstance(envelope["payload"], dict):
        raise ProtocolError("payload 必须是对象")

    if envelope["event"] == "TURN_STOP":
        payload = envelope["payload"]
        if "decision_audio_ms" not in payload:
            raise ProtocolError("TURN_STOP 缺少 decision_audio_ms")
        decision = payload["decision_audio_ms"]
        if not isinstance(decision, int) or isinstance(decision, bool) or decision < 0:
            raise ProtocolError("decision_audio_ms 必须是非负整数")
        if payload.get("reason") not in REASONS:
            raise ProtocolError(f"TURN_STOP.reason 非法：{payload.get('reason')!r}")


def to_json_line(envelope: Dict[str, Any]) -> str:
    """序列化为单行 UTF-8 JSON，供逐行管道传输。"""
    validate_event(envelope)
    return json.dumps(envelope, ensure_ascii=False, allow_nan=False)


def from_json_line(line: str) -> Dict[str, Any]:
    """解析并校验一行事件。"""
    envelope = json.loads(line)
    validate_event(envelope)
    return envelope


def check_against_group_validator(envelope: Dict[str, Any]) -> Optional[str]:
    """用 E 模块的 validate_event 交叉校验；E 不可导入时返回原因字符串。

    仅用于测试与自检，运行时不依赖。
    """
    try:
        from e_module.events import validate_event as group_validate  # type: ignore
    except Exception as exc:  # pragma: no cover - 取决于是否在仓库根目录运行
        return f"e_module 不可导入：{exc}"
    group_validate(envelope)
    return None
