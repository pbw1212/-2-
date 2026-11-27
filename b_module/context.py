"""轮次上下文：解析 D 的 ``TURN_CONTEXT``。

B 只关心三个字段：

* ``question_type``：``open`` / ``targeted`` / ``yes_no``，决定等待档位；
* ``expected_slots``：数组，B 用它在只有一个字段时派生 ``expected_slot``；
* ``reference_date``：B 不使用，仅留档便于日志核对。

两个必须处理的现实情况：

1. **确认轮的 ``expected_slots`` 是空数组**（D 的接口样例第 5 节）。所以不能靠
   ``expected_slots`` 判“这是短答”，必须优先看 ``question_type``。
2. D 的回放日志是 ``{"event": {...}, "note": ...}`` 的嵌套格式，需要兼容。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

VALID_QUESTION_TYPES = ("open", "targeted", "yes_no")
UNKNOWN_QUESTION_TYPE = "unknown"


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    turn_id: int
    question_type: str
    expected_slots: List[str] = field(default_factory=list)
    dialog_phase: str = "idle"
    reference_date: str | None = None
    question_text: str | None = None

    @property
    def expected_slot(self) -> str | None:
        return self.expected_slots[0] if len(self.expected_slots) == 1 else None

    @property
    def is_confirmation(self) -> bool:
        return self.question_type == "yes_no" or self.dialog_phase == "confirming"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "question_type": self.question_type,
            "expected_slots": list(self.expected_slots),
            "expected_slot": self.expected_slot,
            "dialog_phase": self.dialog_phase,
            "reference_date": self.reference_date,
            "question_text": self.question_text,
        }


def _unwrap(raw: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 D 回放日志的 ``{"event": {...}}`` 包装。"""
    if isinstance(raw.get("event"), dict):
        return raw["event"]
    return raw


def from_event(raw: Dict[str, Any]) -> TurnContext:
    """由一条 ``TURN_CONTEXT`` 事件（或 D 日志行）构造上下文。"""
    event = _unwrap(raw)
    name = event.get("event")
    if name != "TURN_CONTEXT":
        raise ValueError(f"不是 TURN_CONTEXT 事件：{name!r}")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("TURN_CONTEXT 缺少 payload")

    question_type = str(payload.get("question_type") or "").strip()
    if question_type not in VALID_QUESTION_TYPES:
        # 非法或缺失都退化为 unknown，由策略回退到固定档，绝不猜
        question_type = UNKNOWN_QUESTION_TYPE

    slots_raw = payload.get("expected_slots")
    if isinstance(slots_raw, list):
        expected_slots = [str(item) for item in slots_raw]
    else:
        # 兼容只有 expected_slot 单值的历史写法
        single = payload.get("expected_slot")
        expected_slots = [str(single)] if isinstance(single, str) and single else []

    turn_id = event.get("turn_id")
    if not isinstance(turn_id, int) or isinstance(turn_id, bool) or turn_id < 0:
        raise ValueError(f"TURN_CONTEXT.turn_id 非法：{turn_id!r}")

    return TurnContext(
        session_id=str(event.get("session_id") or ""),
        turn_id=turn_id,
        question_type=question_type,
        expected_slots=expected_slots,
        dialog_phase=str(payload.get("dialog_phase") or "idle"),
        reference_date=payload.get("reference_date"),
        question_text=payload.get("question_text"),
    )


def load_contexts_from_jsonl(path: str | Path) -> List[TurnContext]:
    """从 D 的回放日志里抽出全部 ``TURN_CONTEXT``，按出现顺序返回。

    用于拿 D 的**真实**轮次上下文驱动 B 的离线回放，而不是手工编造。
    """
    contexts: List[TurnContext] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - 防御
            raise ValueError(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
        event = _unwrap(raw)
        if event.get("event") == "TURN_CONTEXT":
            contexts.append(from_event(event))
    return contexts


def first_context(contexts: Iterable[TurnContext], turn_id: int) -> TurnContext | None:
    for context in contexts:
        if context.turn_id == turn_id:
            return context
    return None
