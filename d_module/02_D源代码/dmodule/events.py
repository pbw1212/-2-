# -*- coding: utf-8 -*-
"""事件封装与校验。

本模块只负责《离线复诊登记助手 全组联调总方案与接口规范（第一版）》中
公共字段与业务 payload 的定义、构造和校验，不做任何对话决策。
D 模块的所有输入输出都以这里的结构为准，便于和 A/B/C/E 用函数、队列或
本地文件（JSONL）互换，不需要改动上层逻辑。
"""
from __future__ import annotations

import time

# 接口规范第一版
SCHEMA_VERSION = 1

# 事件名（D 消费三个，产出两个）
EVENT_NLU_RESULT = "NLU_RESULT"                  # C -> D，理解结果
EVENT_PROFILE_SUGGESTION = "PROFILE_SUGGESTION"  # A -> D，可选，档案建议
EVENT_PLAYBACK_RESULT = "PLAYBACK_RESULT"        # E -> D，播报完成 / 失败
EVENT_TURN_CONTEXT = "TURN_CONTEXT"              # D -> A/B/C，下一轮上下文
EVENT_REPLY_REQUEST = "REPLY_REQUEST"            # D -> E，播报请求

# 枚举取值
INTENTS = ("faq", "book", "provide_slot", "correct", "confirm", "deny", "cancel", "unknown")
NLU_STATUSES = ("ok", "ambiguous", "no_match", "no_input", "error")
SLOT_STATUSES = ("ok", "ambiguous")
QUESTION_TYPES = ("open", "targeted", "yes_no")
DIALOG_PHASES = ("idle", "collecting", "confirming")
PLAYBACK_STATUSES = ("completed", "failed")

COMMON_FIELDS = ("schema_version", "event", "session_id", "turn_id", "emitted_mono_ms")


class SchemaError(ValueError):
    """事件不符合接口规范时抛出，避免把坏数据写进日志或数据库。"""


def now_ms():
    """本机单调时钟毫秒，仅用于事件排序和耗时统计，不做跨设备对齐。"""
    return int(time.monotonic() * 1000)


def make_event(event, session_id, turn_id, payload=None, emitted_mono_ms=None):
    """按公共字段构造一个事件；payload 传字典。"""
    if not isinstance(session_id, str) or not session_id:
        raise SchemaError("session_id 必须是非空字符串")
    turn_id = int(turn_id)
    if turn_id < 0:
        raise SchemaError("turn_id 必须为非负整数")
    return {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "session_id": session_id,
        "turn_id": turn_id,
        "emitted_mono_ms": int(emitted_mono_ms) if emitted_mono_ms is not None else now_ms(),
        "payload": dict(payload or {}),
    }


def validate_envelope(env):
    """校验公共字段；返回原字典，出错抛 SchemaError。"""
    if not isinstance(env, dict):
        raise SchemaError("事件必须是 JSON 对象")
    for field in COMMON_FIELDS:
        if field not in env:
            raise SchemaError("缺少公共字段 %s" % field)
    if env["schema_version"] != SCHEMA_VERSION:
        raise SchemaError("schema_version 不是 %d" % SCHEMA_VERSION)
    if not isinstance(env["event"], str):
        raise SchemaError("event 必须是字符串")
    if not isinstance(env["session_id"], str) or not env["session_id"]:
        raise SchemaError("session_id 必须是非空字符串")
    if not isinstance(env["turn_id"], int) or env["turn_id"] < 0:
        raise SchemaError("turn_id 必须是非负整数")
    if not isinstance(env["emitted_mono_ms"], int):
        raise SchemaError("emitted_mono_ms 必须是整数")
    if not isinstance(env["payload"], dict):
        raise SchemaError("payload 必须是对象")
    return env


def slot_payload(value, raw, status="ok", candidates=None, updated_turn=None):
    """构造单个槽位的结构化对象。

    value 是规范值（科室/医生用组内词表 ID，日期用 YYYY-MM-DD，时段用
    morning/afternoon）；raw 是用户原表达，保留下来便于排查和显示；
    status 为 ok 或 ambiguous，ambiguous 时 value 必须为 None 且给出 candidates。
    """
    if status not in SLOT_STATUSES:
        raise SchemaError("未知的槽位状态：%s" % status)
    if status == "ambiguous" and value is not None:
        raise SchemaError("歧义槽位的 value 必须为 None")
    item = {"value": value, "raw": raw, "status": status}
    if candidates:
        item["candidates"] = list(candidates)
    if updated_turn is not None:
        item["updated_turn"] = int(updated_turn)
    return item


def nlu_payload(intent, slots=None, status="ok", faq_id=None, answer_text=None,
                retrieval_score=None, history_ref=None):
    """构造 C -> D 的理解结果 payload（也供 C 未接通时的替身使用）。"""
    if intent not in INTENTS:
        raise SchemaError("未知意图：%s" % intent)
    if status not in NLU_STATUSES:
        raise SchemaError("未知理解状态：%s" % status)
    payload = {"intent": intent, "slots": dict(slots or {}), "status": status}
    if faq_id is not None:
        payload["faq_id"] = faq_id
    if answer_text is not None:
        payload["answer_text"] = answer_text
    if retrieval_score is not None:
        payload["retrieval_score"] = float(retrieval_score)
    if history_ref is not None:
        payload["history_ref"] = history_ref
    return payload


def validate_nlu_payload(payload):
    """校验并规范化 C 送来的理解结果，返回 (intent, slots, status) 的浅拷贝。

    slots 只保留接口约定的字段名与结构，未提及的字段直接忽略；
    value 为空且非 ambiguous 的项按“没听清”处理，交给策略决定重问。
    """
    if not isinstance(payload, dict):
        raise SchemaError("NLU_RESULT 的 payload 必须是对象")
    intent = payload.get("intent")
    status = payload.get("status")
    if intent not in INTENTS:
        raise SchemaError("未知意图：%r" % (intent,))
    if status not in NLU_STATUSES:
        raise SchemaError("未知理解状态：%r" % (status,))
    slots = payload.get("slots") or {}
    if not isinstance(slots, dict):
        raise SchemaError("slots 必须是对象")
    clean = {}
    for name, item in slots.items():
        if not isinstance(item, dict):
            raise SchemaError("槽位 %s 必须是对象" % name)
        state = item.get("status", "ok")
        if state not in SLOT_STATUSES:
            raise SchemaError("槽位 %s 的状态非法：%r" % (name, state))
        clean[name] = {
            "value": item.get("value"),
            "raw": item.get("raw"),
            "status": state,
            "candidates": list(item.get("candidates") or []),
        }
    return intent, clean, status


def is_yes(text):
    """确认轮的肯定说法（用于 C 未接通时的文本替身）。"""
    t = (text or "").strip()
    return t in ("对", "对的", "是", "是的", "没错", "可以", "好的", "好", "行",
                 "就这样", "没问题", "确认", "嗯", "对呀", "是的呀")


def is_no(text):
    """确认轮的否定说法。注意“不是张医生”这类带字段的否定属于改口。"""
    t = (text or "").strip()
    return t in ("不对", "不是", "错了", "不要", "不用", "不行", "不", "不是的")
