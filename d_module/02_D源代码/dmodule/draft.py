# -*- coding: utf-8 -*-
"""登记草稿、槽位与确认快照。

一份草稿对应一次登记（一个 registration_id）。字段值与来源都记录在槽位里，
确认快照绑定草稿版本，任何有效修改都会让旧快照失效——这是“用户改口后
旧确认不能触发保存”的实现基础。
"""
from __future__ import annotations

from .config import REQUIRED_SLOTS, SLOT_NAMES, SLOT_ORDER

# 槽位状态
MISSING = "missing"          # 还没有值
OK = "ok"                    # 明确且已使用
AMBIGUOUS = "ambiguous"      # C 给出候选，待澄清
CANDIDATE = "candidate"      # 来自历史档案的候选，随本次登记确认
NEEDS_CHECK = "needs_check"  # 依赖变化后待核（例如换科室后的医生）
SKIPPED = "skipped"          # 可选项被用户明确跳过（例如不指定医生）

PRESENT_STATUSES = (OK, CANDIDATE)


class Slot(object):
    """单个槽位：规范值、原表达、来源、状态、更新轮次。"""

    __slots__ = ("field", "value", "raw", "source", "status", "candidates",
                 "updated_turn", "note")

    def __init__(self, field, value=None, raw=None, source="none", status=MISSING,
                 candidates=None, updated_turn=0, note=""):
        self.field = field
        self.value = value
        self.raw = raw
        self.source = source      # user / history / history_ref / none
        self.status = status
        self.candidates = list(candidates or [])
        self.updated_turn = int(updated_turn)
        self.note = note

    @property
    def present(self):
        return self.value is not None and self.status in PRESENT_STATUSES

    @property
    def required(self):
        return self.field in REQUIRED_SLOTS

    def to_dict(self):
        return {
            "field": self.field,
            "value": self.value,
            "raw": self.raw,
            "source": self.source,
            "status": self.status,
            "candidates": self.candidates,
            "updated_turn": self.updated_turn,
            "note": self.note,
        }

    def __repr__(self):
        return "Slot(%s=%r,%s,%s)" % (self.field, self.value, self.source, self.status)


class Draft(object):
    """一次登记的全部状态。"""

    def __init__(self, session_id, registration_id, created_turn=0):
        self.session_id = session_id
        self.registration_id = registration_id
        self.profile_id = None
        self.phase = "idle"
        self.slots = dict((f, Slot(f)) for f in SLOT_ORDER)
        self.revision = 0
        self.pending_field = None
        self.snapshot = None      # 确认快照，见 set_snapshot
        self.reprompt_count = 0
        self.created_turn = created_turn
        self.save_status = None    # None / saved / failed
        self.ended_reason = None
        self.last_action = None

    # ---- 取值 ----
    def get(self, field):
        return self.slots[field]

    def value(self, field):
        return self.slots[field].value

    def to_confirm_fields(self):
        """快照用：只取已出现且仍是候选/明确的字段值。"""
        data = {}
        for field in SLOT_ORDER:
            slot = self.slots[field]
            if slot.present:
                data[field] = {"value": slot.value, "source": slot.source,
                               "status": slot.status}
        return data

    def filled_fields(self):
        return [f for f in SLOT_ORDER if self.slots[f].present]

    def missing_required(self):
        """必填项里仍缺或仍有歧义的字段（按固定顺序）。"""
        return [f for f in SLOT_ORDER
                if self.slots[f].required and not self.slots[f].present]

    def ambiguous_fields(self):
        return [f for f in SLOT_ORDER if self.slots[f].status == AMBIGUOUS]

    def fields_needing_check(self):
        return [f for f in SLOT_ORDER if self.slots[f].status == NEEDS_CHECK]

    def ready_to_confirm(self):
        return not self.missing_required() and not self.ambiguous_fields() \
            and not self.fields_needing_check()

    # ---- 修改 ----
    def apply_slot(self, field, payload, turn, lexicon, check_dependency=True):
        """写入一个来自 C 的槽位，返回是否发生了实质变化。

        换科室时检查医生归属：不再适用的医生清空并标记待核，避免把上一次
        的医生带到新科室上。基线不做这一步（check_dependency=False）。
        """
        slot = self.slots[field]
        status = payload.get("status", "ok")
        changed = False

        if status == "ambiguous":
            # 有歧义时旧值失效，等澄清后再写
            if slot.status != AMBIGUOUS or slot.raw != payload.get("raw"):
                changed = True
            slot.value = None
            slot.raw = payload.get("raw")
            slot.source = "user"
            slot.status = AMBIGUOUS
            slot.candidates = list(payload.get("candidates") or [])
            slot.updated_turn = turn
            slot.note = "歧义，待澄清"
            return changed

        new_value = payload.get("value")
        if new_value is None:
            return False
        if slot.value != new_value or slot.status != OK:
            changed = True
        slot.value = new_value
        slot.raw = payload.get("raw")
        slot.source = "user"
        slot.status = OK
        slot.candidates = []
        slot.updated_turn = turn
        slot.note = ""

        if changed and field == "department" and check_dependency:
            doctor = self.slots["doctor"]
            if doctor.value and doctor.status in PRESENT_STATUSES \
                    and not lexicon.doctor_belongs_to(doctor.value, new_value):
                doctor.value = None
                doctor.status = NEEDS_CHECK
                doctor.note = "科室变更，原医生不再适用"
        return changed

    def mark_revision(self):
        """任何有效修改都会推进草稿版本，并让旧确认快照失效。"""
        self.revision += 1
        self.snapshot = None
        return self.revision

    # ---- 确认快照 ----
    def set_snapshot(self, reply_id, confirm_turn, playback_done=False):
        """绑定草稿版本、播报 ID 与预期确认轮次。"""
        self.snapshot = {
            "revision": self.revision,
            "reply_id": reply_id,
            "confirm_turn": int(confirm_turn),
            "playback_done": bool(playback_done),
            "fields": self.to_confirm_fields(),
        }
        return self.snapshot

    def snapshot_valid_for(self, turn, require_playback=True):
        """只有同一确认轮次、版本未变、该回复确实播完时才允许保存。"""
        snap = self.snapshot
        if not snap:
            return False, "no_snapshot"
        if snap["revision"] != self.revision:
            return False, "revision_changed"
        if int(turn) != int(snap["confirm_turn"]):
            return False, "turn_mismatch"
        if require_playback and not snap["playback_done"]:
            return False, "playback_pending"
        return True, "ok"

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "registration_id": self.registration_id,
            "profile_id": self.profile_id,
            "phase": self.phase,
            "revision": self.revision,
            "slots": dict((f, s.to_dict()) for f, s in self.slots.items()),
            "missing_required": self.missing_required(),
            "ambiguous_fields": self.ambiguous_fields(),
            "snapshot": self.snapshot,
            "save_status": self.save_status,
        }


def slot_label(field):
    return SLOT_NAMES.get(field, field)
