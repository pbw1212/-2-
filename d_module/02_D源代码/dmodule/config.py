# -*- coding: utf-8 -*-
"""配置加载：科室/医生词表、模拟复诊档案、策略开关。

所有业务词表放在 config/ 下，策略代码不写死科室或医生名，方便 C 的词表
和 D 的历史继承引用同一份数据。
"""
from __future__ import annotations

import json
import os

# 槽位定义：顺序即固定顺序基线的补问顺序
SLOT_ORDER = ("department", "date", "period", "doctor")
SLOT_NAMES = {
    "department": "科室",
    "date": "日期",
    "period": "时段",
    "doctor": "医生",
}
REQUIRED_SLOTS = ("department", "date", "period")

# 时段规范值 -> 显示文本（E 负责把文本读成“上午/下午”）
PERIOD_DISPLAY = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}


class PolicyConfig(object):
    """策略开关。基线关闭 inherit_history / clarify_ambiguous，改进组打开。"""

    def __init__(self, inherit_history=True, clarify_ambiguous=True,
                 max_reprompt=2, confirm_requires_playback=True,
                 history_confirm_style="candidate", check_slot_dependency=True):
        self.inherit_history = bool(inherit_history)
        self.clarify_ambiguous = bool(clarify_ambiguous)
        self.max_reprompt = int(max_reprompt)
        self.confirm_requires_playback = bool(confirm_requires_playback)
        # candidate：历史值先作为候选随本次登记一起确认；ok：直接当作已确认
        self.history_confirm_style = history_confirm_style
        # 换科室后是否检查医生归属（改进组打开，基线不做依赖检查）
        self.check_slot_dependency = bool(check_slot_dependency)

    def with_overrides(self, overrides=None):
        data = self.to_dict()
        data.update(overrides or {})
        return PolicyConfig(**data)

    def to_dict(self):
        return {
            "inherit_history": self.inherit_history,
            "clarify_ambiguous": self.clarify_ambiguous,
            "max_reprompt": self.max_reprompt,
            "confirm_requires_playback": self.confirm_requires_playback,
            "history_confirm_style": self.history_confirm_style,
            "check_slot_dependency": self.check_slot_dependency,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            inherit_history=data.get("inherit_history", True),
            clarify_ambiguous=data.get("clarify_ambiguous", True),
            max_reprompt=data.get("max_reprompt", 2),
            confirm_requires_playback=data.get("confirm_requires_playback", True),
            history_confirm_style=data.get("history_confirm_style", "candidate"),
            check_slot_dependency=data.get("check_slot_dependency", True),
        )


class Lexicon(object):
    """科室与医生词表。id 是组内词表标识，C 的 slots.value 与 D 的存储值一致。"""

    def __init__(self, departments=None, doctors=None):
        self.departments = {}
        for item in departments or []:
            self.departments[item["id"]] = {
                "id": item["id"],
                "name": item["name"],
                "aliases": list(item.get("aliases") or []),
            }
        self.doctors = {}
        for item in doctors or []:
            self.doctors[item["id"]] = {
                "id": item["id"],
                "name": item["name"],
                "department": item.get("department"),
                "aliases": list(item.get("aliases") or []),
            }

    # ---- 展示 ----
    def department_name(self, dep_id):
        item = self.departments.get(dep_id)
        return item["name"] if item else (dep_id or "")

    def doctor_name(self, doc_id):
        item = self.doctors.get(doc_id)
        return item["name"] if item else (doc_id or "")

    # ---- 关系 ----
    def doctor_belongs_to(self, doc_id, dep_id):
        """医生是否属于该科室；医生未知或科室为空时返回 False。"""
        item = self.doctors.get(doc_id)
        if not item or not dep_id:
            return False
        return item.get("department") == dep_id

    def doctors_of(self, dep_id):
        return [d for d in self.doctors.values() if d.get("department") == dep_id]

    # ---- 别名匹配（供文本替身与校验使用）----
    def match_department(self, text):
        text = (text or "").strip()
        best = None
        for item in self.departments.values():
            for alias in [item["name"]] + item["aliases"]:
                if alias and alias in text:
                    if best is None or len(alias) > len(best[1]):
                        best = (item["id"], alias)
        return best[0] if best else None

    def match_doctor(self, text):
        text = (text or "").strip()
        best = None
        for item in self.doctors.values():
            for alias in [item["name"]] + item["aliases"]:
                if alias and alias in text:
                    if best is None or len(alias) > len(best[1]):
                        best = (item["id"], alias)
        return best[0] if best else None


class Profiles(object):
    """模拟复诊档案：档案 ID、上次科室、上次医生。

    档案选定后，本次草稿才可读取这两个历史值；档案缺失或医生不唯一时，
    D 针对仍缺失或有歧义的字段提问。
    """

    def __init__(self, items=None):
        self.items = {}
        for item in items or []:
            self.items[item["profile_id"]] = {
                "profile_id": item["profile_id"],
                "display_name": item.get("display_name") or item["profile_id"],
                "last_department": item.get("last_department"),
                "last_doctor": item.get("last_doctor"),
                "note": item.get("note", ""),
                "visits": int(item.get("visits", 1)),
            }

    def get(self, profile_id):
        return self.items.get(profile_id)

    def has_unique_doctor(self, profile_id):
        """档案是否给出唯一且与科室相符的医生。"""
        item = self.get(profile_id)
        if not item:
            return False
        doc = item.get("last_doctor")
        dep = item.get("last_department")
        return bool(doc) and bool(dep)

    def as_list(self):
        return list(self.items.values())


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lexicon(path):
    data = _read_json(path)
    return Lexicon(data.get("departments"), data.get("doctors"))


def load_profiles(path):
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("profiles", [])
    return Profiles(data)


def load_policy(path, overrides=None):
    data = _read_json(path) if os.path.exists(path) else {}
    return PolicyConfig.from_dict(data).with_overrides(overrides)


def project_root():
    """返回源码根目录（dmodule 的上一级），便于默认查找 config/。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_path(*parts):
    return os.path.join(project_root(), *parts)
