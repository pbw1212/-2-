# -*- coding: utf-8 -*-
"""C 的文本替身：按规则把一句话解析成 NLU_RESULT。

C 未接通时，D 用这里的事件走完整流程（方案里的“联调依赖未就绪时的处理”）。
它只做规则匹配，不参与 D 的决策；输出字段与《接口规范》完全一致，正式联调
换成 C 的真实结果即可，D 侧不需要改代码。
"""
from __future__ import annotations

import datetime
import json
import re

from . import events as ev

WEEKDAY_CHARS = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
}

FAQ_KEYWORDS = {
    "带什么": "faq_what_to_bring",
    "要带": "faq_what_to_bring",
    "准备什么": "faq_what_to_bring",
    "流程": "faq_process",
    "怎么走": "faq_process",
    "门诊时间": "faq_time",
    "几点": "faq_time",
    "停车": "faq_parking",
    "取药": "faq_medicine",
    "报销": "faq_cost",
    "医保": "faq_cost",
    "空腹": "faq_fasting",
    "抽血": "faq_fasting",
    "取消": "faq_cancel",
}

CANCEL_WORDS = ("取消这次登记", "取消登记", "不登记了", "不约了", "算了不约", "退出登记", "取消预约")
CONFIRM_WORDS = ("对", "对的", "是", "是的", "没错", "可以", "好的", "好", "行", "就这样",
                 "没问题", "确认", "嗯", "对呀")
DENY_WORDS = ("不对", "不是", "错了", "不要", "不用", "不行", "不", "不是的", "不用了", "不指定")
CORRECT_WORDS = ("改成", "改到", "改为", "换到", "换成", "改一下", "更正", "改个", "另外")
BOOK_WORDS = ("预约", "挂号", "登记", "复诊", "我想约", "帮我约", "约一下")

HISTORY_WORDS = ("还是上次那位", "还是上次的医生", "还是上次", "上次那位", "老样子", "还是老医生")


def _norm(text):
    """全角转半角、去空格，保留“不”“改”等影响意图的字。"""
    if not text:
        return ""
    out = []
    for ch in text.strip():
        code = ord(ch)
        if code == 0x3000:
            continue
        if 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        if ch.isspace():
            continue
        out.append(ch)
    return "".join(out)


def _week_start(day):
    return day - datetime.timedelta(days=day.weekday())


def _parse_date(text, reference_date):
    """解析日期表达，返回 (kind, value, candidates)。

    kind 为 none / ok / ambiguous。只有“周五”这类没说清周次的表达返回
    ambiguous，并给出本周与下周两个候选，由 D 决定追问方式。
    """
    t = text
    if isinstance(reference_date, str):
        ref = datetime.datetime.strptime(reference_date, "%Y-%m-%d").date()
    else:
        ref = reference_date

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return "ok", m.group(0), []
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", t)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = ref.year + (1 if (month, day) < (ref.month, ref.day) else 0)
        return "ok", "%04d-%02d-%02d" % (year, month, day), []

    if "今天" in t:
        return "ok", ref.isoformat(), []
    if "明天" in t:
        return "ok", (ref + datetime.timedelta(days=1)).isoformat(), []
    if "后天" in t:
        return "ok", (ref + datetime.timedelta(days=2)).isoformat(), []

    m = re.search(r"(下下|下|本|这|上)?(?:周|星期|礼拜)([一二三四五六日天1-7])", t)
    if m:
        prefix, ch = m.group(1), m.group(2)
        idx = WEEKDAY_CHARS[ch]
        start = _week_start(ref)
        if prefix in ("本", "这"):
            target = start + datetime.timedelta(days=idx)
        elif prefix == "下":
            target = start + datetime.timedelta(days=7 + idx)
        elif prefix == "上":
            target = start + datetime.timedelta(days=idx - 7)
        else:
            this_day = start + datetime.timedelta(days=idx)
            next_day = this_day + datetime.timedelta(days=7)
            label = ch
            cands = [
                {"value": this_day.isoformat(), "raw": "本周" + label,
                 "label": "本周" + label},
                {"value": next_day.isoformat(), "raw": "下周" + label,
                 "label": "下周" + label},
            ]
            return "ambiguous", None, cands
        return "ok", target.isoformat(), []
    return "none", None, []


def _parse_period(text):
    t = text
    if any(w in t for w in ("上午", "早上", "早晨", "一早")):
        return "morning"
    if any(w in t for w in ("下午", "中午", "傍晚", "午后")):
        return "afternoon"
    if any(w in t for w in ("晚上", "夜里")):
        return "evening"
    return None


def load_faq_stub(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("faqs", data)


def match_faq(text, faqs):
    """关键词命中即返回标准答案；多命中取关键词最长的条目。"""
    best = None
    for item in faqs:
        keys = list(item.get("keywords") or [])
        for key in keys:
            if key and key in text:
                if best is None or len(key) > best[0]:
                    best = (len(key), item)
    if best is None:
        return None
    return best[1]


def parse(text, lexicon, reference_date="2026-11-24", context=None, faqs=None,
          profile_id=None):
    """返回 NLU_RESULT 的 payload。"""
    raw = text or ""
    t = _norm(raw)
    context = context or {}
    pending = context.get("pending_field")
    phase = context.get("dialog_phase")

    # ---- 取消 ----
    if any(w in t for w in CANCEL_WORDS):
        return ev.nlu_payload("cancel", {}, "ok")

    # ---- 明确不指定医生 / 阶段否定 ----
    if any(w in t for w in ("不指定", "不用指定", "不选医生", "不用了", "不用医生")) \
            or (pending == "doctor" and t in ("不用", "不", "不要", "随便", "无所谓")):
        return ev.nlu_payload("deny", {}, "ok")

    # ---- 改口 ----
    is_correct = any(w in t for w in CORRECT_WORDS) and not t.startswith("不")

    # ---- 先抽字段，再定意图（短答依赖上下文）----
    slots = {}

    dep = lexicon.match_department(t)
    if dep:
        slots["department"] = ev.slot_payload(dep, raw, "ok")

    doc = lexicon.match_doctor(t)
    if doc:
        slots["doctor"] = ev.slot_payload(doc, raw, "ok")

    kind, value, cands = _parse_date(t, reference_date)
    if kind == "ok":
        slots["date"] = ev.slot_payload(value, raw, "ok")
    elif kind == "ambiguous":
        slots["date"] = ev.slot_payload(None, raw, "ambiguous", candidates=cands)

    period = _parse_period(t)
    if period:
        slots["period"] = ev.slot_payload(period, raw, "ok")

    history_ref = None
    if any(w in t for w in HISTORY_WORDS):
        history_ref = "last_doctor"

    # ---- FAQ ----
    if faqs:
        hit = match_faq(t, faqs)
        if hit and (("?" in raw or "？" in raw) or any(
                k in t for k in ("带什么", "流程", "几点", "停车", "取药", "报销", "空腹",
                                 "怎么办", "怎么走", "吗"))):
            return ev.nlu_payload("faq", slots, "ok", faq_id=hit["faq_id"],
                                  answer_text=hit["answer_text"],
                                  retrieval_score=hit.get("score", 0.8))

    # ---- 意图判定 ----
    if is_correct:
        return ev.nlu_payload("correct", slots, "ok", history_ref=history_ref)
    if history_ref and not slots:
        return ev.nlu_payload("book", {}, "ok", history_ref=history_ref)
    if any(w in t for w in BOOK_WORDS):
        return ev.nlu_payload("book", slots, "ok", history_ref=history_ref)
    if slots:
        return ev.nlu_payload("provide_slot", slots, "ok", history_ref=history_ref)
    if history_ref:
        return ev.nlu_payload("book", {}, "ok", history_ref=history_ref)

    if phase == "confirming" or pending in (None, ""):
        if t in CONFIRM_WORDS:
            return ev.nlu_payload("confirm", {}, "ok")
        if t in DENY_WORDS:
            return ev.nlu_payload("deny", {}, "ok")
    else:
        if t in CONFIRM_WORDS:
            return ev.nlu_payload("confirm", {}, "ok")
        if t in DENY_WORDS:
            return ev.nlu_payload("deny", {}, "ok")

    return ev.nlu_payload("unknown", {}, "no_match")
