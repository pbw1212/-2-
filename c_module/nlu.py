"""将 ASR_FINAL 与当前 TURN_CONTEXT 转成 NLU_RESULT。"""

import json
import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

from .faq import FaqIndex


DATA_DIR = Path(__file__).with_name("data")
DATE_EXPRESSION = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)|今天|明天|后天|"
    r"(?:(?P<week>下下|下|这|本))?(?:周|星期|礼拜)(?P<day>[一二三四五六日天])"
)
FAQ_QUESTION = re.compile(
    r"请问|问一下|想问|怎么|怎样|如何|什么|哪[儿里个些种]|在哪|多少|多久|几(?:点|楼|分钟|小时|天)|"
    r"有没有|要不要|需不需要|带不带"
)
YES_NO_QUESTION = re.compile(r"吗$|行不行|能否|能不能|可不可以")
BOOK_REQUEST = re.compile(r"预约|想约|想挂|帮我约|帮我挂|挂号|登记")
DIRECT_BOOK_REQUEST = re.compile(
    r"^(?:我想|我要|我打算|想|帮我|请帮我|(?:能不能|可不可以|能否)(?:请你|帮我)?)"
    r"(?:预约|约|挂号|挂|登记)"
)
CANCEL_REQUEST = re.compile(r"^(?:请|我想|我要|帮我)?取消(?:(?:这次|本次)?(?:复诊)?(?:登记|预约|挂号))?[吧了]?$")
CORRECTION = re.compile(r"改期到|改成|改为|改到|换成|换到|更正为")
RESTATEMENT = re.compile(r"^不是.*?而?是")
AFFIRMATION = re.compile(r"(?:对|对的|是|是的|没错|确认|可以|好|好的)[啊呀吧]?$")
NEGATION = re.compile(r"(?:不|不对|不是|不是的|不行|否|不确认)[啊呀吧]?$")
WEEKDAY = {day: number for number, day in enumerate("一二三四五六日")}
WEEKDAY["天"] = 6
WEEK_OFFSET = {"这": 0, "本": 0, "下": 1, "下下": 2}


def load_data(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def make_event(event, session_id, turn_id, payload):
    return {
        "schema_version": 1,
        "event": event,
        "session_id": session_id,
        "turn_id": turn_id,
        "emitted_mono_ms": time.monotonic_ns() // 1_000_000,
        "payload": payload,
    }


def normalize(text):
    characters = []
    positions = []
    for index, original in enumerate(text):
        for char in unicodedata.normalize("NFKC", original):
            if char.isalnum() or char == "-":
                characters.append(char)
                positions.append(index)
    return "".join(characters), positions


def slot_value(raw, values):
    values = list(dict.fromkeys(values))
    if len(values) == 1:
        return {"value": values[0], "raw": raw, "status": "ok"}
    return {"value": None, "raw": raw, "status": "ambiguous", "candidates": values}


def date_candidates(match, reference_date):
    phrase = match.group()
    if "-" in phrase:
        try:
            return [date.fromisoformat(phrase).isoformat()]
        except ValueError:
            return []
    if not reference_date:
        return []

    if phrase in ("今天", "明天", "后天"):
        return [(reference_date + timedelta(days=("今天", "明天", "后天").index(phrase))).isoformat()]
    day = WEEKDAY[match["day"]]
    if match["week"]:
        offset = 7 * WEEK_OFFSET[match["week"]] + day - reference_date.weekday()
        return [(reference_date + timedelta(days=offset)).isoformat()]
    offset = (day - reference_date.weekday()) % 7
    return [(reference_date + timedelta(days=offset + days)).isoformat() for days in (0, 7)]


class CModule:
    def __init__(self, *, retriever="tfidf", segmenter="builtin"):
        self.faq = FaqIndex(load_data("faq.json"), retriever=retriever, segmenter=segmenter)
        self.aliases = load_data("slots.json")
        self.slot_patterns = {
            field: re.compile("|".join(map(re.escape, sorted(aliases, key=len, reverse=True))))
            for field, aliases in self.aliases.items()
        }

    def _extract_slots(self, text, offset, original, positions, reference_date):
        def raw_phrase(matches):
            start = positions[offset + matches[0].start()]
            end = positions[offset + matches[-1].end() - 1]
            return original[start:end + 1]

        slots = {}
        matches = list(DATE_EXPRESSION.finditer(text))
        if matches:
            candidates = [date_candidates(match, reference_date) for match in matches]
            values = [value for options in candidates for value in options]
            raw = raw_phrase(matches)
            if all(candidates):
                slots["date"] = slot_value(raw, values)
            else:
                slots["date"] = {
                    "value": None, "raw": raw, "status": "ambiguous",
                    "candidates": list(dict.fromkeys(values)),
                }
        for field, pattern in self.slot_patterns.items():
            matches = list(pattern.finditer(text))
            if matches:
                values = [self.aliases[field][match.group()] for match in matches]
                slots[field] = slot_value(raw_phrase(matches), values)
        return slots

    def process(self, asr_final, turn_context=None, *, diagnostics=None):
        """处理单轮事件。可传入字典收集 FAQ 排名，用于本地回放日志。"""
        session_id = asr_final["session_id"]
        turn_id = asr_final["turn_id"]
        context = {}
        if turn_context and (
            turn_context.get("event") == "TURN_CONTEXT"
            and turn_context.get("session_id") == session_id
            and turn_context.get("turn_id") == turn_id
        ):
            context = turn_context["payload"]

        asr = asr_final["payload"]
        result = self._understand(asr, context, diagnostics)
        return make_event("NLU_RESULT", session_id, turn_id, result)

    def _understand(self, asr, context, diagnostics):
        result = {"intent": "unknown", "slots": {}, "status": "no_match"}
        if asr["status"] in ("error", "no_input"):
            result["status"] = asr["status"]
            return result

        original = asr.get("final_text") or ""
        text, positions = normalize(original)
        if not any(char.isalnum() for char in text):
            result["status"] = "no_input"
            return result

        correction = CORRECTION.search(text) or RESTATEMENT.match(text)
        if FAQ_QUESTION.search(text) or (
            YES_NO_QUESTION.search(text) and not (DIRECT_BOOK_REQUEST.match(text) or correction)
        ):
            faq_result, candidates = self.faq.search(original)
            if diagnostics is not None:
                name = "tfidf_cosine" if self.faq.retriever == "tfidf" else self.faq.retriever
                diagnostics.update({
                    "retriever": name, "segmenter": self.faq.segmenter, "candidates": candidates,
                })
            result.update(intent="faq", **faq_result)
            return result

        if CANCEL_REQUEST.fullmatch(text):
            result.update(intent="cancel", status="ok")
            return result

        if context.get("dialog_phase") == "confirming" or context.get("question_type") == "yes_no":
            if AFFIRMATION.fullmatch(text):
                result.update(intent="confirm", status="ok")
                return result
            if NEGATION.fullmatch(text):
                result.update(intent="deny", status="ok")
                return result

        if text.startswith(("不用改", "不改", "不要改", "我不想约", "我不想挂", "我不想登记")):
            return result
        offset = correction.end() if correction else 0
        reference_date = date.fromisoformat(context["reference_date"]) if context.get("reference_date") else None
        slots = self._extract_slots(text[offset:], offset, original, positions, reference_date)

        if any(phrase in text for phrase in ("上次那位", "上次的医生", "上次那个医生")):
            result["history_ref"] = "last_doctor"
        elif any(phrase in text for phrase in ("还是上次的", "跟上次一样", "和上次一样")):
            result["history_ref"] = "last_visit"
        has_details = bool(slots or result.get("history_ref"))
        expected_slots = context.get("expected_slots", [])

        if correction:
            result["intent"] = "correct"
        elif BOOK_REQUEST.search(text):
            result["intent"] = "book"
        elif (expected_slots and slots.keys() & set(expected_slots)) or result.get("history_ref"):
            result["intent"] = "provide_slot"
        elif has_details or expected_slots:
            result["status"] = "ambiguous"
            return result
        else:
            return result

        result["slots"] = slots
        result["status"] = "ambiguous" if (
            (correction and not has_details) or any(slot["status"] == "ambiguous" for slot in slots.values())
        ) else "ok"
        return result
