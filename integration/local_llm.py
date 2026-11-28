"""Offline local small LLM layer (Qwen2.5-0.5B-Instruct GGUF) for C fallback & D NLG."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

VALID_INTENTS = {"faq", "book", "provide_slot", "correct", "confirm", "deny", "cancel", "unknown"}
ALLOWED_PERIODS = {"morning", "afternoon", "evening"}


def load_shared_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = path or (Path(__file__).resolve().parent / "shared_config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _build_lookup(shared_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    dept_map: Dict[str, str] = {}
    for item in shared_config.get("departments", []):
        dept_map[item["id"]] = item["id"]
        dept_map[item["name"]] = item["id"]
        for alias in item.get("aliases", []):
            dept_map[alias] = item["id"]

    doc_map: Dict[str, str] = {}
    for item in shared_config.get("doctors", []):
        doc_map[item["id"]] = item["id"]
        doc_map[item["name"]] = item["id"]
        for alias in item.get("aliases", []):
            doc_map[alias] = item["id"]

    period_map: Dict[str, str] = {}
    for pid, info in shared_config.get("periods", {}).items():
        period_map[pid] = pid
        period_map[info["name"]] = pid
        for alias in info.get("aliases", []):
            period_map[alias] = pid

    return {"department": dept_map, "doctor": doc_map, "period": period_map}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _clean_spoken_reply(text: str, max_chars: int = 55) -> str:
    cleaned = re.sub(r"[*_`#>\-\[\]()（）]", "", text or "").strip()
    cleaned = re.sub(r"\s+", "，", cleaned)
    cleaned = re.sub(r"[，,]+", "，", cleaned).strip("，, ")
    if len(cleaned) > max_chars:
        cut = cleaned[:max_chars]
        for end_mark in ("。", "？", "！", "，"):
            idx = cut.rfind(end_mark)
            if idx >= 12:
                return cut[: idx + 1]
        return cut + "。"
    if cleaned and cleaned[-1] not in "。？！?!":
        cleaned += "。"
    return cleaned


class LocalLLM:
    """Offline GGUF wrapper with strict whitelist & factual verification guards."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        *,
        enabled: bool = True,
        n_ctx: int = 1024,
        n_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.enabled = bool(enabled)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None
        self.shared_config = load_shared_config()
        self._lookup = _build_lookup(self.shared_config)

    def is_available(self) -> bool:
        if not self.enabled or not self.model_path.is_file():
            return False
        if self._llm is not None:
            return True
        try:
            self._ensure_loaded()
            return self._llm is not None
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,
            verbose=False,
        )

    def _chat(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 96) -> str:
        self._ensure_loaded()
        assert self._llm is not None
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        return str(out["choices"][0]["message"]["content"]).strip()

    def extract_nlu_fallback(
        self,
        utterance: str,
        turn_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract strictly validated intent/slots from natural colloquial Mandarin."""
        if not self.is_available() or not (utterance or "").strip():
            return None

        ctx = (turn_context or {}).get("payload", turn_context or {})
        ref_date = str(ctx.get("reference_date") or datetime.date.today().isoformat())
        phase = str(ctx.get("dialog_phase") or "idle")
        expected = list(ctx.get("expected_slots") or [])
        try:
            ref_dt = datetime.date.fromisoformat(ref_date)
        except ValueError:
            ref_dt = datetime.date.today()
            ref_date = ref_dt.isoformat()

        tomorrow = (ref_dt + datetime.timedelta(days=1)).isoformat()
        day_after = (ref_dt + datetime.timedelta(days=2)).isoformat()

        dept_desc = ", ".join(f"{d['name']}({d['id']})" for d in self.shared_config["departments"])
        doc_desc = ", ".join(f"{d['name']}({d['id']})" for d in self.shared_config["doctors"])
        system = (
            "你是医院复诊登记系统的语义解析器。只输出一行合法JSON对象，不要任何解释。\n"
            f"允许的intent: book, provide_slot, correct, confirm, deny, cancel, faq, unknown。\n"
            f"允许的department: {dept_desc}。\n"
            f"允许的doctor: {doc_desc}（用户没提医生姓氏时必须填null，严禁编造）。\n"
            "允许的period: morning(上午), afternoon(下午), evening(晚上)。\n"
            f"日期锚点: 今天={ref_date}, 明天={tomorrow}, 后天={day_after}。date输出YYYY-MM-DD或null。\n"
            "没明确提到某字段必须设为null。\n"
            "输出格式: {\"intent\":\"...\",\"department\":null,\"doctor\":null,\"date\":null,\"period\":null,\"history_ref\":null}"
        )
        user = (
            f"当前阶段:{phase}; 期望字段:{expected}; 用户说:{utterance}"
        )

        try:
            raw = self._chat(system, user, temperature=0.05, max_tokens=96)
        except Exception:
            return None

        parsed = _extract_first_json_object(raw)
        if not parsed:
            return None

        intent = str(parsed.get("intent") or "unknown").strip()
        if intent not in VALID_INTENTS:
            intent = "unknown"

        slots: Dict[str, Dict[str, Any]] = {}
        dep_raw = parsed.get("department")
        if isinstance(dep_raw, str) and dep_raw.strip() in self._lookup["department"]:
            dep_id = self._lookup["department"][dep_raw.strip()]
            slots["department"] = {"value": dep_id, "raw": dep_raw.strip(), "status": "ok"}

        doc_raw = parsed.get("doctor")
        if isinstance(doc_raw, str) and doc_raw.strip() in self._lookup["doctor"]:
            doc_id = self._lookup["doctor"][doc_raw.strip()]
            # Grounding guard: reject hallucinated doctor IDs unless user uttered surname/alias
            doc_meta = next((d for d in self.shared_config["doctors"] if d["id"] == doc_id), None)
            if doc_meta is not None:
                cues = [doc_meta["name"][0], doc_meta["name"], *doc_meta.get("aliases", [])]
                if any(cue in utterance for cue in cues):
                    slots["doctor"] = {"value": doc_id, "raw": doc_raw.strip(), "status": "ok"}

        per_raw = parsed.get("period")
        if isinstance(per_raw, str) and per_raw.strip() in self._lookup["period"]:
            per_id = self._lookup["period"][per_raw.strip()]
            slots["period"] = {"value": per_id, "raw": per_raw.strip(), "status": "ok"}

        date_raw = parsed.get("date")
        if isinstance(date_raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_raw.strip()):
            try:
                dt = datetime.date.fromisoformat(date_raw.strip())
                if -7 <= (dt - ref_dt).days <= 366:
                    slots["date"] = {"value": dt.isoformat(), "raw": utterance.strip(), "status": "ok"}
            except ValueError:
                pass

        history_ref = parsed.get("history_ref")
        if history_ref not in ("last_doctor", "last_visit"):
            history_ref = None

        if intent in ("confirm", "deny", "cancel"):
            return {"intent": intent, "slots": {}, "status": "ok"}
        if slots or history_ref:
            final_intent = intent if intent in ("book", "provide_slot", "correct") else (
                "provide_slot" if expected else "book"
            )
            payload: Dict[str, Any] = {"intent": final_intent, "slots": slots, "status": "ok"}
            if history_ref:
                payload["history_ref"] = history_ref
            return payload
        return None

    def answer_open_query(self, utterance: str, *, has_active_draft: bool = False) -> str:
        """Answer out-of-FAQ guidance or everyday elderly chat warmly and concisely."""
        dept_names = "、".join(d["name"] for d in self.shared_config["departments"])
        fallback = (
            f"这是模拟复诊登记服务，目前支持{dept_names}。"
            + ("您随时可以继续刚才的登记。" if has_active_draft else "需要登记复诊请直接告诉我科室、日期和时段。")
        )
        if not self.is_available():
            return fallback

        system = (
            "你是医院门诊一楼导诊台护士，正在用温暖亲切的口语耐心回答一位老年人。"
            f"医院目前设有的科室为：{dept_names}。"
            "请直接说一句自然温柔的话（20到40个字以内，绝不要说'推荐一个最相关的'、不要列编号或英文）。"
            "如果是症状咨询挂号，直接告诉老人建议挂哪个科室，并温和补充仅供参考。"
        )
        try:
            raw = self._chat(system, f"老人问：{utterance}", temperature=0.35, max_tokens=72)
            cleaned = _clean_spoken_reply(raw, max_chars=48)
            if len(cleaned) >= 6 and "最相关" not in cleaned and "提示" not in cleaned:
                return cleaned
        except Exception:
            pass
        return fallback

    def polish_reply(
        self,
        action: str,
        raw_reply_text: str,
        required_facts: Optional[List[str]] = None,
    ) -> str:
        """Rewrite template reply smoothly while strictly preserving all business facts."""
        if not self.is_available() or not raw_reply_text:
            return raw_reply_text
        facts = [f for f in (required_facts or []) if f]
        if action in ("session_closed", "exit_after_reprompt", "cancel_ack"):
            return raw_reply_text

        system = (
            "你是服务老年人的语音助手，请把给定回复改写得更亲切自然、简短易听（纯中文口语，不超过45字，无英文或符号列表）。"
            "重要规则：提示里列出的每一个关键信息词必须一字不差地保留在输出中，不能改动日期、科室、医生姓名或编号。"
        )
        user = f"原回复：{raw_reply_text}\n必须一字不差保留的关键信息：{facts}"
        try:
            rewritten = _clean_spoken_reply(self._chat(system, user, temperature=0.25, max_tokens=80), max_chars=65)
        except Exception:
            return raw_reply_text

        # Hard factuality guard: any dropped fact or too-short/too-long rewrite falls back immediately
        for fact in facts:
            if fact not in rewritten:
                return raw_reply_text
        if len(rewritten) < 4 or len(rewritten) > max(len(raw_reply_text) + 25, 65):
            return raw_reply_text
        if action in ("restate_confirm", "faq_resume_confirm", "retry_save") and not rewritten.endswith(("吗？", "吗。", "行吗？", "好吗？", "？")):
            return raw_reply_text
        return rewritten
