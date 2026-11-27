# -*- coding: utf-8 -*-
"""用例回放：把一段脚本化的用户输入喂给 D，并把 E 的播报结果自动回传。

一个用例就是一条 JSON：用户轮次（nlu）、档案选择（profile）、播报失败
（playback）、保存故障（save_fault）等步骤，最后给出预期登记结果。
回放会记录完整事件序列，便于核对“轮次、字段和值来源”。
"""
from __future__ import annotations

import json
import os

from . import events as ev
from . import nlu_sim
from .config import PolicyConfig
from .policy import DialogManager
from .store import RegistrationStore


class CaseRunner(object):
    def __init__(self, lexicon, profiles, faqs=None, base_config=None):
        self.lexicon = lexicon
        self.profiles = profiles
        self.faqs = faqs or []
        self.base_config = base_config or PolicyConfig()

    # ------------------------------------------------------------------
    def run(self, case, store=None, config_override=None, session_id=None,
            variant="improved"):
        config = self.base_config.with_overrides(case.get("config") or {})
        if config_override:
            config = config.with_overrides(config_override)
        reference_date = case.get("reference_date", "2026-11-24")
        session_id = session_id or case["case_id"]
        # 基线换一套用户说法：系统问法不同，用户的回答自然不同
        if variant == "baseline" and case.get("steps_baseline"):
            steps = case["steps_baseline"]
        else:
            steps = case.get("steps", [])

        own_store = store is None
        store = store or RegistrationStore(":memory:")
        before_ids = set(r["registration_id"] for r in store.list_registrations())

        transcript = []
        note_holder = {"note": None}

        def sink(env, note=None):
            current = note
            if current is None and env["event"] == ev.EVENT_NLU_RESULT:
                current = note_holder["note"]
            transcript.append({"event": env, "note": current})

        manager = DialogManager(
            store=store, lexicon=self.lexicon, profiles=self.profiles,
            config=config, reference_date=reference_date,
            event_sink=sink,
        )
        manager.start_session(session_id)
        state = manager.session(session_id)

        current_turn = 1
        ctx_env = self._latest_context(transcript)
        context = ctx_env["payload"] if ctx_env else None
        user_turns = 0
        playback_failures = 0

        for step in steps:
            kind = step.get("type", "nlu")
            if kind == "profile":
                env = ev.make_event(ev.EVENT_PROFILE_SUGGESTION, session_id, 0,
                                    {"profile_id": step.get("profile_id"),
                                     "match_status": step.get("match_status", "matched")})
                manager.handle(env)
                continue

            if kind == "save_fault":
                store.fail_next_save()
                continue

            if kind == "playback":
                # 手动回传一次播报结果（用于演示旧轮/重复完成）
                reply = state.last_reply
                if reply:
                    env = ev.make_event(ev.EVENT_PLAYBACK_RESULT, session_id,
                                        step.get("turn_id", current_turn),
                                        {"reply_id": step.get("reply_id") or reply["reply_id"],
                                         "status": step.get("status", "completed")})
                    manager.handle(env)
                continue

            if kind in ("nlu", "late_nlu"):
                text = step.get("text", "")
                payload = nlu_sim.parse(text, self.lexicon, reference_date, context,
                                        self.faqs, profile_id=state.profile_id)
                if step.get("payload"):
                    payload = step["payload"]
                turn = int(step["turn_id"]) if kind == "late_nlu" else current_turn
                if kind == "late_nlu" and turn >= current_turn:
                    turn = current_turn - 1
                env = ev.make_event(ev.EVENT_NLU_RESULT, session_id, turn, payload)
                note_holder["note"] = {"user_text": text}
                if kind == "nlu":
                    user_turns += 1
                out = manager.handle(env)
                note_holder["note"] = None
                # E 的播报：默认成功；用例可要求先失败一次
                want_fail = step.get("playback") == "failed" \
                    or (step.get("playback_fail_first") and playback_failures == 0)
                reply = self._reply_of(out)
                if reply is not None:
                    if want_fail:
                        playback_failures += 1
                        manager.handle(ev.make_event(
                            ev.EVENT_PLAYBACK_RESULT, session_id, turn,
                            {"reply_id": reply["payload"]["reply_id"], "status": "failed",
                             "error_code": "device_busy"}))
                        reply = state.last_reply["event"]
                    manager.handle(ev.make_event(
                        ev.EVENT_PLAYBACK_RESULT, session_id, turn,
                        {"reply_id": reply["payload"]["reply_id"], "status": "completed"}))
                if kind == "late_nlu":
                    continue
                ctx_env = self._latest_context(transcript)
                if ctx_env is not None and not state.closed:
                    context = ctx_env["payload"]
                    current_turn = int(ctx_env["turn_id"])
                else:
                    current_turn += 1

        rows = [r for r in store.list_registrations()
                if r["registration_id"] not in before_ids]
        result = self._summarize(case, session_id, transcript, rows, store,
                                 user_turns, len(rows), playback_failures)
        if own_store:
            store.close()
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _reply_of(out):
        for env in out:
            if env["event"] == ev.EVENT_REPLY_REQUEST:
                return env
        return None

    @staticmethod
    def _latest_context(transcript):
        for item in reversed(transcript):
            if item["event"]["event"] == ev.EVENT_TURN_CONTEXT:
                return item["event"]
        return None

    def _summarize(self, case, session_id, transcript, rows, store, user_turns,
                   new_rows, playback_failures):
        replies = []
        for item in transcript:
            env = item["event"]
            if env["event"] == ev.EVENT_REPLY_REQUEST:
                replies.append({
                    "turn": env["turn_id"],
                    "reply_id": env["payload"]["reply_id"],
                    "action": env["payload"].get("action"),
                    "text": env["payload"].get("reply_text"),
                })
        repeats = count_repeat_questions(replies)
        expect = case.get("expect") or {}
        checks = {}
        actual_fields = None
        if rows:
            row = rows[0]
            actual_fields = {
                "department": row.get("department"),
                "doctor": row.get("doctor"),
                "date": row.get("visit_date"),
                "period": row.get("period"),
            }
        checks["registrations"] = (len(rows) == int(expect.get("registrations", 1)))
        if expect.get("fields"):
            checks["fields"] = (actual_fields or {}) == expect["fields"]
        else:
            checks["fields"] = True
        checks["no_duplicate_rows"] = (store.has_duplicates() == [])

        success = all(checks.values())
        within_budget = True
        if expect.get("max_turns") is not None:
            within_budget = user_turns <= int(expect["max_turns"])
        return {
            "case_id": case["case_id"],
            "title": case.get("title", ""),
            "purpose": case.get("purpose", ""),
            "session_id": session_id,
            "reference_date": case.get("reference_date", "2026-11-24"),
            "user_turns": user_turns,
            "within_budget": within_budget,
            "repeat_questions": repeats,
            "replies": replies,
            "registrations": rows,
            "actual_fields": actual_fields,
            "expected": expect,
            "checks": checks,
            "success": bool(success),
            "new_rows": new_rows,
            "playback_failures": playback_failures,
            "transcript": transcript,
        }


def count_repeat_questions(replies):
    """同一字段被连续追问的次数（不含第一次）。

    例如“请问您想约哪一天？”连着问两次，记为 1 次重复询问。
    """
    repeats = 0
    last_key = None
    for reply in replies:
        action = reply.get("action") or ""
        if action.startswith("ask_") or action == "reprompt":
            key = action
            if key == last_key:
                repeats += 1
            last_key = key
        elif action not in ("restate_confirm",):
            last_key = None
    return repeats


def load_case(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    return data if isinstance(data, list) else [data]


def load_cases(directory):
    cases = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json") and name not in ("package.json",):
            path = os.path.join(directory, name)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("cases"):
                cases.extend(data["cases"])
            elif isinstance(data, list):
                cases.extend(data)
            else:
                cases.append(data)
    return cases
