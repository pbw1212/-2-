# -*- coding: utf-8 -*-
"""多轮对话策略：状态机、补问选择、复述确认、历史继承与本地保存。

D 的职责是“决定下一轮做什么”：接收 C 的结构化理解结果，维护登记草稿，
输出回复文本给 E，并把下一轮的阶段、问题类型和期望字段送给 A/B/C。
"""
from __future__ import annotations

import datetime

from . import events as ev
from .config import PERIOD_DISPLAY, SLOT_NAMES, SLOT_ORDER
from .draft import AMBIGUOUS, CANDIDATE, NEEDS_CHECK, OK, SKIPPED, Draft


def _md(iso_date):
    """把 2026-12-03 渲染成 12月3日，便于播报与复述。"""
    try:
        d = datetime.datetime.strptime(iso_date, "%Y-%m-%d")
        return "%d月%d日" % (d.month, d.day)
    except (TypeError, ValueError):
        return iso_date or ""


class SessionState(object):
    """一个会话内的全部可变状态。"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.draft = None
        self.profile_id = None
        self.profile_status = None
        self.closed = False
        self.seq = 0
        self.seen_turns = set()
        self.last_reply = None          # 最近一次 REPLY_REQUEST 的记录
        self.reply_seq = 0
        self.reprompt_count = 0
        self.play_fail_count = 0
        self.pending = None             # 待恢复的问题：(action, text, qtype, slots, field)


class DialogManager(object):
    """登记对话管理器。不依赖具体音频设备，只处理事件。"""

    def __init__(self, store, lexicon, profiles, config, reference_date="2026-11-24",
                 event_sink=None):
        self.store = store
        self.lexicon = lexicon
        self.profiles = profiles
        self.config = config
        self.reference_date = reference_date
        self.sessions = {}
        self.event_sink = event_sink or (lambda env, note=None: None)

    # ================= 会话与事件入口 =================
    def session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        return self.sessions[session_id]

    def start_session(self, session_id):
        """会话开始：发第一轮上下文（待机、开放式、无期望字段）。"""
        state = self.session(session_id)
        ctx = self._turn_context(state, 1, "idle", "open", [],
                                 "请先告诉我您要复诊的科室、日期和时段。")
        self._log(state, 0, "idle", "session_start", {"reference_date": self.reference_date})
        return [ctx]

    def handle(self, env):
        """处理一个输入事件，返回需要发出的输出事件列表。"""
        env = ev.validate_envelope(env)
        name = env["event"]
        if name == ev.EVENT_NLU_RESULT:
            return self._on_nlu(env)
        if name == ev.EVENT_PROFILE_SUGGESTION:
            return self._on_profile(env)
        if name == ev.EVENT_PLAYBACK_RESULT:
            return self._on_playback(env)
        self._log(self.session(env["session_id"]), env["turn_id"], "-", "ignore_event",
                  {"event": name, "reason": "unsupported"})
        return []

    # ================= C -> D =================
    def _on_nlu(self, env):
        state = self.session(env["session_id"])
        turn = env["turn_id"]
        self._emit(env)

        if turn in state.seen_turns:
            self._log(state, turn, "-", "ignore_turn", {"reason": "duplicate_turn"})
            return []
        if turn < self._last_turn(state):
            self._log(state, turn, "-", "ignore_turn", {"reason": "late_turn"})
            return []
        state.seen_turns.add(turn)

        payload = env["payload"]
        try:
            intent, slots, status = ev.validate_nlu_payload(payload)
        except ev.SchemaError as exc:
            self._log(state, turn, "-", "bad_payload", {"error": str(exc)})
            return self._reprompt(state, turn, "抱歉，我没听懂，请再说一次。")

        # 会话已结束（例如保存后又来一条确认）
        if state.closed:
            return self._closed_reply(state, turn, intent)

        if status in ("no_input", "error"):
            return self._reprompt(state, turn)

        if intent == "faq":
            return self._answer_faq(state, turn, payload)

        if intent == "cancel":
            return self._cancel(state, turn)

        if intent in ("book", "provide_slot", "correct"):
            return self._collect(state, turn, intent, slots, payload)

        if intent == "confirm":
            return self._confirm(state, turn)

        if intent == "deny":
            return self._deny(state, turn)

        # unknown / no_match
        return self._reprompt(state, turn)

    # ================= A -> D =================
    def _on_profile(self, env):
        state = self.session(env["session_id"])
        payload = env["payload"] or {}
        profile_id = payload.get("profile_id")
        state.profile_status = payload.get("match_status") or payload.get("status")
        self._emit(env)
        if not profile_id or self.profiles.get(profile_id) is None:
            self._log(state, env["turn_id"], "-", "profile_rejected",
                      {"profile_id": profile_id, "reason": "unknown_profile"})
            return []
        state.profile_id = profile_id
        if state.draft is not None and self.config.inherit_history:
            if self._apply_history(state.draft, state, env["turn_id"], explicit=False):
                state.draft.mark_revision()
        self._log(state, env["turn_id"], "-", "profile_selected",
                  {"profile_id": profile_id, "match_status": state.profile_status})
        return []

    # ================= E -> D =================
    def _on_playback(self, env):
        state = self.session(env["session_id"])
        payload = env["payload"] or {}
        reply_id = payload.get("reply_id")
        status = payload.get("status")
        self._emit(env)
        reply = state.last_reply
        if not reply or reply["reply_id"] != reply_id:
            self._log(state, env["turn_id"], "-", "ignore_playback",
                      {"reply_id": reply_id, "reason": "stale_reply_id"})
            return []

        if status == "completed":
            state.play_fail_count = 0
            if state.draft and state.draft.snapshot \
                    and state.draft.snapshot["reply_id"] == reply_id:
                state.draft.snapshot["playback_done"] = True
            self._log(state, env["turn_id"], state.draft.phase if state.draft else "idle",
                      "playback_completed", {"reply_id": reply_id})
            return []

        # 播报失败：用新的 reply_id 重发同一段文本，确认快照改绑新回复
        state.play_fail_count += 1
        if state.play_fail_count > self.config.max_reprompt:
            self._log(state, env["turn_id"], "-", "playback_failed_give_up",
                      {"reply_id": reply_id})
            return []
        new_reply = self._reissue_reply(state, reply, env["turn_id"])
        self._log(state, env["turn_id"], state.draft.phase if state.draft else "idle",
                  "playback_retry", {"old_reply_id": reply_id,
                                     "new_reply_id": new_reply["reply_id"]})
        return [new_reply["event"]]

    # ================= 对话流程 =================
    def _collect(self, state, turn, intent, slots, payload):
        draft = state.draft
        created = False
        if draft is None:
            state.seq += 1
            reg_id = self.store.next_registration_id()
            draft = Draft(state.session_id, reg_id, created_turn=turn)
            draft.profile_id = state.profile_id
            state.draft = draft
            created = True
            self._log(state, turn, "idle", "draft_created",
                      {"registration_id": reg_id, "intent": intent,
                       "profile_id": draft.profile_id})

        changed = False
        for field in SLOT_ORDER:
            if field in slots:
                if draft.apply_slot(field, slots[field], turn, self.lexicon,
                                    check_dependency=self.config.check_slot_dependency):
                    changed = True

        history_ref = payload.get("history_ref")
        # 新建草稿时先带出历史候选（用户本轮明确给出的字段优先）
        if created and self.config.inherit_history and draft.profile_id and not history_ref:
            if self._apply_history(draft, state, turn, explicit=False):
                changed = True
        # “还是上次那位”：只在明确引用历史且未同时给出科室/医生时生效
        if history_ref and self.config.inherit_history \
                and not ("department" in slots or "doctor" in slots):
            if self._apply_history(draft, state, turn, explicit=True):
                changed = True

        if changed:
            draft.mark_revision()
            state.reprompt_count = 0
            state.pending = None
            self._log(state, turn, draft.phase, "slots_updated",
                      {"slots": slots, "revision": draft.revision,
                       "history_ref": history_ref})

        return self._advance(state, turn)

    def _confirm(self, state, turn):
        draft = state.draft
        if draft is None:
            return self._reprompt(state, turn, "我们还没有开始登记，请问您要复诊哪个科室？")

        ok, reason = draft.snapshot_valid_for(
            turn, require_playback=self.config.confirm_requires_playback)
        if not ok:
            self._log(state, turn, draft.phase, "confirm_rejected",
                      {"reason": reason, "revision": draft.revision})
            # 版本已变或轮次不匹配：按当前草稿重新复述，等新轮次确认
            return self._advance(state, turn)

        return self._save(state, turn)

    def _deny(self, state, turn):
        draft = state.draft
        if draft is None:
            return self._reprompt(state, turn)
        # 补问阶段说“不用”：若上一句是在问医生，视为明确不指定
        if state.last_reply and state.last_reply.get("field") == "doctor":
            draft.slots["doctor"].status = SKIPPED
            draft.slots["doctor"].note = "用户明确不指定医生"
            draft.mark_revision()
            self._log(state, turn, draft.phase, "doctor_skipped", {})
            return self._advance(state, turn)
        if draft.phase == "confirming":
            self._log(state, turn, draft.phase, "correction_requested", {})
            return self._reply(state, turn, draft, state.session_id,
                               "好的，请问需要修改哪一项？科室、日期、时段还是医生？",
                               action="ask_correction", question_type="targeted",
                               expected_slots=["department", "date", "period", "doctor"])
        return self._reprompt(state, turn)

    def _cancel(self, state, turn):
        draft = state.draft
        if draft is None:
            state.closed = True
            return self._reply(state, turn, draft, state.session_id,
                               "好的，没有进行中的登记。", action="cancel_ack",
                               question_type="open", expected_slots=[], end=True)
        draft.phase = "ended"
        draft.ended_reason = "user_cancel"
        draft.snapshot = None
        state.closed = True
        self._log(state, turn, draft.phase, "cancelled",
                  {"registration_id": draft.registration_id,
                   "registrations": self.store.count_registrations()})
        return self._reply(state, turn, draft, state.session_id,
                           "好的，本次登记已取消，没有为您保存。", action="cancel_ack",
                           question_type="open", expected_slots=[], end=True)

    def _answer_faq(self, state, turn, payload):
        answer = payload.get("answer_text") or "这个问题我暂时没有答案，您可以到一楼服务台咨询。"
        draft = state.draft
        if draft is None:
            text = answer + "如果需要登记复诊，请告诉我。"
            return self._reply(state, turn, None, state.session_id, text,
                               action="faq_answer", question_type="open",
                               expected_slots=[])
        if draft.phase == "confirming":
            # 咨询后接回原来的确认动作，草稿版本不变
            text = answer + " 我们继续刚才的登记。" + self._restate_text(draft)
            return self._reply(state, turn, draft, state.session_id, text,
                               action="faq_resume_confirm", question_type="yes_no",
                               expected_slots=[], confirm=True)
        action, qtext, qtype, expected, field = self._question(draft)
        text = answer + " 我们继续刚才的登记。" + qtext
        return self._reply(state, turn, draft, state.session_id, text,
                           action="faq_resume_ask", question_type=qtype,
                           expected_slots=expected, field=field)

    # ================= 推进 =================
    def _advance(self, state, turn):
        draft = state.draft
        if draft.ready_to_confirm():
            return self._restate(state, turn)
        action, text, qtype, expected, field = self._question(draft)
        return self._reply(state, turn, draft, state.session_id, text,
                           action=action, question_type=qtype,
                           expected_slots=expected, field=field)

    def _restate(self, state, turn):
        draft = state.draft
        draft.phase = "confirming"
        return self._reply(state, turn, draft, state.session_id,
                           self._restate_text(draft), action="restate_confirm",
                           question_type="yes_no", expected_slots=[], confirm=True)

    def _save(self, state, turn):
        draft = state.draft
        fields = draft.to_confirm_fields()
        department = fields["department"]["value"]
        date = fields["date"]["value"]
        period = fields["period"]["value"]
        doctor = fields.get("doctor", {}).get("value")
        try:
            reg_id, created = self.store.save_registration(
                draft, department, date, period, doctor)
        except Exception as exc:  # SaveError
            draft.phase = "confirming"
            draft.save_status = "failed"
            self._log(state, turn, draft.phase, "save_failed",
                      {"error": str(exc), "registration_id": draft.registration_id})
            return self._reply(state, turn, draft, state.session_id,
                               "保存没有成功，刚才的信息我都留着。请再说一次“确认”，我再试一次。",
                               action="retry_save", question_type="yes_no",
                               expected_slots=[], confirm=True)

        draft.save_status = "saved"
        draft.phase = "ended"
        state.closed = True
        self._log(state, turn, "ended", "saved" if created else "duplicate_confirm",
                  {"registration_id": reg_id, "created": created,
                   "fields": dict((k, v["value"]) for k, v in fields.items())})
        if created:
            text = "已为您登记：%s%s，%s%s。登记编号 %s。" % (
                self.lexicon.department_name(department),
                self.lexicon.doctor_name(doctor) if doctor else "",
                _md(date), PERIOD_DISPLAY.get(period, period), reg_id)
            action = "saved"
        else:
            text = "这次登记已经保存过了，登记编号 %s，不需要重复登记。" % reg_id
            action = "duplicate_ack"
        return self._reply(state, turn, draft, state.session_id, text,
                           action=action, question_type="open", expected_slots=[], end=True)

    def _closed_reply(self, state, turn, intent):
        draft = state.draft
        reg_id = draft.registration_id if draft else "无"
        if intent in ("confirm", "book"):
            text = "这次登记已经保存过了，登记编号 %s。" % reg_id
        else:
            text = "本次会话已经结束。需要新的登记请重新开始。"
        return self._reply(state, turn, draft, state.session_id, text,
                           action="session_closed", question_type="open",
                           expected_slots=[], end=True)

    def _reprompt(self, state, turn, text=None):
        draft = state.draft
        state.reprompt_count += 1
        limit = self.config.max_reprompt
        if state.reprompt_count > limit:
            if draft:
                draft.phase = "ended"
                draft.ended_reason = "too_many_reprompts"
            state.closed = True
            self._log(state, turn, draft.phase if draft else "idle", "reprompt_exhausted",
                      {"count": state.reprompt_count, "limit": limit})
            return self._reply(state, turn, draft, state.session_id,
                               "我先结束这次登记，您需要的时候再叫我。",
                               action="exit_after_reprompt", question_type="open",
                               expected_slots=[], end=True)
        if text is None:
            if draft is not None and draft.ready_to_confirm():
                # 确认轮没听清：直接再把复述说一遍
                return self._restate(state, turn)
            if draft is not None and not draft.ready_to_confirm():
                action, qtext, qtype, expected, field = self._question(draft)
                return self._reply(state, turn, draft, state.session_id,
                                   "我没有听清，请您再说一遍。" + qtext,
                                   action="reprompt", question_type=qtype,
                                   expected_slots=expected, field=field)
            text = "我没有听清，请您再说一遍。"
        self._log(state, turn, draft.phase if draft else "idle", "reprompt",
                  {"count": state.reprompt_count})
        return self._reply(state, turn, draft, state.session_id, text,
                           action="reprompt", question_type="open", expected_slots=[])

    # ================= 问题与话术 =================
    def _question(self, draft):
        """按固定顺序找下一个需要问的字段，返回 (action, text, qtype, slots, field)。"""
        field, reason = self._pending_slot(draft)
        if field is None:
            # 理论上不会走到这里；兜底为开放式追问
            return ("ask_open", "请告诉我复诊的科室、日期和时段。", "open",
                    list(SLOT_ORDER), None)

        if field == "department":
            text = "请问您要复诊哪个科室？"
            qtype = "targeted" if self.config.clarify_ambiguous else "open"
        elif field == "date":
            if reason == "ambiguous" and self.config.clarify_ambiguous:
                text = self._clarify_date_text(draft)
                qtype = "targeted"
            else:
                text = "请问您想约哪一天？可以说几月几号，也可以说下周几。"
                qtype = "open"
        elif field == "period":
            text = "请问上午还是下午方便？"
            qtype = "targeted"
        elif field == "doctor":
            dept = self.lexicon.department_name(draft.value("department"))
            if reason == "check":
                text = "换到%s后，原来的医生不再适用。请问还要指定医生吗？可以说不指定。" % dept
            else:
                text = "请问要指定医生吗？可以说医生姓名，也可以说不指定。"
            qtype = "targeted"
        else:
            text = "请问%s？" % SLOT_NAMES.get(field, field)
            qtype = "targeted"
        return ("ask_%s" % field, text, qtype, [field], field)

    def _pending_slot(self, draft):
        for field in SLOT_ORDER:
            slot = draft.slots[field]
            if slot.required and not slot.present:
                return field, "missing"
            if slot.status == AMBIGUOUS:
                return field, "ambiguous"
            if slot.status == NEEDS_CHECK:
                return field, "check"
        return None, None

    def _clarify_date_text(self, draft):
        slot = draft.slots["date"]
        parts = []
        for item in slot.candidates[:2]:
            value = item.get("value")
            label = item.get("label") or item.get("raw") or "候选"
            parts.append("%s（%s）" % (label, _md(value)))
        if len(parts) == 2:
            return "您说的是哪一周的周五？%s还是%s？" % (parts[0], parts[1])
        return "请问您说的是具体哪一天？"

    def _restate_text(self, draft):
        dept = self.lexicon.department_name(draft.value("department"))
        doctor = draft.value("doctor")
        doctor_txt = self.lexicon.doctor_name(doctor) if doctor else ""
        date = _md(draft.value("date"))
        period = PERIOD_DISPLAY.get(draft.value("period"), draft.value("period") or "")
        return "为您登记复诊：%s%s，%s%s，对吗？" % (dept, doctor_txt, date, period)

    # ================= 历史继承 =================
    def _apply_history(self, draft, state, turn, explicit=False):
        """把档案的上次科室/医生写入草稿，标记来源；返回是否写入。

        explicit=True 表示用户说“还是上次那位”，只有档案能给出唯一且相符的
        科室与医生时才整体采用，否则留给后续按字段提问。
        """
        profile_id = state.profile_id or draft.profile_id
        profile = self.profiles.get(profile_id)
        if profile is None:
            self._log(state, turn, draft.phase, "history_unavailable",
                      {"reason": "no_profile"})
            return False
        if profile_id and not draft.profile_id:
            draft.profile_id = profile_id

        dep = profile.get("last_department")
        doc = profile.get("last_doctor")
        changed = False

        if explicit:
            if not self.profiles.has_unique_doctor(profile_id):
                self._log(state, turn, draft.phase, "history_ambiguous",
                          {"reason": "doctor_not_unique", "profile_id": profile_id})
                return False
            for field, value in (("department", dep), ("doctor", doc)):
                if not value:
                    continue
                slot = draft.slots[field]
                if slot.present and slot.source == "user":
                    continue
                slot.value = value
                slot.raw = "上次就诊记录"
                slot.source = "history_ref"
                slot.status = OK
                slot.updated_turn = turn
                slot.note = "用户引用上次记录"
                changed = True
            self._log(state, turn, draft.phase, "history_referenced",
                      {"profile_id": profile_id, "department": dep, "doctor": doc})
            return changed

        style = CANDIDATE if self.config.history_confirm_style == "candidate" else OK
        if dep and not draft.slots["department"].present:
            slot = draft.slots["department"]
            slot.value, slot.raw = dep, "上次就诊科室"
            slot.source, slot.status = "history", style
            slot.updated_turn = turn
            slot.note = "历史候选，随本次确认"
            changed = True
        if doc and not draft.slots["doctor"].present:
            slot = draft.slots["doctor"]
            slot.value, slot.raw = doc, "上次就诊医生"
            slot.source, slot.status = "history", style
            slot.updated_turn = turn
            slot.note = "历史候选，随本次确认"
            changed = True
        if changed:
            self._log(state, turn, draft.phase, "history_inherited",
                      {"profile_id": profile_id, "department": dep, "doctor": doc,
                       "status": style})
        return changed

    # ================= 输出 =================
    def _reply(self, state, turn, draft, session_id, text, action, question_type,
               expected_slots, field=None, confirm=False, end=False):
        state.reply_seq += 1
        reply_id = "r-%s-%02d-%02d" % (session_id, turn, state.reply_seq)
        if draft is not None and (action.startswith("ask_") or action == "reprompt"):
            # 回到补问：阶段回到字段收集
            draft.phase = "collecting"
        phase = draft.phase if draft else "idle"
        if phase == "ended":
            phase = "idle"
        payload = {"reply_id": reply_id, "reply_text": text, "action": action,
                   "dialog_phase": phase, "turn_id": turn}
        reply_env = ev.make_event(ev.EVENT_REPLY_REQUEST, session_id, turn, payload)
        record = {"reply_id": reply_id, "text": text, "action": action,
                  "question_type": question_type, "expected_slots": list(expected_slots),
                  "field": field, "turn": turn, "phase": phase, "event": reply_env}
        state.last_reply = record
        if draft is not None:
            draft.last_action = action
        if confirm and draft is not None:
            # 确认回复绑定草稿版本与预期确认轮次，播完后才可保存
            draft.set_snapshot(reply_id, confirm_turn=turn + 1, playback_done=False)
            draft.phase = "confirming"
        self._emit(reply_env)
        self._log(state, turn, phase, action,
                  {"reply_id": reply_id, "text": text,
                   "question_type": question_type, "expected_slots": list(expected_slots),
                   "revision": draft.revision if draft else None})
        out = [reply_env]
        if not end:
            out.append(self._turn_context(state, turn + 1, phase, question_type,
                                          expected_slots, text, field))
        return out

    def _reissue_reply(self, state, reply, turn):
        """播报失败后重发：同一文本、新的 reply_id，确认快照改绑新回复。"""
        state.reply_seq += 1
        reply_id = "r-%s-%02d-%02d" % (state.session_id, turn, state.reply_seq)
        draft = state.draft
        payload = {"reply_id": reply_id, "reply_text": reply["text"],
                   "action": "retry_playback", "dialog_phase": reply["phase"],
                   "turn_id": turn}
        env = ev.make_event(ev.EVENT_REPLY_REQUEST, state.session_id, turn, payload)
        record = dict(reply)
        record.update({"reply_id": reply_id, "action": "retry_playback", "event": env,
                       "turn": turn})
        state.last_reply = record
        if draft is not None and draft.snapshot is not None:
            draft.snapshot["reply_id"] = reply_id
            draft.snapshot["playback_done"] = False
        self._emit(env)
        return record

    def _turn_context(self, state, turn, phase, question_type, expected_slots,
                      text=None, field=None):
        expected = list(expected_slots or [])
        single = expected[0] if len(expected) == 1 else None
        payload = {
            "dialog_phase": phase,
            "question_type": question_type,
            "expected_slots": expected,
            "expected_slot": single,
            "reference_date": self.reference_date,
            "question_text": text or "",
        }
        if state.draft is not None:
            payload["registration_id"] = state.draft.registration_id
            payload["draft_revision"] = state.draft.revision
        if state.profile_id:
            payload["profile_id"] = state.profile_id
        if field:
            payload["pending_field"] = field
        ctx = ev.make_event(ev.EVENT_TURN_CONTEXT, state.session_id, turn, payload)
        self._emit(ctx)
        return ctx

    # ================= 工具 =================
    def _last_turn(self, state):
        return max(state.seen_turns) if state.seen_turns else -1

    def _emit(self, env, note=None):
        try:
            self.event_sink(env, note)
        except Exception:
            pass

    def _log(self, state, turn, phase, action, detail=None):
        self.store.log_turn(state.session_id, turn, phase, action, detail or {})
