# -*- coding: utf-8 -*-
"""D 模块单元测试与集成测试（标准库 unittest，无需第三方依赖）。

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from dmodule import events as ev  # noqa: E402
from dmodule import nlu_sim  # noqa: E402
from dmodule.config import PolicyConfig, default_path, load_lexicon, load_profiles  # noqa: E402
from dmodule.draft import Draft, NEEDS_CHECK, OK  # noqa: E402
from dmodule.policy import DialogManager  # noqa: E402
from dmodule.replay import CaseRunner, load_cases  # noqa: E402
from dmodule.evaluation import run_suite  # noqa: E402
from dmodule.store import RegistrationStore  # noqa: E402


def make_env(lexicon, profiles, faqs, config=None, reference="2026-11-24"):
    store = RegistrationStore(":memory:")
    manager = DialogManager(store, lexicon, profiles, config or PolicyConfig(),
                            reference)
    return store, manager


class EnvelopeTest(unittest.TestCase):
    def test_common_fields(self):
        env = ev.make_event(ev.EVENT_NLU_RESULT, "s1", 1,
                            ev.nlu_payload("book", {}, "ok"))
        self.assertEqual(env["schema_version"], 1)
        self.assertEqual(env["turn_id"], 1)
        ev.validate_envelope(env)

    def test_reject_bad_payload(self):
        with self.assertRaises(ev.SchemaError):
            ev.validate_nlu_payload({"intent": "not_an_intent", "slots": {}, "status": "ok"})

    def test_ambiguous_slot_must_have_null_value(self):
        with self.assertRaises(ev.SchemaError):
            ev.slot_payload("2026-12-03", "周五", "ambiguous")


class DraftTest(unittest.TestCase):
    def setUp(self):
        self.lexicon = load_lexicon(default_path("config", "lexicon.json"))

    def test_department_change_clears_doctor(self):
        draft = Draft("s1", "R1")
        draft.apply_slot("department", {"value": "rehab", "raw": "康复科", "status": "ok"},
                         1, self.lexicon)
        draft.apply_slot("doctor", {"value": "d_rehab_zhang", "raw": "张医生", "status": "ok"},
                         2, self.lexicon)
        changed = draft.apply_slot("department",
                                   {"value": "ortho", "raw": "骨科", "status": "ok"},
                                   3, self.lexicon)
        self.assertTrue(changed)
        self.assertIsNone(draft.value("doctor"))
        self.assertEqual(draft.get("doctor").status, NEEDS_CHECK)

    def test_snapshot_invalidated_by_correction(self):
        draft = Draft("s1", "R1")
        draft.slots["department"].value = "rehab"
        draft.slots["department"].status = OK
        draft.mark_revision()
        draft.set_snapshot("r1", confirm_turn=2, playback_done=True)
        self.assertTrue(draft.snapshot_valid_for(2)[0])
        draft.apply_slot("date", {"value": "2026-12-04", "raw": "下周五", "status": "ok"},
                         2, self.lexicon)
        draft.mark_revision()
        ok, reason = draft.snapshot_valid_for(2)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_snapshot")

    def test_playback_must_finish_before_confirm(self):
        draft = Draft("s1", "R1")
        draft.slots["department"].value = "rehab"
        draft.slots["department"].status = OK
        draft.mark_revision()
        draft.set_snapshot("r1", confirm_turn=2, playback_done=False)
        ok, reason = draft.snapshot_valid_for(2)
        self.assertFalse(ok)
        self.assertEqual(reason, "playback_pending")


class StoreTest(unittest.TestCase):
    def test_same_registration_id_written_once(self):
        store = RegistrationStore(":memory:")
        draft = Draft("s1", "R-1")
        draft.slots["department"].value = "rehab"
        draft.slots["doctor"].value = None
        draft.slots["date"].value = "2026-12-03"
        draft.slots["period"].value = "afternoon"
        first = store.save_registration(draft, "rehab", "2026-12-03", "afternoon", None)
        second = store.save_registration(draft, "rehab", "2026-12-03", "afternoon", None)
        self.assertEqual(first, ("R-1", True))
        self.assertEqual(second, ("R-1", False))
        self.assertEqual(store.count_registrations(), 1)
        self.assertEqual(store.has_duplicates(), [])


class DialogueTest(unittest.TestCase):
    def setUp(self):
        self.lexicon = load_lexicon(default_path("config", "lexicon.json"))
        self.profiles = load_profiles(default_path("config", "profiles.json"))
        self.faqs = nlu_sim.load_faq_stub(default_path("config", "faq_stub.json"))

    def _talk(self, store, manager, session, turns, context=None):
        """逐轮送 C 替身事件，并立刻回传播报完成。"""
        for i, text in enumerate(turns, start=1):
            payload = nlu_sim.parse(text, self.lexicon, "2026-11-24", context, self.faqs)
            env = ev.make_event(ev.EVENT_NLU_RESULT, session, i, payload)
            out = manager.handle(env)
            for item in out:
                if item["event"] == ev.EVENT_REPLY_REQUEST:
                    manager.handle(ev.make_event(
                        ev.EVENT_PLAYBACK_RESULT, session, i,
                        {"reply_id": item["payload"]["reply_id"], "status": "completed"}))
                elif item["event"] == ev.EVENT_TURN_CONTEXT:
                    context = item["payload"]
        return context

    def test_history_inheritance_two_turns(self):
        store, manager = make_env(self.lexicon, self.profiles, self.faqs)
        manager.start_session("s1")
        manager.handle(ev.make_event(ev.EVENT_PROFILE_SUGGESTION, "s1", 0,
                                     {"profile_id": "p001"}))
        self._talk(store, manager, "s1", ["还是上次那位，下周四下午", "对"])
        rows = store.list_registrations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["department"], "rehab")
        self.assertEqual(rows[0]["doctor"], "d_rehab_zhang")
        self.assertEqual(rows[0]["visit_date"], "2026-12-03")

    def test_baseline_without_inheritance(self):
        config = PolicyConfig(inherit_history=False, clarify_ambiguous=False,
                              check_slot_dependency=False)
        store, manager = make_env(self.lexicon, self.profiles, self.faqs, config)
        manager.start_session("s1")
        manager.handle(ev.make_event(ev.EVENT_PROFILE_SUGGESTION, "s1", 0,
                                     {"profile_id": "p001"}))
        state = manager.session("s1")
        payload = nlu_sim.parse("还是上次那位，下周四下午", self.lexicon, "2026-11-24")
        out = manager.handle(ev.make_event(ev.EVENT_NLU_RESULT, "s1", 1, payload))
        actions = [e["payload"].get("action") for e in out
                   if e["event"] == ev.EVENT_REPLY_REQUEST]
        self.assertEqual(actions, ["ask_department"])
        self.assertIsNone(state.draft.value("department"))

    def test_cancel_writes_nothing(self):
        store, manager = make_env(self.lexicon, self.profiles, self.faqs)
        manager.start_session("s1")
        self._talk(store, manager, "s1", ["康复科下周四下午", "取消这次登记"])
        self.assertEqual(store.count_registrations(), 0)

    def test_faq_keeps_draft_and_resumes_confirm(self):
        store, manager = make_env(self.lexicon, self.profiles, self.faqs)
        manager.start_session("s1")
        context = self._talk(store, manager, "s1",
                             ["康复科下周四下午", "复诊要带什么？", "对"])
        rows = store.list_registrations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["visit_date"], "2026-12-03")

    def test_stale_confirm_not_saved(self):
        store, manager = make_env(self.lexicon, self.profiles, self.faqs)
        manager.start_session("s1")
        self._talk(store, manager, "s1", ["康复科下周四下午"])
        # 直接跳到第 4 轮做“确认”，与快照记录的确认轮次不符
        payload = ev.nlu_payload("confirm", {}, "ok")
        manager.handle(ev.make_event(ev.EVENT_NLU_RESULT, "s1", 4, payload))
        self.assertEqual(store.count_registrations(), 0)


class ReplaySuiteTest(unittest.TestCase):
    def test_all_cases_pass_with_improved_policy(self):
        lexicon = load_lexicon(default_path("config", "lexicon.json"))
        profiles = load_profiles(default_path("config", "profiles.json"))
        faqs = nlu_sim.load_faq_stub(default_path("config", "faq_stub.json"))
        config = PolicyConfig()
        cases = load_cases(default_path("cases"))
        runner = CaseRunner(lexicon, profiles, faqs, config)
        suite = run_suite(cases, runner, "improved")
        failed = [r["case_id"] for r in suite["results"] if not r["success"]]
        self.assertEqual(failed, [])
        self.assertEqual(suite["store"].has_duplicates(), [])

    def test_baseline_is_worse_on_dependency_case(self):
        lexicon = load_lexicon(default_path("config", "lexicon.json"))
        profiles = load_profiles(default_path("config", "profiles.json"))
        faqs = nlu_sim.load_faq_stub(default_path("config", "faq_stub.json"))
        runner = CaseRunner(lexicon, profiles, faqs, PolicyConfig())
        cases = load_cases(default_path("cases"))
        suite = run_suite(cases, runner, "baseline")
        c05 = next(r for r in suite["results"] if r["case_id"] == "c05_department_change")
        self.assertFalse(c05["success"])
        self.assertEqual(c05["actual_fields"]["doctor"], "d_rehab_zhang")


if __name__ == "__main__":
    unittest.main(verbosity=2)
