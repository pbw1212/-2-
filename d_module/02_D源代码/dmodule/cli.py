# -*- coding: utf-8 -*-
"""命令行入口：回放、评测、交互式对话与本地库初始化。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import nlu_sim
from .config import PolicyConfig, default_path, load_lexicon, load_policy, load_profiles
from .evaluation import (aggregate, render_cases_csv, render_markdown, run_suite,
                         write_transcript_log)
from .policy import DialogManager
from .replay import CaseRunner, load_cases
from .store import RegistrationStore


def _load_all():
    lexicon = load_lexicon(default_path("config", "lexicon.json"))
    profiles = load_profiles(default_path("config", "profiles.json"))
    faqs = nlu_sim.load_faq_stub(default_path("config", "faq_stub.json"))
    policy = load_policy(default_path("config", "policy.json"))
    return lexicon, profiles, faqs, policy


def _print_reply(env):
    payload = env["payload"]
    print("D  → %s" % payload.get("reply_text"))
    print("      [action=%s, phase=%s]" % (payload.get("action"),
                                           payload.get("dialog_phase")))


def print_transcript(result, show_events=False):
    """把回放结果打印成一段可读的对话。"""
    print("=" * 78)
    print("用例：%s  %s" % (result["case_id"], result["title"]))
    print("=" * 78)
    last_turn = None
    for item in result["transcript"]:
        env = item["event"]
        note = item.get("note") or {}
        if env["event"] == "NLU_RESULT":
            if last_turn is not None and env["turn_id"] != last_turn:
                pass
            last_turn = env["turn_id"]
            text = note.get("user_text")
            if not text:
                payload = env["payload"]
                text = "[C 事件] intent=%s status=%s slots=%s" % (
                    payload.get("intent"), payload.get("status"),
                    json.dumps(payload.get("slots"), ensure_ascii=False))
            print("\n用户(turn %d)：%s" % (env["turn_id"], text))
        elif env["event"] == "REPLY_REQUEST":
            _print_reply(env)
        elif env["event"] == "TURN_CONTEXT":
            payload = env["payload"]
            print("      [TURN_CONTEXT turn=%d phase=%s qtype=%s expected=%s]"
                  % (env["turn_id"], payload.get("dialog_phase"),
                     payload.get("question_type"), payload.get("expected_slots")))
        elif env["event"] == "PLAYBACK_RESULT" and show_events:
            print("      [PLAYBACK_RESULT %s]" % env["payload"].get("status"))
    print("\n结果：%s；用户轮数=%d；重复询问=%d；记录数=%d"
          % ("成功" if result["success"] else "未通过",
             result["user_turns"], result["repeat_questions"], len(result["registrations"])))
    for row in result["registrations"]:
        print("  记录 %s：科室=%s 医生=%s 日期=%s 时段=%s"
              % (row["registration_id"], row["department"], row["doctor"],
                 row["visit_date"], row["period"]))


def cmd_eval(args):
    lexicon, profiles, faqs, policy = _load_all()
    cases = load_cases(args.cases or default_path("cases"))
    runner = CaseRunner(lexicon, profiles, faqs, policy)
    out = args.out
    os.makedirs(os.path.join(out, "db"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)

    improved = run_suite(cases, runner, "improved",
                         db_path=os.path.join(out, "db", "registrations_improved.sqlite"),
                         log_dir=os.path.join(out, "logs"))
    baseline = run_suite(cases, runner, "baseline",
                         db_path=os.path.join(out, "db", "registrations_baseline.sqlite"),
                         log_dir=os.path.join(out, "logs"))

    summary = {
        "reference_date": "2026-11-24",
        "cases": len(cases),
        "improved": aggregate(improved),
        "baseline": aggregate(baseline),
    }
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary["improved"], summary["baseline"], cases))
    with open(os.path.join(out, "cases.csv"), "w", encoding="utf-8-sig") as f:
        f.write(render_cases_csv(summary["improved"]))
        f.write(render_cases_csv(summary["baseline"]))

    print("评测完成：%d 个用例" % len(cases))
    print("  改进组：成功 %d/%d，登记成功率 %.1f%%，用户轮数中位数 %s"
          % (summary["improved"]["successes"], summary["improved"]["cases"],
             100 * (summary["improved"]["planned_success_rate"] or 0),
             summary["improved"]["user_turns_median"]))
    print("  基线：  成功 %d/%d，登记成功率 %.1f%%，用户轮数中位数 %s"
          % (summary["baseline"]["successes"], summary["baseline"]["cases"],
             100 * (summary["baseline"]["planned_success_rate"] or 0),
             summary["baseline"]["user_turns_median"]))
    failed = [c["case_id"] for c in summary["improved"]["per_case"] if not c["success"]]
    if failed:
        print("  未通过用例：%s" % ", ".join(failed))
    print("  结果目录：%s" % os.path.abspath(out))
    return 0


def cmd_replay(args):
    lexicon, profiles, faqs, policy = _load_all()
    cases = load_cases(args.cases or default_path("cases"))
    case = next((c for c in cases if c["case_id"] == args.case), None)
    if case is None:
        print("找不到用例：%s" % args.case)
        return 2
    runner = CaseRunner(lexicon, profiles, faqs, policy)
    store = RegistrationStore(args.db or ":memory:")
    result = runner.run(case, store=store, variant=args.variant)
    print_transcript(result, show_events=args.events)
    if args.out:
        write_transcript_log(result, os.path.dirname(os.path.abspath(args.out)), args.variant)
    return 0


def cmd_demo(args):
    """三阶段验收用的完整回放：继承历史 → 改日期 → 显式确认。"""
    lexicon, profiles, faqs, policy = _load_all()
    case = {
        "case_id": "demo_inherit_correct_confirm",
        "title": "沿用上次医生 → 改日期 → 显式确认",
        "reference_date": "2026-11-24",
        "steps": [
            {"type": "profile", "profile_id": "p001"},
            {"type": "nlu", "text": "还是上次那位，下周四下午"},
            {"type": "nlu", "text": "改成下周五"},
            {"type": "nlu", "text": "对"},
        ],
        "expect": {
            "registrations": 1,
            "fields": {"department": "rehab", "doctor": "d_rehab_zhang",
                       "date": "2026-12-04", "period": "afternoon"},
        },
    }
    runner = CaseRunner(lexicon, profiles, faqs, policy)
    out = args.out
    os.makedirs(out, exist_ok=True)
    store = RegistrationStore(os.path.join(out, "demo_registrations.sqlite"))
    result = runner.run(case, store=store, variant="improved")
    print_transcript(result, show_events=True)
    write_transcript_log(result, os.path.join(out, "logs"), "demo")
    print("\n本次登记库：%s" % os.path.abspath(os.path.join(out, "demo_registrations.sqlite")))
    store.close()
    return 0


def cmd_chat(args):
    """终端里的纯文本对话，用 C 的文本替身跑真实流程。"""
    lexicon, profiles, faqs, policy = _load_all()
    store = RegistrationStore(args.db or os.path.join(default_path(), "chat.sqlite"))
    manager = DialogManager(store, lexicon, profiles, policy, args.reference)
    session_id = args.session
    manager.start_session(session_id)
    state = manager.session(session_id)
    print("D 已就绪（输入 q 退出，输入 :profile p001 选择模拟档案）")
    context = None
    turn = 1
    while True:
        try:
            raw = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if raw in ("q", "quit", "exit"):
            break
        if raw.startswith(":profile "):
            profile_id = raw.split(" ", 1)[1].strip()
            from . import events as ev
            manager.handle(ev.make_event(ev.EVENT_PROFILE_SUGGESTION, session_id, 0,
                                         {"profile_id": profile_id, "match_status": "matched"}))
            print("已选择模拟档案：%s" % profile_id)
            continue
        payload = nlu_sim.parse(raw, lexicon, args.reference, context, faqs)
        from . import events as ev
        out = manager.handle(ev.make_event(ev.EVENT_NLU_RESULT, session_id, turn, payload))
        for env in out:
            if env["event"] == ev.EVENT_REPLY_REQUEST:
                print("D：%s" % env["payload"]["reply_text"])
                manager.handle(ev.make_event(ev.EVENT_PLAYBACK_RESULT, session_id, turn,
                                             {"reply_id": env["payload"]["reply_id"],
                                              "status": "completed"}))
            elif env["event"] == ev.EVENT_TURN_CONTEXT:
                context = env["payload"]
                turn = int(env["turn_id"])
        if state.closed:
            print("\n（会话结束，记录已写入 %s）" % args.db)
            break
    store.close()
    return 0


def cmd_initdb(args):
    store = RegistrationStore(args.path)
    print("已初始化本地登记库：%s" % os.path.abspath(args.path))
    print("表：registrations（registration_id 唯一）、dialog_log")
    store.close()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="离线复诊登记助手 D 模块：多轮对话管理与本地登记")
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser("eval", help="跑基线与改进组的对照评测")
    p_eval.add_argument("--out", default=os.path.join(default_path(), "..", "out_eval"))
    p_eval.add_argument("--cases", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_replay = sub.add_parser("replay", help="回放单个用例")
    p_replay.add_argument("--case", required=True)
    p_replay.add_argument("--cases", default=None)
    p_replay.add_argument("--variant", default="improved", choices=["improved", "baseline"])
    p_replay.add_argument("--db", default=None)
    p_replay.add_argument("--out", default=None)
    p_replay.add_argument("--events", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_demo = sub.add_parser("demo", help="跑“继承历史→改日期→确认”的完整回放")
    p_demo.add_argument("--out", default=os.path.join(default_path(), "..", "out_demo"))
    p_demo.set_defaults(func=cmd_demo)

    p_chat = sub.add_parser("chat", help="终端文本对话")
    p_chat.add_argument("--db", default=None)
    p_chat.add_argument("--session", default="chat01")
    p_chat.add_argument("--reference", default="2026-11-24")
    p_chat.set_defaults(func=cmd_chat)

    p_init = sub.add_parser("initdb", help="初始化本地登记库")
    p_init.add_argument("--path", default=os.path.join(default_path(), "registrations.sqlite"))
    p_init.set_defaults(func=cmd_initdb)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)
