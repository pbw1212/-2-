# -*- coding: utf-8 -*-
"""评测：同一组用例分别跑基线与改进配置，统计登记成功率、用户轮数等指标。

指标口径（与 D 模块方案一致）：
- 登记成功率：计划完成登记的任务中，正确保存且字段与目标一致的比例；
  取消用例以“退出成功、记录未写入”作为验收。
- 用户轮数与重问：按任务记录用户发言轮数和被重复询问次数，报告成功任务的
  中位数及样本数。
- 局部更正与保存：改口后目标字段正确、其他仍适用的字段保持，以及记录条数。
"""
from __future__ import annotations

import json
import os
import statistics

from .replay import CaseRunner
from .store import RegistrationStore

# 固定顺序表单基线：不做历史继承，也不做按字段澄清
BASELINE_OVERRIDES = {
    "inherit_history": False,
    "clarify_ambiguous": False,
    "history_confirm_style": "ok",
    "check_slot_dependency": False,
}


def run_suite(cases, runner, variant="improved", db_path=":memory:", log_dir=None,
              fresh=True):
    overrides = BASELINE_OVERRIDES if variant == "baseline" else {}
    if fresh and db_path not in (":memory:", "", None) and os.path.exists(db_path):
        os.remove(db_path)
    store = RegistrationStore(db_path)
    results = []
    for case in cases:
        result = runner.run(case, store=store, config_override=overrides,
                            variant=variant)
        results.append(result)
        if log_dir:
            write_transcript_log(result, log_dir, variant)
    return {"variant": variant, "results": results, "db_path": db_path,
            "store": store}


def write_transcript_log(result, log_dir, variant):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "%s_%s.jsonl" % (result["case_id"], variant))
    with open(path, "w", encoding="utf-8") as f:
        for item in result["transcript"]:
            f.write(json.dumps({"event": item["event"], "note": item.get("note")},
                               ensure_ascii=False) + "\n")


def _median(values):
    values = list(values)
    if not values:
        return None
    return round(float(statistics.median(values)), 2)


def _mean(values):
    values = list(values)
    if not values:
        return None
    return round(sum(values) / float(len(values)), 2)


SUBSETS = {
    "历史继承类": ["c03_inherit_history", "c15_inherit_other_profile"],
    "歧义澄清类": ["c09_ambiguous_date"],
    "局部更正类": ["c04_date_correction", "c05_department_change", "c14_deny_then_fix"],
    "异常恢复类": ["c07_cancel", "c08_duplicate_confirm", "c11_playback_retry",
                   "c12_save_failure", "c13_late_turn", "c10_no_input"],
}


def _subset_metrics(per_case, ids):
    picked = [c for c in per_case if c["case_id"] in ids]
    if not picked:
        return None
    return {
        "cases": len(picked),
        "success": sum(1 for c in picked if c["success"]),
        "turns_mean": round(sum(c["user_turns"] for c in picked) / float(len(picked)), 2),
        "repeat_total": sum(c["repeat_questions"] for c in picked),
    }


def aggregate(suite):
    results = suite["results"]
    total = len(results)
    successes = [r for r in results if r["success"]]
    planned = [r for r in results if int(r["expected"].get("registrations", 1)) == 1]
    planned_ok = [r for r in planned if r["success"]]

    field_total = 0
    field_ok = 0
    for r in results:
        expected = r["expected"].get("fields") or {}
        actual = r["actual_fields"] or {}
        for key, value in expected.items():
            field_total += 1
            if actual.get(key) == value:
                field_ok += 1

    duplicate_rows = sum(max(0, int(item["n"]) - 1)
                         for item in suite["store"].has_duplicates())
    written = suite["store"].count_registrations()
    cancel_case = next((r for r in results if r["case_id"] == "c07_cancel"), None)
    cancel_ok = bool(cancel_case and cancel_case["success"]
                     and len(cancel_case["registrations"]) == 0
                     and suite["store"].count_by_session(cancel_case["session_id"]) == 0)

    per_case = [
        {
            "case_id": r["case_id"],
            "title": r["title"],
            "success": r["success"],
            "user_turns": r["user_turns"],
            "within_budget": r["within_budget"],
            "repeat_questions": r["repeat_questions"],
            "registrations": len(r["registrations"]),
            "actual_fields": r["actual_fields"],
            "expected_fields": r["expected"].get("fields") or {},
            "actions": [x["action"] for x in r["replies"]],
        }
        for r in results
    ]

    return {
        "variant": suite["variant"],
        "cases": total,
        "successes": len(successes),
        "success_rate": round(len(successes) / float(total), 4) if total else None,
        "planned_tasks": len(planned),
        "planned_success_rate": round(len(planned_ok) / float(len(planned)), 4) if planned else None,
        "user_turns_median": _median(r["user_turns"] for r in successes),
        "user_turns_mean": _mean(r["user_turns"] for r in successes),
        "user_turns_values": [r["user_turns"] for r in successes],
        "repeat_questions_median": _median(r["repeat_questions"] for r in successes),
        "repeat_questions_mean": _mean(r["repeat_questions"] for r in successes),
        "repeat_questions_total": sum(r["repeat_questions"] for r in successes),
        "field_total": field_total,
        "field_ok": field_ok,
        "field_accuracy": round(field_ok / float(field_total), 4) if field_total else None,
        "duplicate_rows": duplicate_rows,
        "registrations_written": written,
        "cancel_no_write": cancel_ok,
        "playback_retry_ok": bool(next((r["success"] for r in results
                                        if r["case_id"] == "c11_playback_retry"), False)),
        "save_retry_ok": bool(next((r["success"] for r in results
                                    if r["case_id"] == "c12_save_failure"), False)),
        "per_case": per_case,
        "turns_within_budget": sum(1 for r in results if r["within_budget"]),
        "subsets": dict((name, _subset_metrics(per_case, ids))
                        for name, ids in SUBSETS.items()),
    }


def render_markdown(improved, baseline, cases):
    lines = []
    lines.append("# D 模块评测结果\n")
    lines.append("用例数：%d；参考日固定为 2026-11-24；样本来源为结构化回放用例（非真人录音）。\n"
                 % int(improved["cases"]))
    lines.append("## 一、基线与改进组总体对照\n")
    lines.append("| 指标 | 固定顺序基线 | 历史继承与按字段澄清 | 口径 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| 用例成功数 | %d/%d | %d/%d | 字段与记录条数均符合预期 |"
                 % (baseline["successes"], baseline["cases"],
                    improved["successes"], improved["cases"]))
    lines.append("| 登记成功率 | %.1f%% | %.1f%% | 计划完成登记的任务 |"
                 % (100 * (baseline["planned_success_rate"] or 0),
                    100 * (improved["planned_success_rate"] or 0)))
    lines.append("| 用户轮数中位数 | %s | %s | 仅成功任务 |"
                 % (baseline["user_turns_median"], improved["user_turns_median"]))
    lines.append("| 用户轮数均值 | %s | %s | 仅成功任务 |"
                 % (baseline["user_turns_mean"], improved["user_turns_mean"]))
    lines.append("| 重复询问中位数 | %s | %s | 连续追问同一字段的次数 |"
                 % (baseline["repeat_questions_median"], improved["repeat_questions_median"]))
    lines.append("| 关键字段正确率 | %.1f%% | %.1f%% | %d 个字段 |"
                 % (100 * (baseline["field_accuracy"] or 0),
                    100 * (improved["field_accuracy"] or 0),
                    int(improved["field_total"])))
    lines.append("| 重复写入行数 | %d | %d | 同一 registration_id 的多余记录 |"
                 % (baseline["duplicate_rows"], improved["duplicate_rows"]))
    lines.append("| 取消不写入 | %s | %s | c07 取消用例 |"
                 % ("通过" if baseline["cancel_no_write"] else "未通过",
                    "通过" if improved["cancel_no_write"] else "未通过"))
    lines.append("")
    lines.append("## 二、按场景分组对照\n")
    lines.append("| 场景 | 用例数 | 基线成功 | 改进成功 | 基线平均轮数 | 改进平均轮数 | 基线重复询问 | 改进重复询问 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in SUBSETS:
        b = (baseline.get("subsets") or {}).get(name) or {}
        i = (improved.get("subsets") or {}).get(name) or {}
        lines.append("| %s | %s | %s/%s | %s/%s | %s | %s | %s | %s |"
                     % (name, i.get("cases"), b.get("success"), b.get("cases"),
                        i.get("success"), i.get("cases"), b.get("turns_mean"),
                        i.get("turns_mean"), b.get("repeat_total"),
                        i.get("repeat_total")))
    lines.append("")
    lines.append("注：分组平均轮数按该组全部用例计算（含未通过用例）；总体表中的均值只统计成功任务。\n")
    lines.append("## 三、逐用例结果（改进组）\n")
    lines.append("| 用例 | 内容 | 成功 | 用户轮数 | 重复询问 | 记录数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in improved["per_case"]:
        lines.append("| %s | %s | %s | %d | %d | %d |"
                     % (item["case_id"], item["title"], "是" if item["success"] else "否",
                        item["user_turns"], item["repeat_questions"],
                        item["registrations"]))
    lines.append("")
    lines.append("## 四、逐用例结果（基线）\n")
    lines.append("| 用例 | 内容 | 成功 | 用户轮数 | 重复询问 | 记录数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in baseline["per_case"]:
        lines.append("| %s | %s | %s | %d | %d | %d |"
                     % (item["case_id"], item["title"], "是" if item["success"] else "否",
                        item["user_turns"], item["repeat_questions"],
                        item["registrations"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def render_cases_csv(suite):
    lines = ["case_id,title,variant,success,user_turns,repeat_questions,registrations,"
             "department,doctor,date,period,actions"]
    for item in suite["per_case"]:
        fields = item["actual_fields"] or {}
        actions = " > ".join(item["actions"])
        lines.append(",".join([
            item["case_id"], '"%s"' % item["title"], suite["variant"],
            "1" if item["success"] else "0", str(item["user_turns"]),
            str(item["repeat_questions"]), str(item["registrations"]),
            str(fields.get("department")), str(fields.get("doctor")),
            str(fields.get("date")), str(fields.get("period")), '"%s"' % actions,
        ]))
    return "\n".join(lines) + "\n"
