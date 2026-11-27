"""回放 C 模块阶段一的文本事件并输出 JSON 日志。"""

import argparse
import json
from datetime import date

from .faq import RETRIEVERS, SEGMENTERS
from .nlu import CModule, load_data, make_event


def replay(module, session_id, turn_id, reference_date, case):
    context = None
    if case.get("with_context", True):
        expected_slots = case.get("expected_slots", [])
        phase = case.get("phase", "idle")
        question_type = "open"
        if phase == "confirming":
            question_type = "yes_no"
        elif expected_slots:
            question_type = "targeted"
        context = make_event("TURN_CONTEXT", session_id, turn_id, {
            "dialog_phase": phase,
            "question_type": question_type,
            "expected_slots": expected_slots,
            "expected_slot": expected_slots[0] if len(expected_slots) == 1 else None,
            "reference_date": reference_date,
        })

    asr = make_event("ASR_FINAL", session_id, turn_id, {
        "status": case.get("asr_status", "ok"),
        "final_text": case["text"],
        "start_audio_ms": None,
        "end_audio_ms": None,
    })
    if asr["payload"]["status"] == "error":
        asr["payload"]["error_code"] = "mock_decoder_error"

    diagnostics = {}
    result = module.process(asr, context, diagnostics=diagnostics)
    record = {
        "case": case["id"],
        "turn_context": context,
        "asr_final": asr,
        "nlu_result": result,
    }
    if diagnostics:
        record["faq_diagnostics"] = diagnostics
    expected = case.get("expected")
    if expected is not None:
        record["passed"] = all(result["payload"].get(key) == value for key, value in expected.items())
        if not record["passed"]:
            record["expected"] = expected
    print(json.dumps(record, ensure_ascii=False))
    return record.get("passed", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--case", help="只回放指定的内置样例")
    modes.add_argument("--text", help="手工输入一句 ASR 最终转写")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="tfidf")
    parser.add_argument("--segmenter", choices=SEGMENTERS, default="builtin")
    parser.add_argument("--reference-date", type=date.fromisoformat, help="自定义文本的会话参考日 YYYY-MM-DD")
    parser.add_argument("--phase", choices=("idle", "collecting", "confirming"), default="idle")
    parser.add_argument("--expect", action="append", choices=("department", "doctor", "date", "period"), default=[])
    args = parser.parse_args()

    try:
        module = CModule(retriever=args.retriever, segmenter=args.segmenter)
    except ImportError as exc:
        parser.error(str(exc))
    fixtures = load_data("replay.json")
    if args.text is not None:
        case = {"id": "custom", "text": args.text, "phase": args.phase, "expected_slots": args.expect}
        reference_date = args.reference_date.isoformat() if args.reference_date else fixtures["reference_date"]
        replay(module, "c-phase1-custom", 1, reference_date, case)
        return 0

    cases = fixtures["cases"]
    selected = [(turn, case) for turn, case in enumerate(cases, 1) if not args.case or case["id"] == args.case]
    if not selected:
        parser.error(f"样例不存在：{args.case}")
    outcomes = [
        replay(module, fixtures["session_id"], turn, fixtures["reference_date"], case)
        for turn, case in selected
    ]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
