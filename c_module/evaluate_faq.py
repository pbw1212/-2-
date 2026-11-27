"""固定问答库和分词，使用开发集校准阈值，再评测留出问句。"""

import argparse
import json
from itertools import product

from .faq import FaqIndex, retrieval_status
from .nlu import load_data, normalize


# The two scorers use different scales. Both grids include a zero margin.
THRESHOLD_GRID = {
    "tfidf": ([n / 100 for n in range(5, 85, 5)], [n / 100 for n in range(0, 19, 3)]),
    "bm25": ([n / 2 for n in range(1, 31)], [n / 4 for n in range(0, 13)]),
}


def validate_cases(entries, splits, replay_cases):
    faq_ids = {entry["faq_id"] for entry in entries}
    seen = {
        normalize(question)[0].lower()
        for entry in entries
        for question in [entry["question"], *entry["variants"]]
    }
    replay_keys = {normalize(case["text"])[0].lower() for case in replay_cases}
    for split, cases in splits.items():
        for case in cases:
            key = normalize(case["text"])[0].lower()
            if not key or key in seen or (split == "test" and key in replay_keys):
                raise ValueError(f"评测问句为空或与入库问句、其他样例或回放重复：{case['text']}")
            if case["faq_id"] is not None and case["faq_id"] not in faq_ids:
                raise ValueError(f"未知 FAQ 标注：{case['faq_id']}")
            seen.add(key)


def measure(cases, min_score, min_margin):
    summary = {
        "total": len(cases),
        "in_scope": 0,
        "top1_correct": 0,
        "answered": 0,
        "answered_correct": 0,
        "wrong_answers": 0,
        "out_of_scope": 0,
        "out_of_scope_rejected": 0,
    }
    details = []
    for case, ranked in cases:
        expected = case["faq_id"]
        top_id, score = ranked[0] if ranked else (None, 0.0)
        status = retrieval_status(ranked, min_score, min_margin)
        answered = status == "ok"
        correct = expected is not None and score > 0 and top_id == expected
        summary["in_scope"] += expected is not None
        summary["top1_correct"] += correct
        summary["answered"] += answered
        summary["answered_correct"] += answered and correct
        summary["wrong_answers"] += answered and not correct
        summary["out_of_scope"] += expected is None
        summary["out_of_scope_rejected"] += expected is None and not answered
        details.append({
            "text": case["text"],
            "expected_faq_id": expected,
            "top_faq_id": top_id if score > 0 else None,
            "score": round(score, 4),
            "status": status,
        })
    return summary, details


def matches_replay(expected, ranked, min_score, min_margin):
    status = retrieval_status(ranked, min_score, min_margin)
    return status == expected["status"] and (
        status != "ok" or ranked[0][0] == expected["faq_id"]
    )


def calibrate(cases, regression, retriever):
    def calibration_quality(thresholds):
        min_score, min_margin = thresholds
        summary, _ = measure(cases, min_score, min_margin)
        return (
            summary["answered_correct"] + summary["out_of_scope_rejected"],
            -summary["wrong_answers"],
            summary["answered_correct"],
            -min_score,
            -min_margin,
        )

    eligible = [
        (min_score, min_margin)
        for min_score, min_margin in product(*THRESHOLD_GRID[retriever])
        if all(matches_replay(expected, ranked, min_score, min_margin) for expected, ranked in regression)
    ]
    if not eligible:
        raise ValueError(f"{retriever} 的参数网格无解，无法保留第一阶段 FAQ 回放结果")
    return max(eligible, key=calibration_quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="同时输出每条留出问句的首选和状态")
    args = parser.parse_args()
    entries = load_data("faq.json")
    splits = load_data("faq_eval.json")
    replay_cases = [
        case for case in load_data("replay.json")["cases"] if case["expected"]["intent"] == "faq"
    ]
    validate_cases(entries, splits, replay_cases)
    results = {}

    for retriever in THRESHOLD_GRID:
        index = FaqIndex(entries, retriever=retriever)
        development = [(case, index.rank(case["text"])) for case in splits["development"]]
        test = [(case, index.rank(case["text"])) for case in splits["test"]]
        regression = [(case["expected"], index.rank(case["text"])) for case in replay_cases]
        min_score, min_margin = calibrate(development, regression, retriever)
        dev_summary, _ = measure(development, min_score, min_margin)
        test_summary, details = measure(test, min_score, min_margin)
        default_summary, _ = measure(test, index.min_score, index.min_margin)
        result = {
            "min_score": min_score,
            "min_margin": min_margin,
            "development": dev_summary,
            "test": test_summary,
            "default_test": {
                "min_score": index.min_score,
                "min_margin": index.min_margin,
                **default_summary,
            },
        }
        if args.details:
            result["test_details"] = details
        results[retriever] = result

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
