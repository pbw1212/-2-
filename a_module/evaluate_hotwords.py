"""用固定录音比较 Zipformer 开关热词的效果。"""

import argparse
import json
import math
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from module_a.asr import (
    DEFAULT_ASR_MODEL, DEFAULT_BIAS_WEIGHT, DEFAULT_HOTWORDS, MODEL_ID, MODULE_DIR, Zipformer,
    load_hotwords,
)
from module_a.audio import SAMPLE_RATE, duration_ms, float_samples, read_wav

MANIFEST = MODULE_DIR / "examples" / "clinic" / "cases.json"
NOISE_SNR_DB = -5
TUNING_WEIGHTS = (0.0, 1.5, 3.0)


def normalize(text):
    return "".join(char.lower() for char in text if char.isalnum())


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        current = [index]
        for column, b in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[-1] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def case_pcm(example, manifest):
    base = manifest.parent / "audio"
    clips = [
        read_wav(base / f"{example['id']}-{index:02d}.wav")
        for index in range(1, len(example["parts"]) + 1)
    ]
    gap = bytes(round(example.get("pause_ms", 0) * SAMPLE_RATE / 1000) * 2)
    return gap.join(clips)


def add_noise(pcm, case_id):
    samples = float_samples(pcm)
    seed = int.from_bytes(case_id.encode("ascii"), "little")
    noise = np.random.default_rng(seed).standard_normal(samples.shape, dtype=np.float32)
    signal_rms = np.sqrt(np.mean(samples**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    noise *= signal_rms / (noise_rms * 10 ** (NOISE_SNR_DB / 20))
    return np.clip(np.rint((samples + noise) * 32768), -32768, 32767).astype("<i2").tobytes()


def prepare(examples, manifest, noisy):
    rows = []
    for example in examples:
        pcm = case_pcm(example, manifest)
        if noisy:
            pcm = add_noise(pcm, example["id"])
        rows.append({
            "id": example["id"],
            "voice": example["voice"],
            "context": example["context"],
            "reference": "".join(example["parts"]),
            "targets": example["targets"],
            "audio_ms": round(duration_ms(pcm), 2),
            "pcm": pcm,
        })
    return rows


def predict(model, row, lexicon, weight):
    started = perf_counter()
    stream = model.stream(lexicon[row["context"]], weight=weight)
    # The recognizer still decodes sequential chunks when the whole file is already buffered.
    stream.push(row["pcm"])
    text = stream.finish()
    elapsed = perf_counter() - started
    hyp = normalize(text)
    ref = normalize(row["reference"])
    spurious = [word for word in lexicon["registration"] if word in hyp and word not in ref]
    return {
        "text": text,
        "edits": edit_distance(hyp, ref),
        "target_hits": sum(normalize(word) in hyp for word in row["targets"]),
        "spurious_hotwords": sorted(spurious),
        "rtf": round(1000 * elapsed / row["audio_ms"], 4),
    }


def summarize(rows, decoded):
    reference_chars = sum(len(normalize(row["reference"])) for row in rows)
    edit_count = sum(item["edits"] for item in decoded)
    positive = [(row, item) for row, item in zip(rows, decoded) if row["targets"]]
    controls = [item for row, item in zip(rows, decoded) if not row["targets"]]
    return {
        "cases": len(rows),
        "cer": round(edit_count / reference_chars, 6),
        "edit_count": edit_count,
        "reference_chars": reference_chars,
        "target_slots_hit": sum(item["target_hits"] for item in decoded),
        "target_slots_total": sum(len(row["targets"]) for row in rows),
        "all_targets_hit_cases": sum(item["target_hits"] == len(row["targets"]) for row, item in positive),
        "positive_cases": len(positive),
        "spurious_control_cases": sum(bool(item["spurious_hotwords"]) for item in controls),
        "control_cases": len(controls),
        "median_rtf": round(median(item["rtf"] for item in decoded), 4),
    }


def select_weight(model, rows, lexicon):
    sweep = []
    for weight in TUNING_WEIGHTS:
        results = [predict(model, row, lexicon, weight) for row in rows]
        sweep.append({"weight": weight, **summarize(rows, results)})
    control_limit = sweep[0]["spurious_control_cases"]
    eligible = (entry for entry in sweep if entry["spurious_control_cases"] <= control_limit)
    winner = min(eligible, key=lambda entry: (
        -entry["target_slots_hit"], entry["edit_count"], entry["weight"],
    ))
    return winner["weight"], sweep


def compare(model, rows, lexicon, weight):
    records = []
    beam = []
    hotwords = []
    for row in rows:
        beam.append(predict(model, row, lexicon, 0.0))
        hotwords.append(predict(model, row, lexicon, weight))
        records.append({
            **{key: value for key, value in row.items() if key != "pcm"},
            "beam": beam[-1],
            "hotwords": hotwords[-1],
        })
    return {
        "summary": {"beam": summarize(rows, beam), "hotwords": summarize(rows, hotwords)},
        "results": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_ASR_MODEL)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--hotwords", type=Path, default=DEFAULT_HOTWORDS)
    parser.add_argument("--weight", type=float, default=DEFAULT_BIAS_WEIGHT)
    parser.add_argument("--tune", action="store_true", help="只用带噪的开发样例选热词权重")
    parser.add_argument("--dev-only", action="store_true", help="只运行开发样例")
    parser.add_argument("--output", type=Path, help="保存逐句结果")
    args = parser.parse_args()
    if not math.isfinite(args.weight) or args.weight < 0:
        parser.error("热词权重需要是非负数")
    torch.set_num_threads(2)
    model = Zipformer(args.asr_model)
    lexicon = load_hotwords(args.hotwords)
    examples = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    weight = args.weight
    report = {
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()}",
        "model": MODEL_ID,
        "noise_snr_db": NOISE_SNR_DB,
    }
    if args.tune:
        dev = prepare([case for case in examples if case["split"] == "dev"], args.manifest, noisy=True)
        weight, report["dev_weight_sweep"] = select_weight(model, dev, lexicon)
    report["hotword_weight"] = weight
    split = "dev" if args.dev_only else "test"
    rows = [case for case in examples if case["split"] == split]
    report["split"] = split
    report["conditions"] = {
        "clean": compare(model, prepare(rows, args.manifest, noisy=False), lexicon, weight),
        "noise": compare(model, prepare(rows, args.manifest, noisy=True), lexicon, weight),
    }
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "hotword_weight": weight,
        "results": {key: group["summary"] for key, group in report["conditions"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
