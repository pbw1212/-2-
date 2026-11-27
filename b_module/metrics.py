"""指标与对照表（S11 / S12）。

口径按 B 方案与全组联调方案统一：

==============  ==========================================================
提前截断率      早于人工标注轮末（容差 1 帧 = 20 ms）的触发轮数 ÷ 有效轮数
端点延迟        触发时刻 − 真值轮末，**只在未提前截断的轮次上统计**（中位数、P90）
未触发          回放结束仍未发出 TURN_STOP 的轮数
==============  ==========================================================

分母一律同时打印（分子/分母），避免“只给一个百分比”的含糊。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from . import paths
from .replay import ReplayResult

GROUP_ORDER = ["open_pause", "targeted", "open_long", "open_full", "yes_no_proxy"]

# 汇报用的两个池：开放式轮（需要耐心等）与短答轮（需要快）
POOLED_GROUPS = {
    "open": ["open_pause", "open_long", "open_full"],
    "short": ["targeted", "yes_no_proxy"],
}


@dataclass(frozen=True)
class Outcome:
    strategy: str
    case_id: str
    group: str
    question_type: str
    truth_end_ms: float
    decision_ms: float | None
    untriggered: bool
    early: bool
    latency_ms: float | None
    reason: str | None
    basis: str | None

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        for key in ("truth_end_ms", "decision_ms", "latency_ms"):
            if row[key] is not None:
                row[key] = round(float(row[key]), 2)
        return row


def evaluate(result: ReplayResult, tolerance_ms: float) -> Outcome:
    if result.stop is None:
        return Outcome(
            strategy=result.strategy,
            case_id=result.case_id,
            group=result.group,
            question_type=result.question_type,
            truth_end_ms=result.truth_end_ms,
            decision_ms=None,
            untriggered=True,
            early=False,
            latency_ms=None,
            reason=None,
            basis=None,
        )
    decision = float(result.stop.decision_audio_ms)
    early = decision < result.truth_end_ms - tolerance_ms
    return Outcome(
        strategy=result.strategy,
        case_id=result.case_id,
        group=result.group,
        question_type=result.question_type,
        truth_end_ms=result.truth_end_ms,
        decision_ms=decision,
        untriggered=False,
        early=early,
        latency_ms=None if early else decision - result.truth_end_ms,
        reason=result.stop.reason,
        basis=result.stop.basis,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _bucket(outcomes: Iterable[Outcome]) -> Dict[str, Any]:
    items = list(outcomes)
    if not items:
        return {"turns": 0}
    triggered = [item for item in items if not item.untriggered]
    early = [item for item in triggered if item.early]
    normal = [item for item in triggered if not item.early]
    latencies = [float(item.latency_ms) for item in normal if item.latency_ms is not None]
    return {
        "turns": len(items),
        "triggered": len(triggered),
        "untriggered": len(items) - len(triggered),
        "early": len(early),
        "early_rate": round(len(early) / len(triggered), 4) if triggered else None,
        "early_denominator": len(triggered),
        "latency_median_ms": None if not latencies else round(_percentile(latencies, 50), 2),
        "latency_p90_ms": None if not latencies else round(_percentile(latencies, 90), 2),
        "latency_n": len(latencies),
        "reasons": {},
    }


def summarize(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """按策略汇总；同时给出分组明细（open_pause 是汇报主口径）。"""
    strategies = list(dict.fromkeys(item.strategy for item in outcomes))
    summary: Dict[str, Any] = {"strategies": {}, "group_order": GROUP_ORDER}
    for strategy in strategies:
        subset = [item for item in outcomes if item.strategy == strategy]
        overall = _bucket(subset)
        for item in subset:
            if item.reason:
                overall["reasons"][item.reason] = overall["reasons"].get(item.reason, 0) + 1
        by_group = {
            group: _bucket([item for item in subset if item.group == group])
            for group in GROUP_ORDER
            if any(item.group == group for item in subset)
        }
        pooled = {
            name: _bucket([item for item in subset if item.group in members])
            for name, members in POOLED_GROUPS.items()
        }
        summary["strategies"][strategy] = {"overall": overall, "by_group": by_group,
                                           "pooled": pooled}
    return summary


def write_csv(outcomes: Sequence[Outcome], path: str | Path) -> Path:
    path = Path(path)
    paths.ensure_dir(path.parent)
    rows = [item.to_row() for item in outcomes]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(value: Any, suffix: str = "", digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def to_markdown(summary: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> str:
    lines: List[str] = ["# 模块 B 阶段一：固定阈值与自适应端点的对照结果", ""]
    if meta:
        lines.append(
            f"样本 {meta.get('case_count')} 条（{meta.get('group_text')}），"
            f"容差 {meta.get('tolerance_ms')} ms，数据来源 `a_module/examples`，"
            f"标注口径见 `data/annotations.json`。"
        )
        lines.append("")

    header = ("| 策略 | 触发轮数 | 提前截断 | 截断率 | 延迟中位数 | 延迟 P90 | 延迟样本数 | 未触发 |\n"
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append("## 1. 总体")
    lines.append("")
    lines.append(header)
    for strategy, payload in summary["strategies"].items():
        overall = payload["overall"]
        early_cell = f"{overall['early']}/{overall['early_denominator']}"
        rate = "—" if overall["early_rate"] is None else f"{overall['early_rate']*100:.0f}%"
        lines.append(
            f"| `{strategy}` | {overall['triggered']} | {early_cell} | {rate} | "
            f"{_fmt(overall['latency_median_ms'], digits=0)} | "
            f"{_fmt(overall['latency_p90_ms'], digits=0)} | "
            f"{overall['latency_n']} | {overall['untriggered']} |"
        )
    lines.append("")
    lines.append("口径：截断率分母为**触发轮数**；延迟只在未提前截断的轮次上统计"
                 "（`latency_n` 即纳入轮数）。")
    lines.append("")

    lines.append("## 2. 分组明细（截断率 / 延迟中位数）")
    lines.append("")
    group_header = "| 策略 | " + " | ".join(GROUP_ORDER) + " |"
    group_rule = "| :--- | " + " | ".join([":---:"] * len(GROUP_ORDER)) + " |"
    lines.extend([group_header, group_rule])
    for strategy, payload in summary["strategies"].items():
        cells = []
        for group in GROUP_ORDER:
            bucket = payload["by_group"].get(group)
            if not bucket or not bucket.get("triggered"):
                cells.append("—")
                continue
            rate = "—" if bucket["early_rate"] is None else f"{bucket['early_rate']*100:.0f}%"
            med = "—" if bucket["latency_median_ms"] is None else f"{bucket['latency_median_ms']:.0f}"
            cells.append(f"{rate} / {med}")
        lines.append(f"| `{strategy}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("单元格为 `提前截断率 / 端点延迟中位数(ms)`；`—` 表示该策略在该组没有触发轮次。")
    lines.append("")

    lines.append("## 3. 触发原因分布")
    lines.append("")
    lines.append("| 策略 | " + " | ".join(["fixed_silence", "adaptive_silence", "no_speech", "timeout"]) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * 4) + " |")
    for strategy, payload in summary["strategies"].items():
        reasons = payload["overall"]["reasons"]
        cells = [str(reasons.get(name, 0)) for name in
                 ("fixed_silence", "adaptive_silence", "no_speech", "timeout")]
        lines.append(f"| `{strategy}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def plot_summary(summary: Mapping[str, Any], path: str | Path, *, group: str = "open_pause") -> Path:
    """两栏主图。

    * **左栏**：带句中停顿的开放式轮——截断率 vs 等待时间，固定阈值是一条取舍曲线，
      两个自适应点落在曲线的末端（0% 截断）。
    * **右栏**：在左栏已确认“同等 0% 截断”的前提下，看**短答轮**要等多久。
      自适应的收益全部在这里：短答从 1380 ms 降到 470–480 ms。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    paths.ensure_dir(path.parent)

    panel_group = group if group != "all" else "open_pause"

    fixed_points: List[tuple[float, float, str]] = []
    adaptive_points: List[tuple[float, float, str]] = []
    for strategy, payload in summary["strategies"].items():
        bucket = payload["by_group"].get(panel_group) or payload["overall"]
        if not bucket.get("triggered") or bucket.get("early_rate") is None:
            continue
        rate = bucket["early_rate"] * 100.0
        latency = bucket.get("latency_median_ms")
        if strategy.startswith("fixed:"):
            label = strategy.split(":", 1)[1]
            if latency is None and rate < 99.9:
                continue
            # 全部轮次都截断时没有“未截断轮”可算延迟，此时等待量就是阈值本身
            fixed_points.append((float(latency) if latency is not None else float(label), rate, label))
        elif strategy == "adaptive_question":
            if latency is not None:
                adaptive_points.append((float(latency), rate, "adaptive-1 (question type)"))
        elif strategy == "adaptive_text":
            if latency is not None:
                adaptive_points.append((float(latency), rate, "adaptive-2 (text cues)"))

    fig, (axis, axis2) = plt.subplots(1, 2, figsize=(11.6, 4.3), dpi=130,
                                      gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ---- 左：开放式轮的取舍
    if fixed_points:
        fixed_points.sort()
        axis.plot([p[0] for p in fixed_points], [p[1] for p in fixed_points],
                  marker="o", color="#1f77b4", label="fixed threshold", linewidth=1.6)
        for index, (x, y, label) in enumerate(fixed_points):
            last = index == len(fixed_points) - 1
            axis.annotate(f"{label}ms", (x, y), textcoords="offset points",
                          xytext=(-6, 34) if last else (4, 7),
                          ha="right" if last else "left", fontsize=7)

    palette = ["#d62728", "#2ca02c"]
    for index, (x, y, label) in enumerate(adaptive_points):
        # 两个自适应点几乎重合：横向轻抖只为可读性，文字合成一条避免叠字
        x_display = x + (-18 if index == 0 else 18)
        axis.scatter([x_display], [y], s=150, marker="*", zorder=5,
                     color=palette[index % len(palette)], label=label)

    if adaptive_points:
        xs = [point[0] for point in adaptive_points]
        ys = [point[1] for point in adaptive_points]
        axis.annotate(f"A1/A2: {xs[0]:.0f} ms, {ys[0]:.0f}%",
                      (min(xs) - 18, ys[0]), textcoords="offset points",
                      xytext=(-10, 62), fontsize=7.5, ha="right",
                      arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.6))

    axis.set_xlabel("latency after true turn end (median, ms)  → slower")
    axis.set_ylabel("early cutoff rate (%)  → worse")
    axis.set_ylim(-6, 128)
    axis.set_xlim(200, 1420)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    axis.set_title("open turns with mid-sentence pause (n=6): cut vs wait", fontsize=10)
    axis.legend(fontsize=7.5, loc="center right")

    # ---- 右：同样的截断水平下，短答要等多久
    show = ["fixed:1200", "fixed:1500", "adaptive_question", "adaptive_text"]
    names, latencies, colors = [], [], []
    for strategy in show:
        payload = summary["strategies"].get(strategy)
        if not payload:
            continue
        short = payload.get("pooled", {}).get("short", {})
        if not short.get("triggered") or short.get("latency_median_ms") is None:
            continue
        quality = (payload["by_group"].get(panel_group) or {}).get("early_rate")
        quality_text = "—" if quality is None else f"{quality*100:.0f}%"
        names.append(f"{strategy.replace('fixed:', 'fixed ').replace('adaptive_', 'adaptive-')}\n(open-turn cut {quality_text})")
        latencies.append(float(short["latency_median_ms"]))
        colors.append("#2ca02c" if strategy.startswith("adaptive") else "#1f77b4")

    positions = range(len(names))
    bars = axis2.bar(positions, latencies, color=colors, width=0.6)
    axis2.set_xticks(list(positions))
    axis2.set_xticklabels(names, fontsize=7.5)
    axis2.set_ylabel("short-answer latency (median, ms)  → slower")
    axis2.set_title("short answers: what you pay for the same safety", fontsize=10)
    axis2.set_ylim(0, max(latencies + [1]) * 1.18)
    axis2.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    for bar, value in zip(bars, latencies):
        axis2.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, value),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def dump_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    paths.ensure_dir(path.parent)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
