# -*- coding: utf-8 -*-
"""生成实践报告用的插图（结构图、状态机、时序、对照结果）。

只用于写报告，不属于 D 模块运行依赖；需要 matplotlib 与中文字体。

    python tools/make_figures.py --out ../01_D实践报告/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

EDGE = "#3f5f8a"
C_D = "#dbe9f7"
C_OTHER = "#f4f4f4"
C_DB = "#e6f2e6"
C_END = "#fbe3e3"
BOXSTYLE = "round,pad=0.02,rounding_size=0.08"


def _box(ax, x, y, w, h, text, color=C_OTHER, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=BOXSTYLE,
                                facecolor=color, edgecolor=EDGE, linewidth=1.1))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.45)


def _arrow(ax, start, end, rad=0.0, color="#2f2f2f", style="-|>", lw=1.1,
           dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle="arc3,rad=%s" % rad))


def _num(ax, x, y, text, fontsize=10):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#8a3b00",
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#d9a066",
                      linewidth=0.8))


def figure_architecture(path):
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # 第一行：采集 → 识别 → 理解 → 对话 → 合成
    _box(ax, 0.20, 3.20, 1.15, 0.85, "麦克风\n16 kHz")
    _box(ax, 1.75, 3.20, 1.30, 0.85, "A\n语音识别")
    _box(ax, 3.45, 3.20, 1.35, 0.85, "C\n口语理解 + FAQ")
    _box(ax, 5.20, 3.15, 1.70, 0.95, "D\n多轮对话管理\n本地登记", C_D, 10.5)
    _box(ax, 7.30, 3.20, 1.35, 0.85, "E\n合成与播报")
    _box(ax, 9.05, 3.20, 1.15, 0.85, "扬声器")

    # 第二行：端点检测、本地库
    _box(ax, 1.75, 1.55, 1.30, 0.80, "B\n端点检测")
    _box(ax, 5.20, 1.55, 1.70, 0.85, "SQLite\nregistrations\n/ dialog_log", C_DB, 9)

    # 主链
    _arrow(ax, (1.35, 3.62), (1.75, 3.62))
    _num(ax, 1.55, 3.90, "①")
    _arrow(ax, (3.05, 3.62), (3.45, 3.62))
    _num(ax, 3.25, 3.90, "②")
    _arrow(ax, (4.80, 3.62), (5.20, 3.62))
    _num(ax, 5.00, 3.90, "③")
    _arrow(ax, (6.90, 3.62), (7.30, 3.62))
    _num(ax, 7.10, 3.90, "④")
    _arrow(ax, (8.65, 3.62), (9.05, 3.62))
    _num(ax, 8.85, 3.90, "⑤")

    # 端点链路
    _arrow(ax, (2.25, 3.20), (2.25, 2.35))
    _num(ax, 2.02, 2.78, "⑥", 9)
    _arrow(ax, (2.58, 2.35), (2.58, 3.20))
    _num(ax, 2.82, 2.78, "⑦", 9)

    # 本地库
    _arrow(ax, (6.05, 3.15), (6.05, 2.40), style="<|-|>")
    _num(ax, 6.28, 2.78, "⑧", 9)

    # 反向：下一轮上下文、播报结果
    _arrow(ax, (5.35, 4.10), (4.45, 4.10), rad=0.55, color="#8a5a00", dashed=True)
    _num(ax, 4.90, 4.62, "⑨")
    _arrow(ax, (7.90, 4.10), (6.40, 4.10), rad=0.45, color="#1f6b3a", dashed=True)
    _num(ax, 7.15, 4.62, "⑩")

    legend = [
        "① 音频", "② 最终转写", "③ 意图/字段 NLU_RESULT", "④ 回复文本 REPLY_REQUEST",
        "⑤ 播报", "⑥ 流式进度 ASR_TICK", "⑦ 结束请求 TURN_STOP", "⑧ 写库 / 读库",
        "⑨ 下一轮上下文 TURN_CONTEXT（D→A/B/C）", "⑩ 播报结果 PLAYBACK_RESULT（E→D）",
    ]
    for i, text in enumerate(legend):
        col = i // 5
        row = i % 5
        ax.text(0.25 + col * 5.4, 1.05 - row * 0.20, text, fontsize=8.6,
                color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_state(path):
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    _box(ax, 0.25, 2.85, 1.45, 0.85, "idle\n待机", "#ffffff")
    _box(ax, 2.70, 2.85, 1.75, 0.85, "collecting\n字段收集", C_D)
    _box(ax, 5.40, 2.85, 1.80, 0.85, "confirming\n等待确认", C_D)
    _box(ax, 5.40, 1.00, 1.80, 0.80, "ended\n结束", C_END)
    _box(ax, 7.95, 1.00, 1.70, 0.80, "本地库\nSQLite", C_DB)

    _arrow(ax, (1.70, 3.27), (2.70, 3.27))
    ax.text(2.20, 3.50, "预约 / 给字段", fontsize=8.8, ha="center")
    _arrow(ax, (4.45, 3.27), (5.40, 3.27))
    ax.text(4.92, 3.50, "字段齐全", fontsize=8.8, ha="center")
    _arrow(ax, (5.35, 3.70), (4.50, 3.70), rad=0.65, dashed=True)
    ax.text(4.92, 4.10, "改口 / 否认", fontsize=8.8, ha="center")

    _arrow(ax, (6.30, 2.85), (6.30, 1.80))
    ax.text(6.45, 2.32, "确认且快照有效", fontsize=8.8, ha="left")

    _arrow(ax, (3.95, 2.85), (5.40, 1.50), rad=-0.25)
    ax.text(3.35, 2.05, "取消 / 重问超限", fontsize=8.8, ha="left")

    _arrow(ax, (7.20, 1.40), (7.95, 1.40))
    ax.text(7.58, 1.58, "写一条", fontsize=8.8, ha="center")

    ax.text(0.30, 0.35,
            "confirming 写库条件：确认轮次一致 + 草稿版本一致 + 该回复已播完；"
            "任一条件不满足则按当前草稿重新复述，不写库。",
            fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_sequence(path):
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    lanes = [("用户", 0.95), ("C 口语理解", 3.00), ("D 对话管理", 5.20),
             ("E 播报", 7.30), ("SQLite", 9.00)]
    for name, x in lanes:
        _box(ax, x - 0.78, 4.15, 1.56, 0.50, name, "#eef2f7", 9.5)
        ax.plot([x, x], [0.35, 4.15], linestyle="--", linewidth=0.7, color="#9a9a9a")

    steps = [
        (0.95, 3.00, 3.62, "①“还是上次那位，下周四下午”"),
        (3.00, 5.20, 3.20, "②NLU_RESULT（provide_slot + history_ref）"),
        (5.20, 7.30, 2.78, "③REPLY_REQUEST（复述确认）"),
        (7.30, 5.20, 2.36, "④PLAYBACK_RESULT completed"),
        (0.95, 3.00, 1.94, "⑤“改成下周五”"),
        (3.00, 5.20, 1.52, "⑥NLU_RESULT（correct）"),
        (5.20, 7.30, 1.10, "⑦REPLY_REQUEST（新复述）"),
        (5.20, 9.00, 0.62, "⑧确认后写库"),
    ]
    for x1, x2, y, text in steps:
        _arrow(ax, (x1, y), (x2, y), color="#2f2f2f")
        ax.text((x1 + x2) / 2.0, y + 0.13, text, fontsize=8.2, ha="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_metrics(path, baseline, improved):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    labels = ["历史继承类\n平均轮数", "歧义类\n轮数", "歧义类\n重复询问"]
    base_vals = [baseline["inherit_turns"], baseline["ambiguous_turns"],
                 baseline["ambiguous_repeat"]]
    imp_vals = [improved["inherit_turns"], improved["ambiguous_turns"],
                improved["ambiguous_repeat"]]
    x = range(len(labels))
    width = 0.34
    ax = axes[0]
    ax.bar([i - width / 2 for i in x], base_vals, width, label="固定顺序基线",
           color="#b8c6d9")
    ax.bar([i + width / 2 for i in x], imp_vals, width, label="历史继承 + 按字段澄清",
           color="#3f5f8a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("次数 / 轮数")
    ax.set_title("(a) 交互成本（越小越好）", fontsize=10.5)
    ax.set_ylim(0, 4.8)
    ax.legend(fontsize=8.5, loc="upper right")
    for i, (b, m) in enumerate(zip(base_vals, imp_vals)):
        ax.text(i - width / 2, b + 0.08, "%g" % b, ha="center", fontsize=8.5)
        ax.text(i + width / 2, m + 0.08, "%g" % m, ha="center", fontsize=8.5)

    labels2 = ["登记成功率", "关键字段正确率"]
    base2 = [baseline["success_rate"], baseline["field_accuracy"]]
    imp2 = [improved["success_rate"], improved["field_accuracy"]]
    x2 = range(len(labels2))
    ax = axes[1]
    ax.bar([i - width / 2 for i in x2], base2, width, color="#b8c6d9")
    ax.bar([i + width / 2 for i in x2], imp2, width, color="#3f5f8a")
    ax.set_xticks(list(x2))
    ax.set_xticklabels(labels2, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("(b) 结果正确性（越高越好）", fontsize=10.5)
    for i, (b, m) in enumerate(zip(base2, imp2)):
        ax.text(i - width / 2, b + 1.5, "%.1f" % b, ha="center", fontsize=8.5)
        ax.text(i + width / 2, m + 1.5, "%.1f" % m, ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                      "..", "figures"))
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    figure_architecture(os.path.join(out, "fig1_architecture.png"))
    figure_state(os.path.join(out, "fig2_state_machine.png"))
    figure_sequence(os.path.join(out, "fig4_sequence.png"))
    figure_metrics(os.path.join(out, "fig3_metrics.png"), {
        "inherit_turns": 3.0, "ambiguous_turns": 4.0, "ambiguous_repeat": 1,
        "success_rate": 92.9, "field_accuracy": 98.2,
    }, {
        "inherit_turns": 2.0, "ambiguous_turns": 3.0, "ambiguous_repeat": 0,
        "success_rate": 100.0, "field_accuracy": 100.0,
    })
    print("插图已生成：%s" % out)


if __name__ == "__main__":
    main()
