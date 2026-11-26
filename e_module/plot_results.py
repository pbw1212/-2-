"""Generate 4 publication-grade defense figures from verified experiment JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
from scipy import signal
from scipy.io import wavfile

from e_module.config import OUTPUTS_DIR


def _setup_cjk_font() -> None:
    """Configure an available CJK font on macOS/Linux to prevent tofu boxes."""
    preferred = ["PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred if name in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300


def plot_fig1_attention_alignment(fig_dir: Path) -> Path:
    """Plot 4-way Attention alignment comparison on short_prompt & multi_field_confirm."""
    align_dir = OUTPUTS_DIR / "alignments"
    modes = [
        ("basic_pinyin", "① basic_pinyin (无标点/无韵律，失控)"),
        ("tn_only", "② tn_only (有TN标点，缺#1/#3)"),
        ("tn_prosody", "③ tn_prosody (jieba规则#1/#3)"),
        ("tn_psp_ec", "④ tn_psp_ec (BiGRU-CRF + EC-Viterbi)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    sample_specs = [
        ("short_prompt", "短句: 请问您想约哪天？"),
        ("multi_field_confirm", "多字段确认第1段 / 完整对比"),
    ]

    for row, (sid, row_label) in enumerate(sample_specs):
        for col, (m_key, m_title) in enumerate(modes):
            ax = axes[row, col]
            p = align_dir / f"{sid}_{m_key}.json"
            if not p.exists():
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            # Concatenate per-segment alignments diagonally for visualization
            mats = [np.asarray(seg["alignment"], dtype=np.float32) for seg in data]
            total_in = sum(m.shape[0] for m in mats)
            total_dec = sum(m.shape[1] for m in mats)
            canvas = np.zeros((total_in, total_dec), dtype=np.float32)
            r_off, c_off = 0, 0
            for idx_seg, mat in enumerate(mats):
                ti, td = mat.shape
                canvas[r_off : r_off + ti, c_off : c_off + td] = mat
                # Draw Mel spans if present (on tn_psp_ec)
                if m_key == "tn_psp_ec":
                    for ms, me, lbl in data[idx_seg].get("mel_spans", []):
                        ds, de = ms / 3.0, me / 3.0
                        rect = patches.Rectangle(
                            (c_off + ds, r_off),
                            max(1.0, de - ds),
                            ti,
                            linewidth=1.2,
                            edgecolor="#ffdd57",
                            facecolor="none",
                            linestyle="--",
                        )
                        ax.add_patch(rect)
                r_off += ti
                c_off += td

            ax.imshow(canvas, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
            if row == 0:
                ax.set_title(m_title, fontsize=9.5, pad=6)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nEncoder Tokens", fontsize=8.5)
            ax.set_xlabel(f"Decoder Steps ({total_dec})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("图1  Tacotron2 Attention 对齐热力图四联对照（黄色虚线框为对齐定位的医疗关键字段）", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = fig_dir / "fig1_attention_alignment.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_fig2_prosody_f1(fig_dir: Path) -> Path:
    """Plot 1000-utterance Biaobei holdout F1 comparison and clinic entity integrity."""
    exp1 = json.loads((OUTPUTS_DIR / "exp1_prosody_eval.json").read_text(encoding="utf-8"))
    comp = exp1["test_set_comparison"]
    clinic = exp1["clinic_entity_integrity"]

    methods = [
        ("no_prosody", "无韵律"),
        ("punct_only", "仅标点#3"),
        ("fixed_stride_2", "双音节固定步长"),
        ("syllable_bigru_crf", "BiGRU-CRF (本文)"),
    ]
    labels = [m[1] for m in methods]
    f1_macro = [comp[m[0]]["macro_f1"] for m in methods]
    f1_p1 = [comp[m[0]]["per_class"]["#1"]["f1"] for m in methods]
    f1_p3 = [comp[m[0]]["per_class"]["#3"]["f1"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [2.2, 1.1]})
    x = np.arange(len(labels))
    w = 0.25

    b1 = ax1.bar(x - w, f1_p1, width=w, label="#1 韵律词 F1", color="#4e79a7")
    b2 = ax1.bar(x, f1_p3, width=w, label="#3 韵律短语 F1", color="#59a14f")
    b3 = ax1.bar(x + w, f1_macro, width=w, label="Macro-F1", color="#f28e2b")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("F1 分数", fontsize=10)
    ax1.set_title("1,000 句标贝留出测试集韵律边界预测 F1", fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Clinic entity integrity rate (Unconstrained BiGRU-CRF vs EC-Viterbi)
    tot_ent = max(clinic["total_multi_syllable_entities"], 1)
    plain_ok = (tot_ent - clinic["unconstrained_internal_#3_splits"]) / tot_ent * 100.0
    ec_ok = clinic["ec_viterbi_entity_integrity_rate"] * 100.0
    ax2.bar(["无约束\nBiGRU-CRF", "EC-Viterbi\n实体约束 (本文)"], [plain_ok, ec_ok], color=["#e15759", "#59a14f"], width=0.5)
    ax2.text(0, plain_ok + 1.5, f"{plain_ok:.1f}%\n(拆散{clinic['unconstrained_internal_#3_splits']}处)", ha="center", va="bottom", fontsize=8.5)
    ax2.text(1, ec_ok + 1.5, f"{ec_ok:.1f}%\n(拆散{clinic['ec_viterbi_internal_#3_splits']}处)", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("医疗关键实体完整率 (%)", fontsize=10)
    ax2.set_title(f"12 句复诊测试集关键字段完整性 (n={tot_ent})", fontsize=10.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("图2  数据驱动中文韵律模型（BiGRU-CRF）与医疗实体约束维特比解码（EC-Viterbi）评测", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / "fig2_prosody_f1.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_fig3_vocoder_quant_ttfa(fig_dir: Path) -> Path:
    """Plot 99-utterance MCD vs speed/size tradeoff + whole vs pipelined TTFA."""
    obj = json.loads((OUTPUTS_DIR / "exp2_4_objective_report.json").read_text(encoding="utf-8"))
    mcd_cfgs = obj["exp2a_biaobei_mcd"]["configs"]
    pipe_rows = obj["exp4_pipelined_ttfa"]["samples"]

    cfg_order = [
        ("fp32_gl50", "FP32\nGL-50"),
        ("fp32_gl25", "FP32\nGL-25"),
        ("fp32_fastgl25", "FP32\nFastGL-25"),
        ("int8_naive_fastgl25", "INT8-Naive\nFastGL-25"),
        ("int8_lstm_converted_fastgl25", "INT8-LSTM\nFastGL-25"),
    ]
    names = [c[1] for c in cfg_order]
    mcds = [mcd_cfgs[c[0]]["mcd_mean_db"] for c in cfg_order]
    lats = [mcd_cfgs[c[0]]["synth_ms_median"] for c in cfg_order]
    sizes = [mcd_cfgs[c[0]]["model_size_mb"] for c in cfg_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: Dual-axis latency bar (bottom) + MCD line (top) with clean vertical separation
    x = np.arange(len(names))
    bars = ax1.bar(x, lats, width=0.5, color=["#9c755f", "#bab0ac", "#4e79a7", "#e15759", "#59a14f"], label="中位合成耗时 (ms)")
    for b, sz in zip(bars, sizes):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 12, f"{h:.0f}ms\n({sz:.0f}MB)", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8.5)
    ax1.set_ylabel("中位耗时 (ms)", fontsize=9.5)
    ax1.set_ylim(0, max(lats) * 1.55)
    ax1.set_title("99 句标贝真值集：耗时 / 体积 / DTW-MCD 保真度", fontsize=10)

    ax1_r = ax1.twinx()
    ax1_r.plot(x, mcds, color="#d62728", marker="o", linewidth=2.0, label="DTW-MCD (dB ↓)")
    for xi, m_val in zip(x, mcds):
        ax1_r.text(xi, m_val + 0.06, f"{m_val:.2f}", color="#d62728", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1_r.set_ylabel("DTW-MCD vs 真值 (dB，越低越好)", color="#d62728", fontsize=9.5)
    ax1_r.set_ylim(min(mcds) - 1.4, max(mcds) + 0.25)

    # Right: Whole vs Chunk-Pipelined TTFA
    sids = [r["sample_id"] for r in pipe_rows]
    w_fp32 = [r["whole_fp32_ttfa_ms"] for r in pipe_rows]
    w_int8 = [r["whole_int8_ttfa_ms"] for r in pipe_rows]
    p_int8 = [r["pipelined_int8_ttfa_ms"] for r in pipe_rows]

    xp = np.arange(len(sids))
    wp = 0.25
    b_w1 = ax2.bar(xp - wp, w_fp32, width=wp, label="整句 FP32+FastGL", color="#4e79a7")
    b_w2 = ax2.bar(xp, w_int8, width=wp, label="整句 INT8+FastGL", color="#f28e2b")
    b_p = ax2.bar(xp + wp, p_int8, width=wp, label="分句流水线先播 (本文)", color="#59a14f")
    for b_grp in (b_w1, b_w2, b_p):
        for b in b_grp:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 15, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(xp)
    ax2.set_xticklabels(["短提示\n(1段)", "多字段确认\n(3段)", "长FAQ\n(4段)"], fontsize=9)
    ax2.set_ylabel("首音开播延迟 TTFA (ms)", fontsize=9.5)
    ax2.set_ylim(0, max(w_fp32) * 1.22)
    ax2.set_title("整句合成 vs 分句流水线首句先播 TTFA 对比", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("图3  声码器加速、LSTMCellAsLSTM INT8 量化保真度与流水线 TTFA 评测", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / "fig3_vocoder_quant_ttfa.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_fig4_clear_speech_acoustics(fig_dir: Path) -> Path:
    """Plot Clear Speech waveform stretching/pauses and 1.5-4 kHz PSD comparison."""
    demo_clean = OUTPUTS_DIR / "clear_speech_demo" / "clean"
    std_path = demo_clean / "standard" / "multi_field_confirm.wav"
    cs_path = demo_clean / "clear_speech_full" / "multi_field_confirm.wav"

    sr_s, d_s = wavfile.read(str(std_path))
    sr_c, d_c = wavfile.read(str(cs_path))
    y_std = d_s.astype(np.float32) / 32768.0
    y_cs = d_c.astype(np.float32) / 32768.0

    t_std = np.arange(len(y_std)) / sr_s
    t_cs = np.arange(len(y_cs)) / sr_c

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: Time-domain waveform showing calibrated pauses & local entity stretching
    ax1.plot(t_std, y_std + 0.9, color="#4e79a7", linewidth=0.5, label=f"标准合成 ({len(y_std)/sr_s:.2f}s)")
    ax1.plot(t_cs, y_cs - 0.9, color="#59a14f", linewidth=0.5, label=f"老年清晰语调控 ({len(y_cs)/sr_c:.2f}s)")
    ax1.set_yticks([-0.9, 0.9])
    ax1.set_yticklabels(["Clear Speech\n(保调降速+精调停顿)", "标准合成\n(无静音裁剪/无EQ)"], fontsize=8.5)
    ax1.set_xlabel("时间 (s)", fontsize=9.5)
    ax1.set_title("多字段确认句时域波形对比（精准 280ms 停顿 + 实体放慢）", fontsize=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    # Right: Welch PSD highlighting 1.5 - 4.0 kHz consonant clarity band
    f_s, psd_s = signal.welch(y_std, fs=sr_s, nperseg=2048)
    f_c, psd_c = signal.welch(y_cs, fs=sr_c, nperseg=2048)
    psd_s_db = 10.0 * np.log10(psd_s + 1e-12)
    psd_c_db = 10.0 * np.log10(psd_c + 1e-12)

    mask = f_s <= 6000.0
    ax2.axvspan(1500.0, 4000.0, color="#ffe8a1", alpha=0.45, label="辅音辨识关键区 (1.5–4.0 kHz)")
    ax2.plot(f_s[mask], psd_s_db[mask], color="#4e79a7", linewidth=1.6, label="标准合成 PSD")
    ax2.plot(f_c[mask], psd_c_db[mask], color="#d62728", linewidth=1.6, label="Clear Speech EQ PSD (+2.8dB)")
    ax2.set_xlabel("频率 (Hz)", fontsize=9.5)
    ax2.set_ylabel("功率谱密度 (dB/Hz)", fontsize=9.5)
    ax2.set_title("频域对比：1.5–4.0 kHz 清晰度带能量增强", fontsize=10)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("图4  面向老年听众的 Clear Speech 声学调控前后时频域对比", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / "fig4_clear_speech_acoustics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def generate_all_figures() -> list[Path]:
    _setup_cjk_font()
    fig_dir = OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_fig1_attention_alignment(fig_dir),
        plot_fig2_prosody_f1(fig_dir),
        plot_fig3_vocoder_quant_ttfa(fig_dir),
        plot_fig4_clear_speech_acoustics(fig_dir),
    ]
    for p in paths:
        print("Generated:", p)
    return paths


if __name__ == "__main__":
    generate_all_figures()
