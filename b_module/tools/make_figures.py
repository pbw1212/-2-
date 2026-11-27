"""汇报用配图生成（`python -m b_module.tools.make_figures`）。

目前生成一张图：

``results/coda_profile.png`` —— 尾音剖面示意，用于讲"为什么 −45 dBFS 会裁掉字尾"。
横轴是轮末附近的毫秒数，纵轴是帧功率；两条阈值线之间被阴影标出的区间，就是 v1 版本
按 −45 dBFS 硬裁时丢掉的部分，v2 改成 −60 dBFS 后保留了下来。

数据用真实的 ``test_09``（尾音衰减 180 ms，是 6 条样本里最有代表性的）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import annotate, dataset, paths, vad


def plot_coda_profile(
    case_id: str = "test_09",
    *,
    output: str | Path | None = None,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    context_ms: float = 600.0,
    config: dict | None = None,
) -> Path:
    """画轮末附近的帧功率剖面，标出 −45 / −60 两条阈值与被裁掉的区间。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = config or paths.load_config()
    output = Path(output) if output else paths.RESULTS_DIR / "coda_profile.png"
    frame_ms = float(config["audio"]["frame_ms"])
    detect_dbfs = float(config["audio"]["detect_dbfs"])
    annotate_dbfs = float(config["audio"]["annotate_dbfs"])

    specs = {spec.case_id: spec for spec in dataset.load_cases(a_root)}
    spec = specs[case_id]
    case_audio = dataset.build_case_audio(spec, a_root=a_root, config=config, use_cache=True)
    record = annotate.annotate_case(spec, case_audio, config)

    energies = vad.frame_energies_db(case_audio.samples, case_audio.sample_rate, frame_ms)
    times = np.arange(energies.size) * frame_ms
    end_ms = float(record["turn_end_ms"])

    # 注意：阈值边界要在**完整数组**上算，再用绝对时间换算成相对轮末的偏移
    voiced45_all = np.flatnonzero(energies >= detect_dbfs)
    voiced60_all = np.flatnonzero(energies >= annotate_dbfs)
    end45 = (voiced45_all[-1] + 1) * frame_ms - end_ms if voiced45_all.size else None
    end60 = (voiced60_all[-1] + 1) * frame_ms - end_ms if voiced60_all.size else None

    keep = (times >= end_ms - context_ms) & (times <= end_ms + 200.0)
    times = times[keep]
    energies = energies[keep]
    relative = times - end_ms

    fig, axis = plt.subplots(figsize=(7.4, 3.2), dpi=150)
    axis.plot(relative, energies, color="#1f77b4", linewidth=1.1)
    axis.fill_between(relative, -130, energies, color="#1f77b4", alpha=0.10)

    axis.axhline(detect_dbfs, color="#d62728", linestyle="--", linewidth=1.0,
                 label=f"detect threshold {detect_dbfs:.0f} dBFS")
    axis.axhline(annotate_dbfs, color="#2ca02c", linestyle=":", linewidth=1.1,
                 label=f"trim / truth threshold {annotate_dbfs:.0f} dBFS")

    if end45 is not None and end60 is not None and end60 > end45:
        axis.axvspan(end45, end60, color="#d62728", alpha=0.12)
        for x, color, text, y in ((end45, "#d62728", "v1 end", -88),
                                  (end60, "#2ca02c", "v2 end", -88)):
            axis.axvline(x, color=color, linewidth=1.0, linestyle="--", alpha=0.8)
            axis.annotate(text, (x, y), xytext=(3, 0), textcoords="offset points",
                          color=color, fontsize=7.5, va="bottom")
        # 双箭头标出 v1 丢掉的那段
        axis.annotate("", xy=(end45, -68), xytext=(end60, -68),
                      arrowprops=dict(arrowstyle="<->", color="#d62728", linewidth=1.0))
        axis.annotate(f"{abs(end60 - end45):.0f} ms lost by v1 hard trim",
                      ((end45 + end60) / 2.0, -68), xytext=(0, 7),
                      textcoords="offset points", ha="center", fontsize=7.5,
                      color="#d62728")

    axis.set_xlim(relative.min(), relative.max())
    axis.set_ylim(-95, 5)
    axis.set_xlabel("time relative to turn end (ms)")
    axis.set_ylabel("frame energy (dBFS)")
    axis.set_title(f"coda profile · {case_id} · v1 hard-trimmed at -45 dBFS, v2 keeps to -60", fontsize=9)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    axis.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    paths.ensure_dir(output.parent)
    fig.savefig(output)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m b_module.tools.make_figures",
                                     description="生成汇报用配图")
    parser.add_argument("--case", default="test_09", help="用哪条样本画尾音剖面")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    path = plot_coda_profile(args.case, output=args.out)
    print(f"已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
