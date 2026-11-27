"""听审材料生成（人工确认标注用）。

人工"听"不出毫秒级时刻，但可以回答两个可判定的问题：

1. **字尾是否完整？** —— 取 ``[turn_end - 400, turn_end + 400]`` 的"边界特写"片段，
   标注轮末正好在片段**正中间**。听完整 800 ms：字尾应当自然收掉，中间不应出现
   "突然变哑"的切口。
2. **轮末是否偏早？** —— ``[turn_end, turn_end + head_ms]`` 必须**纯静音**；
   曲线图上紫线右侧不应还有绿色语音段。

> v2（2026-11-27）：初版把每个分段的尾音按 −45 dBFS 硬裁，实测平均丢掉 122 ms、
> 最多 220 ms 的渐弱尾音（听感是"最后一个字被吞掉飘出来的那一点"）。现在重建与真值
> 都改用 −60 dBFS，并把片段改成**跨过轮末**，避免"片段在边界处结束"本身造成的截断感。

生成物（都在本地，不入库）：

    b_module/.cache/listen/<case>__coda.wav   边界特写（轮末在正中间）
    b_module/.cache/listen/<case>__tail.wav   句尾 + 轮末之后一段
    b_module/.cache/listen/<case>__head.wav   轮末之后的片段（应为静音）
    b_module/.cache/listen/<case>__full.wav   整条
    b_module/results/listen/index.html         一页式试听台（含曲线图）
    b_module/results/listen/听审表.md          可打印/可记录的表格
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from . import annotate, audio, dataset, paths, vad

DEFAULT_TAIL_MS = 1500.0
DEFAULT_HEAD_MS = 800.0
DEFAULT_POST_MS = 400.0      # 句尾片段跨过轮末的长度（消除"片段在边界结束"的错觉）
DEFAULT_CODA_MS = 400.0      # 边界特写：轮末前后各 400 ms
LISTEN_DIR = paths.RESULTS_DIR / "listen"
CACHE_DIR = paths.CACHE_DIR / "listen"

# 建议优先听审的 10 条：6 条带句中停顿（最关键的对照）+ 各组代表
SUGGESTED = [
    "dev_01", "test_01", "test_02", "test_07", "test_09", "test_12",
    "smoke_short", "dev_02", "test_13", "dev_04",
]


def _slice(samples, sample_rate: int, start_ms: float, end_ms: float):
    start = max(int(round(sample_rate * start_ms / 1000.0)), 0)
    end = min(int(round(sample_rate * end_ms / 1000.0)), len(samples))
    if end <= start:
        return samples[0:0]
    return samples[start:end]


def prepare(
    case_ids: Sequence[str] | None = None,
    *,
    tail_ms: float = DEFAULT_TAIL_MS,
    head_ms: float = DEFAULT_HEAD_MS,
    post_ms: float = DEFAULT_POST_MS,
    coda_ms: float = DEFAULT_CODA_MS,
    all_cases: bool = False,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """生成试听片段与两页清单。"""
    config = config or paths.load_config()
    specs = {spec.case_id: spec for spec in dataset.load_cases(a_root)}

    if all_cases:
        selected = list(specs)
    elif case_ids:
        selected = [item for item in case_ids if item in specs]
        missing = [item for item in case_ids if item not in specs]
        if missing:
            raise KeyError(f"未知样本：{missing}")
    else:
        selected = [item for item in SUGGESTED if item in specs]

    paths.ensure_dir(CACHE_DIR)
    paths.ensure_dir(LISTEN_DIR)

    rows: List[Dict[str, Any]] = []
    for case_id in selected:
        spec = specs[case_id]
        case_audio = dataset.build_case_audio(spec, a_root=a_root, config=config, use_cache=True)
        record = annotate.annotate_case(spec, case_audio, config)
        end_ms = float(record["turn_end_ms"])
        sr = case_audio.sample_rate

        tail = _slice(case_audio.samples, sr, max(end_ms - tail_ms, 0.0), end_ms + post_ms)
        head = _slice(case_audio.samples, sr, end_ms, end_ms + head_ms)
        coda = _slice(case_audio.samples, sr, max(end_ms - coda_ms, 0.0), end_ms + coda_ms)
        tail_path = CACHE_DIR / f"{case_id}__tail.wav"
        head_path = CACHE_DIR / f"{case_id}__head.wav"
        coda_path = CACHE_DIR / f"{case_id}__coda.wav"
        full_path = CACHE_DIR / f"{case_id}__full.wav"
        audio.write_wav(tail_path, tail, sr)
        audio.write_wav(head_path, head, sr)
        audio.write_wav(coda_path, coda, sr)
        audio.write_wav(full_path, case_audio.samples, sr)

        head_peak_db = _peak_dbfs(head, sr, config)
        # 轮末前最后 100 ms 的峰值：若明显低于检测阈值，说明边界压在了很弱的尾音上，
        # 人耳可能觉得"还没说完"，需要重点听。
        tail_last100 = _slice(case_audio.samples, sr, max(end_ms - 100.0, 0.0), end_ms)
        rows.append({
            "case_id": case_id,
            "group": record["group"],
            "question_type": record["question_type"],
            "turn_end_ms": end_ms,
            "duration_ms": record["duration_ms"],
            "mid_pause_design_ms": record["mid_pause_design_ms"],
            "mid_pause_ms": record["mid_pause_ms"],
            "tail_silence_ms": record["tail_silence_ms"],
            "verified": bool(record.get("verified")),
            "tail_path": tail_path,
            "head_path": head_path,
            "coda_path": coda_path,
            "full_path": full_path,
            "head_peak_dbfs": head_peak_db,
            "tail_last100_peak_dbfs": _peak_dbfs(tail_last100, sr, config),
            "boundary_at_ms": min(tail_ms, end_ms),
            "coda_boundary_ms": coda.size * 1000.0 / sr / 2.0,
            "plot": Path("results") / "annotation" / f"{case_id}.png",
        })

    index = LISTEN_DIR / "index.html"
    index.write_text(_render_html(rows, tail_ms, head_ms, post_ms, coda_ms), encoding="utf-8")
    sheet = LISTEN_DIR / "听审表.md"
    sheet.write_text(_render_markdown(rows, tail_ms, head_ms, post_ms, coda_ms), encoding="utf-8")
    return {"rows": rows, "index": index, "sheet": sheet}


def _peak_dbfs(samples, sample_rate: int, config: Mapping[str, Any]) -> float | None:
    if samples.size == 0:
        return None
    energies = vad.frame_energies_db(samples, sample_rate, float(config["audio"]["frame_ms"]))
    return None if energies.size == 0 else float(energies.max())


def _render_html(rows: List[Dict[str, Any]], tail_ms: float, head_ms: float,
                 post_ms: float, coda_ms: float) -> str:
    parts = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        "<title>B 模块轮末标注 · 听审台</title>",
        "<style>",
        "body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;max-width:1100px;color:#222}",
        "h1{font-size:20px} h2{font-size:15px;margin-top:28px;border-top:2px solid #eee;padding-top:16px}",
        ".hint{background:#f6f8fa;border-left:4px solid #1f77b4;padding:10px 14px;font-size:13px;line-height:1.7}",
        ".row{display:flex;gap:18px;align-items:flex-start}",
        ".meta{font-size:13px;line-height:1.8;min-width:310px}",
        ".meta code{background:#f2f2f2;padding:1px 4px;border-radius:3px}",
        "audio{width:320px;display:block;margin:6px 0}",
        "img{width:620px;border:1px solid #ddd;border-radius:4px}",
        ".bad{color:#c00;font-weight:600} .ok{color:#2a7a2a;font-weight:600}",
        "</style></head><body>",
        "<h1>B 模块轮末标注 · 听审台（v2：片段跨过轮末）</h1>",
        '<div class="hint"><b>主判据是 ① 边界特写</b>：片段总长 '
        f'{coda_ms*2:.0f} ms，<b>标注轮末正好在正中间（第 {coda_ms:.0f} ms 处）</b>。'
        "听完整的 800 ms——字尾应当自然收掉；若在正中间听到\"突然变哑\"的硬切口，说明还在裁尾音。<br>"
        f"② 句尾片段跨过轮末（前 {tail_ms:.0f} ms + 后 {post_ms:.0f} ms），可对着进度条感受完整收尾。"
        f"③ 轮末之后 {head_ms:.0f} ms 应完全静音（有声音＝标早了）。"
        "④ 整条 + 曲线：紫色竖线右侧不应再有绿色语音段。<br>"
        "结论写进 <code>results/listen/听审表.md</code>，再用 "
        "<code>python -m b_module verify &lt;case...&gt; --note \"听审 姓名 日期\"</code> 记录。</div>",
    ]
    for row in rows:
        peak = row["head_peak_dbfs"]
        peak_text = "—" if peak is None else f"{peak:.1f} dBFS"
        flag = "" if peak is None or peak < -55 else ' <span class="bad">⚠ 可能有残留语音</span>'
        last100 = row["tail_last100_peak_dbfs"]
        last100_text = "—" if last100 is None else f"{last100:.1f} dBFS"
        weak = "" if last100 is None or last100 > -55 else ' <span class="bad">⚠ 尾音很弱，重点听</span>'
        case = html.escape(row["case_id"])
        parts.append("<h2>" + case +
                     (f' <span class="ok">已确认</span>' if row["verified"] else "") + "</h2>")
        parts.append('<div class="row">')
        parts.append('<div class="meta">')
        parts.append(f'<div>分组 <code>{row["group"]}</code> · 问题类型 <code>{row["question_type"]}</code></div>')
        parts.append(f'<div>标注轮末 <code>{row["turn_end_ms"]:.0f} ms</code> · 总时长 <code>{row["duration_ms"]:.0f} ms</code></div>')
        parts.append(f'<div>句中停顿 设计 <code>{row["mid_pause_design_ms"]:.0f}</code> / 实测 <code>{row["mid_pause_ms"]:.0f}</code> ms</div>')
        parts.append(f'<div>轮末之后峰值 <code>{peak_text}</code>{flag}</div>')
        parts.append(f'<div>轮末前 100 ms 峰值 <code>{last100_text}</code>{weak}</div>')
        parts.append("</div>")
        parts.append("<div>")
        parts.append(f'<div>① 边界特写（{coda_ms:.0f} ms 前 + {coda_ms:.0f} ms 后，轮末在正中间）—— 主判据</div>')
        parts.append(f'<audio controls preload="none" src="../../.cache/listen/{case}__coda.wav"></audio>')
        parts.append(f'<div>② 句尾 + 轮末后 {post_ms:.0f} ms（轮末在片段第 {tail_ms:.0f} ms 处）</div>')
        parts.append(f'<audio controls preload="none" src="../../.cache/listen/{case}__tail.wav"></audio>')
        parts.append(f'<div>③ 轮末之后 {head_ms:.0f} ms —— 应完全静音</div>')
        parts.append(f'<audio controls preload="none" src="../../.cache/listen/{case}__head.wav"></audio>')
        parts.append('<div>④ 整条（对照下面的曲线）</div>')
        parts.append(f'<audio controls preload="none" src="../../.cache/listen/{case}__full.wav"></audio>')
        parts.append(f'<img loading="lazy" src="../annotation/{case}.png" alt="energy">')
        parts.append("</div></div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_markdown(rows: List[Dict[str, Any]], tail_ms: float, head_ms: float,
                     post_ms: float, coda_ms: float) -> str:
    lines = [
        "# B 模块轮末标注 · 听审表（v2）",
        "",
        f"共 {len(rows)} 条。主判据是**边界特写**：{coda_ms*2:.0f} ms 片段，标注轮末在正中间。",
        "",
        "判定标准：",
        "",
        "1. **边界特写**：字尾自然收掉，正中间没有\"突然变哑\"的硬切口。",
        "2. **轮末之后必须完全静音**——有声音说明标早了，系统会砍断用户。",
        "3. **曲线图**里紫色竖线右侧不应再有绿色语音段。",
        "",
        "| # | 样本 | 分组 | 轮末(ms) | 停顿 设计/实测 | 轮末后峰值 | 轮末前100ms | 边界特写 | 句尾+后 | 轮末后 | 整条 | 我的结论 |",
        "| --: | :--- | :--- | --: | :--- | --: | --: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for index, row in enumerate(rows, start=1):
        peak = "—" if row["head_peak_dbfs"] is None else f"{row['head_peak_dbfs']:.1f} dBFS"
        last100 = ("—" if row["tail_last100_peak_dbfs"] is None
                   else f"{row['tail_last100_peak_dbfs']:.1f} dBFS")
        coda_rel = f"../../.cache/listen/{row['case_id']}__coda.wav"
        tail_rel = f"../../.cache/listen/{row['case_id']}__tail.wav"
        head_rel = f"../../.cache/listen/{row['case_id']}__head.wav"
        full_rel = f"../../.cache/listen/{row['case_id']}__full.wav"
        lines.append(
            f"| {index} | `{row['case_id']}` | {row['group']} | {row['turn_end_ms']:.0f} | "
            f"{row['mid_pause_design_ms']:.0f} / {row['mid_pause_ms']:.0f} | {peak} | {last100} | "
            f"[coda]({coda_rel}) | [tail]({tail_rel}) | [head]({head_rel}) | [full]({full_rel}) |  |"
        )
    lines += [
        "",
        "## 记录结论",
        "",
        "全部通过（不用改）：",
        "",
        "```powershell",
        "python -m b_module verify dev_01 test_02 --note \"听审 你的名字 2026-11-28\"",
        "```",
        "",
        "某条标早了/标晚了：用 `--set-end` 覆盖真值（毫秒），再重跑 `scan` 更新结果：",
        "",
        "```powershell",
        "python -m b_module verify --set-end dev_01=1540 --note \"听审 你的名字 2026-11-28\"",
        "python -m b_module scan",
        "```",
        "",
    ]
    return "\n".join(lines)


def play(path: str | Path) -> int:
    """用系统默认方式播放一个 wav（Windows 走 SoundPlayer，其余平台给提示）。"""
    path = Path(path).resolve()
    if not path.is_file():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 1
    if sys.platform.startswith("win"):
        command = [
            "powershell", "-NoProfile", "-Command",
            f"(New-Object Media.SoundPlayer '{path}').PlaySync()",
        ]
        return subprocess.call(command)
    print(f"请用系统播放器打开：{path}")
    return 0
