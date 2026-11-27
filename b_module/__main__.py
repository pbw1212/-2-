"""B 模块命令行入口。

阶段一可用命令：

    python -m b_module manifest                       # S4：生成 data/manifest.json
    python -m b_module annotate                       # S5：重建音频 + 真值 + 能量曲线
    python -m b_module verify dev_01 --set-end dev_01=1540
    python -m b_module summary                        # 打印标注摘要
    python -m b_module replay  --case dev_01 --strategy fixed:600
    python -m b_module replay  --case dev_01 --strategy adaptive_text --realtime
    python -m b_module scan                           # S11/S12：全样本对照 + 表 + 主图

在仓库根目录（``b_module`` 的父目录）执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (annotate, context as context_mod, dataset, listen, metrics, paths, replay,
               strategies)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


# --------------------------------------------------------------------- manifest
def _cmd_manifest(args: argparse.Namespace) -> int:
    config = paths.load_config(args.config)
    specs = dataset.load_cases(args.a_root)
    payload = dataset.build_manifest(specs, a_root=args.a_root, config=config, output=args.out)
    print(f"样例数：{payload['case_count']}  分组：{payload['group_counts']}")
    print(f"已写出：{Path(args.out or paths.MANIFEST_PATH)}")
    return 0


# --------------------------------------------------------------------- annotate
def _cmd_annotate(args: argparse.Namespace) -> int:
    config = paths.load_config(args.config)
    specs = dataset.load_cases(args.a_root)
    data = annotate.build_annotations(
        specs,
        a_root=args.a_root,
        config=config,
        output=args.out,
        plot_dir=args.plot_dir,
        make_plots=not args.no_plots,
        use_cache=not args.no_cache,
        reset_verified=args.reset_verified,
    )
    print(f"标注完成：{data['case_count']} 条（人工确认 {data['verified_count']} 条）")
    _print_annotation_rows(annotate.summary_rows(data))
    print(f"\n已写出：{Path(args.out or paths.ANNOTATIONS_PATH)}")
    if not args.no_plots:
        print(f"曲线目录：{Path(args.plot_dir or (paths.RESULTS_DIR / 'annotation'))}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    data = annotate.apply_manual_updates(
        args.out or paths.ANNOTATIONS_PATH,
        verify=list(args.cases or []),
        set_end=list(args.set_end or []),
        note=args.note,
    )
    print(f"人工确认后：{data['verified_count']}/{data['case_count']} 条已确认")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    output = Path(args.out or paths.ANNOTATIONS_PATH)
    if not output.is_file():
        print(f"尚未生成标注文件：{output}", file=sys.stderr)
        return 1
    data = json.loads(output.read_text(encoding="utf-8"))
    _print_annotation_rows(annotate.summary_rows(data))
    return 0


# ----------------------------------------------------------------------- replay
def _cmd_replay(args: argparse.Namespace) -> int:
    config = paths.load_config(args.config)
    ctx = None
    if args.context_log:
        contexts = context_mod.load_contexts_from_jsonl(args.context_log)
        if not contexts:
            print(f"{args.context_log} 里没有 TURN_CONTEXT", file=sys.stderr)
            return 1
        ctx = context_mod.first_context(contexts, args.turn_id) or contexts[0]
        print(f"用 D 的真实上下文：turn_id={ctx.turn_id} question_type={ctx.question_type} "
              f"expected_slots={ctx.expected_slots}")

    result = replay.replay_turn(
        args.case,
        args.strategy,
        config=config,
        a_root=args.a_root,
        context=ctx,
        turn_id=args.turn_id,
        realtime=args.realtime,
        trace_path=args.trace,
        trace_ticks=args.trace_ticks,
    )
    if args.trace:
        print(f"trace：{Path(args.trace)}")
    return 0 if result.stop is not None else 2


# ------------------------------------------------------------------------- scan
def _cmd_scan(args: argparse.Namespace) -> int:
    config = paths.load_config(args.config)
    specs = dataset.load_cases(args.a_root)
    known = [spec.case_id for spec in specs]

    case_ids = known
    if args.cases and args.cases.lower() != "all":
        case_ids = [item.strip() for item in args.cases.split(",") if item.strip()]
        missing = [item for item in case_ids if item not in known]
        if missing:
            print(f"未知样本：{missing}", file=sys.stderr)
            return 1

    if args.strategies:
        strategy_specs = [item.strip() for item in args.strategies.split(",") if item.strip()]
    else:
        strategy_specs = [f"fixed:{int(item)}" for item in config["strategies"]["fixed"]["thresholds_ms"]]
        strategy_specs += ["adaptive_question", "adaptive_text"]

    tolerance = float(config["evaluation"]["tolerance_ms"])
    results = replay.replay_many(case_ids, strategy_specs, config=config, a_root=args.a_root)
    outcomes = [metrics.evaluate(item, tolerance) for item in results]
    summary = metrics.summarize(outcomes)

    out_dir = Path(args.out_dir or paths.RESULTS_DIR)
    paths.ensure_dir(out_dir)
    csv_path = metrics.write_csv(outcomes, out_dir / "scan_details.csv")
    summary_path = metrics.dump_summary(summary, out_dir / "metrics.json")

    group_counts: dict[str, int] = {}
    for spec in specs:
        if spec.case_id in case_ids:
            group_counts[spec.group] = group_counts.get(spec.group, 0) + 1
    meta = {
        "case_count": len(case_ids),
        "group_text": "，".join(f"{key} {value}" for key, value in sorted(group_counts.items())),
        "tolerance_ms": tolerance,
    }
    md_path = out_dir / "metrics.md"
    md_path.write_text(metrics.to_markdown(summary, meta), encoding="utf-8")

    figure_path = None
    if not args.no_plot:
        figure_path = metrics.plot_summary(summary, out_dir / "figure_tradeoff.png",
                                          group=args.group)

    _print_scan_table(summary)
    print(f"\n明细：{csv_path}")
    print(f"汇总：{summary_path}")
    print(f"报告：{md_path}")
    if figure_path:
        print(f"主图：{figure_path}（group={args.group}）")
    return 0


# ------------------------------------------------------------------------ listen
def _cmd_listen(args: argparse.Namespace) -> int:
    config = paths.load_config(args.config)
    case_ids = None
    if args.cases and args.cases.lower() != "all":
        case_ids = [item.strip() for item in args.cases.split(",") if item.strip()]
    result = listen.prepare(
        case_ids,
        tail_ms=args.tail_ms,
        head_ms=args.head_ms,
        post_ms=args.post_ms,
        coda_ms=args.coda_ms,
        all_cases=args.all or (args.cases or "").lower() == "all",
        a_root=args.a_root,
        config=config,
    )
    print(f"已生成 {len(result['rows'])} 条听审材料")
    print(f"试听台（浏览器打开）：{result['index']}")
    print(f"听审表（记录用）：    {result['sheet']}")
    print("\n提示：轮末之后应为纯静音；若有声音，说明标注偏早。")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    target = Path(args.file) if args.file else (
        paths.CACHE_DIR / "listen" / f"{args.case}__{args.clip}.wav")
    return listen.play(target)


# ---------------------------------------------------------------------- 打印
def _print_annotation_rows(rows: list[dict]) -> None:
    header = (f"{'case':<12}{'group':<14}{'qtype':<10}{'sr':>6}{'dur_ms':>9}"
              f"{'end_ms':>9}{'pause_d':>9}{'pause_m':>9}{'reg':>5}{'ok':>4}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['case']:<12}{row['group']:<14}{row['qtype']:<10}{row['sr']:>6}"
            f"{row['dur_ms']:>9.0f}{row['end_ms']:>9.0f}{row['pause_design']:>9.0f}"
            f"{row['pause_meas']:>9.0f}{row['regions']:>5}{'Y' if row['verified'] else '-':>4}"
        )


def _print_scan_table(summary: dict) -> None:
    header = (f"{'strategy':<20}{'triggered':>10}{'early':>10}{'rate':>7}"
              f"{'lat_med':>9}{'lat_p90':>9}{'n':>4}{'no_stop':>9}")
    print(header)
    print("-" * len(header))
    for strategy, payload in summary["strategies"].items():
        overall = payload["overall"]
        rate = "—" if overall["early_rate"] is None else f"{overall['early_rate']*100:.0f}%"
        med = "—" if overall["latency_median_ms"] is None else f"{overall['latency_median_ms']:.0f}"
        p90 = "—" if overall["latency_p90_ms"] is None else f"{overall['latency_p90_ms']:.0f}"
        print(
            f"{strategy:<20}{overall['triggered']:>10}"
            f"{str(overall['early'])+'/'+str(overall['early_denominator']):>10}"
            f"{rate:>7}{med:>9}{p90:>9}{overall['latency_n']:>4}{overall['untriggered']:>9}"
        )


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m b_module",
                                     description="B 模块：自适应语音端点检测")
    parser.add_argument("--config", type=Path, default=None, help="自定义 endpoint.json 路径")
    parser.add_argument("--a-root", type=Path, default=paths.DEFAULT_A_ROOT, help="A 模块目录（只读）")
    parser.add_argument("--out", type=Path, default=None, help="输出 JSON 路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("manifest", help="S4：生成 data/manifest.json")

    annotate_parser = subparsers.add_parser("annotate", help="S5：重建音频并生成真值")
    annotate_parser.add_argument("--no-plots", action="store_true", help="不生成能量曲线图")
    annotate_parser.add_argument("--no-cache", action="store_true", help="忽略 .cache 重建音频")
    annotate_parser.add_argument("--plot-dir", type=Path, default=None, help="曲线输出目录")
    annotate_parser.add_argument("--reset-verified", action="store_true",
                                 help="清空人工确认标记（真值口径变更后必须用）")

    verify_parser = subparsers.add_parser("verify", help="人工确认标注")
    verify_parser.add_argument("cases", nargs="*", help="标记为已确认的 case_id")
    verify_parser.add_argument("--set-end", action="append", default=[], metavar="CASE=MS",
                               help="覆盖轮末时刻，例如 dev_01=1540")
    verify_parser.add_argument("--note", default=None, help="确认人或说明")

    subparsers.add_parser("summary", help="打印标注摘要表")

    replay_parser = subparsers.add_parser("replay", help="S10：回放一条样本，打印端点决策")
    replay_parser.add_argument("--case", required=True, help="样本 id，例如 dev_01")
    replay_parser.add_argument("--strategy", default="fixed:600",
                               help="fixed:<ms> / adaptive_question / adaptive_text")
    replay_parser.add_argument("--turn-id", type=int, default=1, help="组协议的整数轮次号")
    replay_parser.add_argument("--realtime", action="store_true", help="按真实时间回放（现场演示）")
    replay_parser.add_argument("--trace", type=Path, default=None, help="trace JSONL 输出路径")
    replay_parser.add_argument("--trace-ticks", action="store_true", help="trace 里同时写入每条 ASR_TICK")
    replay_parser.add_argument("--context-log", type=Path, default=None,
                               help="用 D 的回放日志里的真实 TURN_CONTEXT")

    scan_parser = subparsers.add_parser("scan", help="S11/S12：批量对照 + 指标表 + 主图")
    scan_parser.add_argument("--cases", default="all", help="逗号分隔的样本 id，默认全部")
    scan_parser.add_argument("--strategies", default=None,
                             help="逗号分隔的策略，默认固定 5 档 + 两档自适应")
    scan_parser.add_argument("--out-dir", type=Path, default=None, help="结果目录，默认 b_module/results")
    scan_parser.add_argument("--group", default="all",
                             help="主图口径：all（29 条总体，默认）或某个分组名")
    scan_parser.add_argument("--no-plot", action="store_true", help="不生成主图")

    listen_parser = subparsers.add_parser("listen", help="生成人工听审材料（试听片段 + 试听台）")
    listen_parser.add_argument("--cases", default=None, help="逗号分隔的样本 id；省略则用建议的 10 条")
    listen_parser.add_argument("--all", action="store_true", help="全部 29 条")
    listen_parser.add_argument("--tail-ms", type=float, default=listen.DEFAULT_TAIL_MS,
                               help="句尾片段长度，默认 1500 ms")
    listen_parser.add_argument("--head-ms", type=float, default=listen.DEFAULT_HEAD_MS,
                               help="轮末之后的检查片段长度，默认 800 ms")
    listen_parser.add_argument("--post-ms", type=float, default=listen.DEFAULT_POST_MS,
                               help="句尾片段跨过轮末的长度，默认 400 ms")
    listen_parser.add_argument("--coda-ms", type=float, default=listen.DEFAULT_CODA_MS,
                               help="边界特写片段在轮末前后各取多长，默认 400 ms")

    play_parser = subparsers.add_parser("play", help="播放某条听审片段")
    play_parser.add_argument("--case", default="dev_01")
    play_parser.add_argument("--clip", choices=("coda", "tail", "head", "full"), default="coda")
    play_parser.add_argument("--file", default=None, help="直接指定 wav 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "manifest": _cmd_manifest,
        "annotate": _cmd_annotate,
        "verify": _cmd_verify,
        "summary": _cmd_summary,
        "replay": _cmd_replay,
        "scan": _cmd_scan,
        "listen": _cmd_listen,
        "play": _cmd_play,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
