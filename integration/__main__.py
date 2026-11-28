"""CLI entry point for integrated offline follow-up assistant (mic / wav / text)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from e_module.audio_player import MockAudioPlayer
from integration.coordinator import DEFAULT_RUNTIME_DIR, FullSystemCoordinator, InProcessABackend
from integration.mic_runner import HalfDuplexMicrophone, enroll_demo_profiles


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m integration",
        description="离线复诊登记助手 - 全组 A+B+C+D+E+小模型 整机联调入口",
    )
    parser.add_argument("--mode", choices=("text", "wav", "mic"), default="text")
    parser.add_argument("--wavs", nargs="*", type=Path, default=[], help="--mode wav 时按顺序输入每轮音频")
    parser.add_argument("--script", nargs="*", default=[], help="--mode text 非交互脚本话术列表")
    parser.add_argument("--session-id", default="s01")
    parser.add_argument("--reference-date", default=None, help="会话参考日期 YYYY-MM-DD（默认取当天）")
    parser.add_argument("--profile", default=None, help="预设模拟档案 ID（如 p001），或开启声纹匹配")
    parser.add_argument("--identify", action="store_true", help="首轮开启 CAM++ 声纹匹配推荐档案")
    parser.add_argument("--llm", choices=("local", "off"), default="local", help="是否启用本地离线小模型")
    parser.add_argument("--no-polish", action="store_true", help="关闭小模型润色回复，仅用于 NLU 兜底与导诊")
    parser.add_argument("--tts", choices=("real", "mock"), default="real", help="E 合成后端（real/mock）")
    parser.add_argument("--player", choices=("auto", "mock"), default="auto", help="播放器（auto 真实播音 / mock 静音）")
    parser.add_argument("--b-strategy", default="adaptive_text", help="B 端点策略（adaptive_text/adaptive_question/fixed:900）")
    parser.add_argument("--c-retriever", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--vad-dbfs", type=float, default=None, help="覆盖自动底噪校准的 VAD 阈值")
    parser.add_argument("--no-greeting", action="store_true", help="跳过 turn_id=0 开场播报")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)

    mic: Optional[HalfDuplexMicrophone] = None
    a_backend: Optional[InProcessABackend] = None

    if args.mode in ("wav", "mic"):
        speaker_db = None
        if args.identify:
            speaker_db = enroll_demo_profiles(DEFAULT_RUNTIME_DIR / "speaker_profiles.sqlite3")
        vad_threshold = args.vad_dbfs if args.vad_dbfs is not None else -45.0
        a_backend = InProcessABackend(
            args.session_id,
            vad_dbfs=vad_threshold,
            enable_speaker=bool(args.identify),
            speaker_db_path=speaker_db,
        )

    if args.mode == "mic":
        mic = HalfDuplexMicrophone()
        mic.open()
        if args.vad_dbfs is None:
            calibrated = mic.calibrate_noise_floor()
            assert a_backend is not None
            a_backend.set_vad_threshold(calibrated)
            print(f"[MIC] 自动底噪校准 VAD 阈值：{calibrated} dBFS", flush=True)

    def _on_reply(action: str, text: str) -> None:
        print(f"助手[{action}]> {text}", flush=True)

    coord = FullSystemCoordinator(
        session_id=args.session_id,
        db_path=args.db,
        log_path=args.log,
        reference_date=args.reference_date,
        c_retriever=args.c_retriever,
        b_strategy=args.b_strategy,
        llm_enabled=(args.llm == "local"),
        polish_replies=(not args.no_polish),
        tts_backend=args.tts,
        player=(MockAudioPlayer() if args.player == "mock" else None),
        a_backend=a_backend,
        profile_id=args.profile,
        on_mic_pause=(mic.pause if mic else None),
        on_mic_flush_and_open=(mic.flush_and_open if mic else None),
        on_agent_reply=_on_reply,
    )

    try:
        opening = None if args.no_greeting else "您好，这里是离线复诊登记助手，请问您要预约复诊还是咨询流程？"
        coord.start(opening_greeting=opening)

        if args.mode == "text":
            if args.script:
                for line in args.script:
                    if coord.is_closed:
                        break
                    print(f"患者> {line}", flush=True)
                    coord.step_text(line)
            else:
                while not coord.is_closed:
                    try:
                        line = input("患者> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if line.lower() in ("quit", "exit"):
                        break
                    coord.step_text(line)

        elif args.mode == "wav":
            if not args.wavs:
                raise SystemExit("--mode wav 需要传入至少一个音频文件")
            for idx, wav_path in enumerate(args.wavs):
                if coord.is_closed:
                    break
                print(f"患者(WAV)> {wav_path}", flush=True)
                coord.run_wav_turn(wav_path, identify=(args.identify and idx == 0))

        elif args.mode == "mic":
            assert mic is not None
            turn_idx = 0
            while not coord.is_closed:
                print("\n[请说话…]", flush=True)
                coord.run_pcm_turn(
                    mic.iter_frames(),
                    identify=(args.identify and turn_idx == 0),
                )
                turn_idx += 1

        print(f"\n[完成] 登记记录数: {coord.store.count_registrations()}, 协议日志: {coord.log_path}")
        for row in coord.store.list_registrations():
            print("  SQLite ->", dict(row))
    finally:
        if mic is not None:
            mic.close()
        coord.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
