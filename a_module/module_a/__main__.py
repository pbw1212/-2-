from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import sys
from pathlib import Path
from time import perf_counter, sleep

import torch

from .asr import DEFAULT_ASR_MODEL, DEFAULT_BIAS_WEIGHT, DEFAULT_HOTWORDS, SAMPLE_WAV, Zipformer, load_hotwords
from .audio import (
    DEFAULT_VAD_DBFS, FRAME_BYTES, SAMPLE_RATE, duration_ms, frames, open_microphone, read_wav,
)
from .session import ModuleA
from .speaker import DEFAULT_MODEL, DEFAULT_PROFILES, SpeakerMatcher


def output(event: dict):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def replay(module: ModuleA, pcm: bytes, args):
    turn_id = "wav-1"
    output(module.start_turn(
        turn_id, standby=args.standby, identify=args.identify, context=args.context,
    ))
    started = perf_counter()
    played_seconds = 0.0
    for frame in frames(pcm):
        played_seconds += duration_ms(frame) / 1000
        if args.realtime:
            sleep(max(0.0, started + played_seconds - perf_counter()))
        events = module.push_audio(turn_id, frame)
        for event in events:
            if args.audio_events or event["type"] != "audio":
                output(event)
        if any(event["type"] == "wake" for event in events):
            return
    for event in module.finish_turn(turn_id):
        output(event)


def microphone(module: ModuleA, args):
    turn_id = "mic-1"
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    silence_ms = 0.0
    has_speech = False
    with open_microphone(device) as stream:
        output(module.start_turn(
            turn_id, standby=args.standby, identify=args.identify, context=args.context,
        ))
        for _ in range(math.ceil(args.seconds * 50)):
            payload, overflowed = stream.read(FRAME_BYTES // 2)
            if overflowed:
                module.cancel_turn(turn_id)
                output({"type": "error", "turn_id": turn_id, "message": "Microphone data overflow"})
                return 1
            events = module.push_audio(turn_id, bytes(payload))
            for event in events:
                if args.audio_events or event["type"] != "audio":
                    output(event)
            if events[-1]["type"] == "wake":
                return 0
            has_speech = has_speech or events[0]["voiced"]
            silence_ms = 0.0 if events[0]["voiced"] else silence_ms + 20
            if has_speech and silence_ms >= args.silence_ms:
                break
        for event in module.finish_turn(turn_id):
            output(event)
    return 0


def serve(module: ModuleA, default_context: str = "general"):
    """Standard input/output protocol. Each request produces zero or more events."""
    output({"type": "ready", "sample_rate": SAMPLE_RATE, "frame_ms": 20})
    for line in sys.stdin:
        turn_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Each request must be an object")
            turn_id = request["turn_id"]
            action = request["op"]
            if action == "start":
                events = [module.start_turn(
                    turn_id, standby=request.get("standby", False), identify=request.get("identify", False),
                    context=request.get("context", default_context),
                )]
            elif action == "pcm":
                events = module.push_audio(turn_id, base64.b64decode(request["data"], validate=True))
            elif action == "finish":
                events = module.finish_turn(turn_id)
            elif action == "cancel":
                events = module.cancel_turn(turn_id)
            else:
                raise ValueError(f"Unknown op: {action}")
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            output({"type": "error", "turn_id": turn_id, "message": str(exc)})
            continue
        for event in events:
            output(event)


def matcher(args) -> SpeakerMatcher:
    return SpeakerMatcher(
        args.speaker_model, args.profiles, threshold=args.threshold, vad_dbfs=args.vad_dbfs
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复诊登记助手的离线语音输入")
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_ASR_MODEL)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--vad-dbfs", type=float, default=DEFAULT_VAD_DBFS)
    parser.add_argument("--wake-word", default="你好助手")
    parser.add_argument("--speaker-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--decoder", "--final-decoder", choices=("beam", "hotwords"), default="beam")
    parser.add_argument("--hotwords", type=Path, default=DEFAULT_HOTWORDS)
    parser.add_argument("--bias-weight", type=float, default=DEFAULT_BIAS_WEIGHT)
    parser.add_argument("--context", default="general")
    commands = parser.add_subparsers(dest="command", required=True)
    stream = commands.add_parser("stream", help="把音频文件分帧送给识别模型")
    stream.add_argument("wav", type=Path, nargs="?")
    stream.add_argument("--realtime", action="store_true")
    mic = commands.add_parser("mic", help="用麦克风录一轮，静音后结束")
    mic.add_argument("--device")
    mic.add_argument("--seconds", type=float, default=15)
    mic.add_argument("--silence-ms", type=float, default=1500)
    for capture in (stream, mic):
        capture.add_argument("--audio-events", action="store_true")
        capture.add_argument("--standby", action="store_true")
        capture.add_argument("--identify", action="store_true")
    commands.add_parser("devices", help="查看可用的麦克风和扬声器")
    service = commands.add_parser("serve", help="启动给组员调用的语音输入服务")
    service.add_argument("--enable-speaker", action="store_true")
    enroll = commands.add_parser("enroll", help="用录音登记一个模拟档案")
    enroll.add_argument("profile_id")
    enroll.add_argument("wav", type=Path, nargs="+")
    identify = commands.add_parser("identify", help="查看录音能否匹配已登记档案")
    identify.add_argument("wav", type=Path)
    return parser


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    if args.threads < 1 or not -120 <= args.vad_dbfs <= 0:
        parser.error("线程数至少为 1，音量阈值应在 -120 到 0 之间")
    if not math.isfinite(args.bias_weight) or args.bias_weight < 0:
        parser.error("热词权重需要是非负数")
    torch.set_num_threads(args.threads)
    if args.command == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if args.command == "enroll":
        output(matcher(args).enroll(args.profile_id, [read_wav(wav) for wav in args.wav]))
        return 0
    if args.command == "identify":
        output(matcher(args).match(read_wav(args.wav)))
        return 0
    if args.command == "mic" and not (
        math.isfinite(args.seconds) and math.isfinite(args.silence_ms)
        and args.seconds > 0 and args.silence_ms > 0
    ):
        parser.error("录音时长和静音时长都需要大于 0")

    try:
        hotwords = load_hotwords(args.hotwords) if args.decoder == "hotwords" else None
        recognizer = Zipformer(args.asr_model, threads=args.threads)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if hotwords is not None and args.context not in hotwords:
        parser.error(f"找不到这个问题对应的词表 {args.context}")
    enable_speaker = getattr(args, "enable_speaker", False) or getattr(args, "identify", False)
    module = ModuleA(
        recognizer, vad_dbfs=args.vad_dbfs,
        wake_word=args.wake_word, speaker=matcher(args) if enable_speaker else None,
        final_decoder=args.decoder, hotwords=hotwords, bias_weight=args.bias_weight,
    )
    if args.command == "serve":
        serve(module, args.context)
    elif args.command == "stream":
        replay(module, read_wav(args.wav or SAMPLE_WAV), args)
    elif args.command == "mic":
        return microphone(module, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
