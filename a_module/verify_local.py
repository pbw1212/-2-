"""检查复诊录音、轮次接口、唤醒和声纹。"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from module_a.asr import DEFAULT_ASR_MODEL, MODULE_DIR, SAMPLE_WAV, Zipformer
from module_a.audio import FRAME_BYTES, frames, open_microphone, read_wav
from module_a.session import ModuleA
from module_a.speaker import DEFAULT_MODEL, SpeakerMatcher

EXAMPLES = MODULE_DIR / "examples"
SMOKE = EXAMPLES / "smoke"


def service_check(model_dir):
    requests = []

    def call(op, turn_id, **data):
        requests.append(dict(op=op, turn_id=turn_id, **data))

    def feed(turn_id, pcm):
        for frame in frames(pcm):
            call("pcm", turn_id, data=base64.b64encode(frame).decode("ascii"))

    demo = read_wav(SAMPLE_WAV)
    short = read_wav(SMOKE / "short.wav")
    call("start", "demo")
    feed("demo", demo)
    call("finish", "demo")
    call("finish", "demo")
    for turn_id, context in (("without_hotwords", "general"), ("with_hotwords", "registration")):
        call("start", turn_id, context=context)
        feed(turn_id, demo)
        call("finish", turn_id)
    call("start", "short")
    call("finish", "demo")
    feed("demo", short[:FRAME_BYTES])
    feed("short", short)
    call("finish", "short")
    call("start", "cancelled")
    call("cancel", "cancelled")
    call("finish", "cancelled")

    result = subprocess.run(
        [
            sys.executable, "-X", "utf8", "-m", "module_a", "--asr-model", str(model_dir),
            "--decoder", "hotwords", "serve",
        ],
        input="\n".join(map(json.dumps, requests)) + "\n",
        cwd=MODULE_DIR, encoding="utf-8", capture_output=True, timeout=75,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0]["type"] == "ready"
    assert not any(event["type"] == "error" for event in events)
    finals = [event for event in events if event["type"] == "final"]
    assert [event["turn_id"] for event in finals] == ["demo", "without_hotwords", "with_hotwords", "short"]
    assert finals[0]["text"] == "我想约康复科下周四下午"
    assert any(event["type"] == "partial" and event["turn_id"] == "demo" for event in events)
    assert finals[2]["status"] == "ok"
    assert finals[-1]["text"] == "对"
    assert any(event["type"] == "closed" and event["status"] == "cancelled" for event in events)
    first = next(event for event in events if event["type"] == "partial" and event["turn_id"] == "demo")
    return {
        "demo_first_partial": first,
        "finals": finals,
        "duplicate_finish_ignored": True,
    }


def check_wake_and_voiceprint(model_dir):
    model = Zipformer(model_dir)
    waking = ModuleA(model, wake_word="你好助手")
    waking.start_turn("wake", standby=True)
    events = []
    for frame in frames(read_wav(SMOKE / "wake.wav")):
        events.extend(waking.push_audio("wake", frame))
    events.extend(waking.finish_turn("wake"))
    wake, = [event for event in events if event["type"] == "wake"]

    enrollment = read_wav(SMOKE / "speaker1_a_cn_16k.wav")
    same = read_wav(SMOKE / "speaker1_b_cn_16k.wav")
    other = read_wav(SMOKE / "speaker2_a_cn_16k.wav")
    with TemporaryDirectory(prefix="module-a-profiles-") as temp:
        speaker = SpeakerMatcher(DEFAULT_MODEL, Path(temp) / "profiles.sqlite3")
        speaker.enroll("demo-01", [enrollment])
        module = ModuleA(model, speaker=speaker)
        module.start_turn("profile", identify=True)
        for frame in frames(same):
            module.push_audio("profile", frame)
        final, match = module.finish_turn("profile")
        rejection = speaker.match(other)
        assert final["type"] == "final" and match["type"] == "speaker"
        assert match["status"] == "candidate" and match["profile_id"] == "demo-01"
        assert rejection["status"] == "unknown"
        assert speaker.match(same[:16000])["status"] == "too_short"
    return {"wake": wake, "candidate": match, "unknown": rejection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_ASR_MODEL)
    parser.add_argument("--check-mic", action="store_true", help="额外测试麦克风采帧")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = {
        "service": service_check(args.asr_model.resolve()),
        "wake_and_speaker": check_wake_and_voiceprint(args.asr_model),
    }
    if args.check_mic:
        with open_microphone() as microphone:
            for _ in range(10):
                frame, overflow = microphone.read(FRAME_BYTES // 2)
                assert len(frame) == FRAME_BYTES and not overflow
        result["microphone"] = {"frames": 10, "frame_bytes": FRAME_BYTES}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
