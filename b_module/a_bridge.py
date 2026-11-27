"""A ↔ 组协议转接层（S14）。

A 的 ``serve`` 协议是**内层协议**，与组协议不是一回事。差异一共四条，全部在这里吸收，
A 的代码一行不改：

==============  ==========================  =====================================
差异              A 的 serve                  组协议 schema_version=1
==============  ==========================  =====================================
轮次标识          ``turn_id`` 是字符串         非负整数
                  （``"visit-1"`` 之类）
字段命名          ``voiced`` / ``text`` /      ``is_speech`` / ``partial_text`` /
                  ``status: no_text``          ``status: no_input``
时间戳            ``partial`` 不带时间戳       ``partial_updated_ms`` 必填
事件粒度          ``audio`` + 可选 ``partial`` 每帧一条完整 ``ASR_TICK``
==============  ==========================  =====================================

另外 ``speaker`` 事件要映射成 ``PROFILE_SUGGESTION``：A 的 ``candidate`` → ``matched``，
``unknown`` → ``low_score``，其余 → ``none``。

用法（真实联调，需要 A 自己的 venv 与模型）：

    python -m b_module.a_bridge --case dev_01 --strategy fixed:600 --python D:/path/to/venv/python.exe

只验证翻译逻辑（不需要 A）：

    python -m pytest b_module/tests/test_a_bridge.py -q
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from . import audio, dataset, paths, protocol, replay, strategies
from .context import TurnContext
from .detector import Detector, Tick

A_SAMPLE_RATE = 16_000
A_FRAME_MS = 20.0
A_FRAME_BYTES = 640  # 16 kHz * 20 ms * 2 byte

SAMPLE_RATE_TOLERANCE_MS = 0.01


class ABridgeError(RuntimeError):
    """A 子进程不可用或协议不符。"""


# ---------------------------------------------------------------- 协议翻译
@dataclass
class AProtocolTranslator:
    """A 的内层事件 -> 组协议事件。纯函数式，便于单测。"""

    session_id: str
    start_audio_ms: float = 0.0
    text: str = ""
    text_changed_at_ms: float | None = None
    last_audio_ms: float = 0.0
    frame_ms: float = A_FRAME_MS

    # ---- 轮次标识
    @staticmethod
    def map_turn_id(turn_id: int) -> str:
        """组协议的整数轮次 -> A 内层字符串标识（同一会话内唯一）。"""
        if not isinstance(turn_id, int) or isinstance(turn_id, bool) or turn_id < 0:
            raise ValueError(f"turn_id 必须是非负整数：{turn_id!r}")
        return f"turn{turn_id}"

    # ---- 词表上下文
    @staticmethod
    def context_name(context: TurnContext | None) -> str:
        """D 的 expected_slot -> A 的热词词表分组。"""
        if context is None:
            return "general"
        slot = context.expected_slot
        if slot == "department":
            return "department"
        if slot == "date":
            return "date"
        if slot in ("period", "doctor", "confirm"):
            return "registration"
        if context.question_type in ("open", "targeted", "yes_no"):
            return "registration" if context.question_type != "yes_no" else "general"
        return "general"

    def reset(self) -> None:
        self.text = ""
        self.text_changed_at_ms = None
        self.last_audio_ms = 0.0

    # ---- 主翻译
    def observe(self, event: Mapping[str, Any], turn_id: int,
                expect_a_turn: str | None = None) -> List[Dict[str, Any]]:
        """把一条 A 事件翻译成零条或多条组协议事件。"""
        a_turn = event.get("turn_id")
        if expect_a_turn is not None and a_turn is not None and a_turn != expect_a_turn:
            return []  # 迟到的旧轮次，丢弃

        kind = event.get("type")
        if kind == "audio":
            return [self._tick(turn_id, event)]
        if kind == "partial":
            # A 只在文本真的变化时才发 partial，正好可以用来打时间戳
            self.text = str(event.get("text") or "")
            self.text_changed_at_ms = float(event.get("audio_ms") or 0.0)
            return []
        if kind == "final":
            return [self._final(turn_id, event)]
        if kind == "speaker":
            return self._speaker(turn_id, event)
        if kind in ("started", "wake", "closed", "ready"):
            return []
        if kind == "error":
            raise ABridgeError(f"A 报错：{event.get('message')}")
        return []

    def _tick(self, turn_id: int, event: Mapping[str, Any]) -> Dict[str, Any]:
        audio_ms = float(event.get("audio_ms") or 0.0)
        self.last_audio_ms = audio_ms
        payload = {
            "audio_ms": round(audio_ms, 2),
            "is_speech": bool(event.get("voiced")),
            "partial_text": self.text or None,
            "partial_updated_ms": None if not self.text else round(
                float(self.text_changed_at_ms or 0.0), 2),
        }
        return protocol.make_event("ASR_TICK", self.session_id, turn_id, payload)

    def _final(self, turn_id: int, event: Mapping[str, Any]) -> Dict[str, Any]:
        a_status = str(event.get("status") or "")
        status = {"ok": "ok", "no_text": "no_input"}.get(a_status, "error")
        payload: Dict[str, Any] = {
            "status": status,
            "final_text": (str(event.get("text")) if event.get("text") else None),
            "start_audio_ms": round(self.start_audio_ms, 2),
            "end_audio_ms": round(float(event.get("audio_ms") or self.last_audio_ms), 2),
        }
        if status == "error":
            payload["error_code"] = f"asr_final_status:{a_status or 'unknown'}"
        return protocol.make_event("ASR_FINAL", self.session_id, turn_id, payload)

    def _speaker(self, turn_id: int, event: Mapping[str, Any]) -> List[Dict[str, Any]]:
        a_status = str(event.get("status") or "")
        match_status = {
            "candidate": "matched",
            "unknown": "low_score",
        }.get(a_status, "none")
        payload = {
            "profile_id": event.get("profile_id"),
            "match_status": match_status,
        }
        return [protocol.make_event("PROFILE_SUGGESTION", self.session_id, turn_id, payload)]


# ------------------------------------------------------------- 子进程封装
@dataclass
class ABridge:
    """启动 A 的 ``serve`` 子进程，并用组协议与之对话。"""

    session_id: str = "s01"
    python_exe: str | None = None
    a_root: Path = field(default_factory=lambda: paths.DEFAULT_A_ROOT)
    a_args: Sequence[str] = ()
    command: Sequence[str] | None = None   # 测试可注入假 A
    event_timeout_s: float = 30.0
    translator: AProtocolTranslator = field(init=False)

    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _queue: "queue.Queue[dict]" = field(default_factory=queue.Queue, init=False, repr=False)
    _reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _stderr_tail: List[str] = field(default_factory=list, init=False, repr=False)
    _stderr_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    ready: Dict[str, Any] = field(default_factory=dict, init=False)
    sample_rate: int = A_SAMPLE_RATE
    frame_ms: float = A_FRAME_MS
    _a_turn: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.translator = AProtocolTranslator(self.session_id)

    # ---- 生命周期
    @property
    def frame_bytes(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0)) * 2

    def open(self) -> Dict[str, Any]:
        if self._process is not None:
            return self.ready
        command = list(self.command) if self.command else [
            self.python_exe or sys.executable, "-m", "module_a", "serve", *self.a_args,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.a_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise ABridgeError(
                f"无法启动 A：{exc}\n"
                f"提示：A 需要独立的 Python 环境（Python 3.11/3.12 + a_module/requirements.txt），"
                f"用 --python 指定该解释器。"
            ) from exc

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        ready = self._next_event(self.event_timeout_s)
        if ready.get("type") != "ready":
            raise ABridgeError(f"A 没有按协议回 ready：{ready}")
        self.ready = ready
        self.sample_rate = int(ready.get("sample_rate") or A_SAMPLE_RATE)
        self.frame_ms = float(ready.get("frame_ms") or A_FRAME_MS)
        return ready

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.stdin and process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    def __enter__(self) -> "ABridge":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- 一轮
    def begin_turn(self, turn_id: int, context: TurnContext | None = None,
                   identify: bool = False) -> List[Dict[str, Any]]:
        self.open()
        a_turn = self.translator.map_turn_id(turn_id)
        self._a_turn = a_turn
        self.translator.reset()
        self.translator.text = ""
        self.translator.text_changed_at_ms = None
        request = {
            "op": "start",
            "turn_id": a_turn,
            "context": self.translator.context_name(context),
            "identify": bool(identify),
        }
        self._send(request)
        events: List[Dict[str, Any]] = []
        while True:
            event = self._next_event(self.event_timeout_s)
            if event.get("type") == "error":
                raise ABridgeError(f"A 拒绝 start：{event.get('message')}")
            events.extend(self.translator.observe(event, turn_id, a_turn))
            if event.get("type") == "started":
                return events

    def push_frame(self, turn_id: int, pcm: bytes) -> List[Dict[str, Any]]:
        """送入一帧 PCM（16 kHz/单声道/16 bit），返回对应的一条 ASR_TICK。"""
        if self._process is None:
            raise ABridgeError("A 尚未启动")
        self._send({
            "op": "pcm",
            "turn_id": self._a_turn,
            "data": base64.b64encode(pcm).decode("ascii"),
        })
        events: List[Dict[str, Any]] = []
        while True:
            event = self._next_event(self.event_timeout_s)
            events.extend(self.translator.observe(event, turn_id, self._a_turn))
            if event.get("type") == "audio":
                return events

    def finish_turn(self, turn_id: int, *, speaker_timeout_s: float = 0.5) -> List[Dict[str, Any]]:
        """结束本轮，返回 ASR_FINAL（若有声纹再附带 PROFILE_SUGGESTION）。"""
        if self._process is None:
            raise ABridgeError("A 尚未启动")
        self._send({"op": "finish", "turn_id": self._a_turn})
        events: List[Dict[str, Any]] = []
        while True:
            event = self._next_event(self.event_timeout_s)
            events.extend(self.translator.observe(event, turn_id, self._a_turn))
            if event.get("type") == "final":
                break
        # speaker 事件紧随 final；它可缺省，不能无限等
        while True:
            try:
                event = self._next_event(speaker_timeout_s)
            except ABridgeError:
                break
            events.extend(self.translator.observe(event, turn_id, self._a_turn))
            if event.get("type") == "speaker":
                break
        return events

    def cancel_turn(self, turn_id: int) -> None:
        if self._process is None:
            return
        self._send({"op": "cancel", "turn_id": self._a_turn})

    # ---- 内部工具
    def _send(self, request: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise ABridgeError("A 尚未启动")
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ABridgeError(f"写 A 的 stdin 失败：{exc}\n{self._stderr_text()}") from exc

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._queue.put({"type": "_eof"})

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 40:
                self._stderr_tail.pop(0)

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail[-10:])

    def _next_event(self, timeout_s: float) -> Dict[str, Any]:
        try:
            event = self._queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise ABridgeError(
                f"等待 A 的事件超时（{timeout_s:.1f}s）。stderr 末尾：\n{self._stderr_text()}"
            ) from exc
        if event.get("type") == "_eof":
            self._queue.put(event)  # 让后续等待也立刻失败
            raise ABridgeError(f"A 进程已退出。stderr 末尾：\n{self._stderr_text()}")
        return event


# ------------------------------------------------------------------ 端到端演示
def load_pcm_16k(case_id: str, config: Mapping[str, Any] | None = None,
                 a_root: str | Path = paths.DEFAULT_A_ROOT) -> bytes:
    """把 B 的重建音频转成 A 需要的 16 kHz 单声道 16 bit PCM。"""
    config = config or paths.load_config()
    specs = {spec.case_id: spec for spec in dataset.load_cases(a_root)}
    if case_id not in specs:
        raise KeyError(f"manifest 里没有 {case_id}")
    case_audio = dataset.build_case_audio(specs[case_id], a_root=a_root, config=config,
                                          use_cache=True)
    samples = audio.resample_linear(case_audio.samples, case_audio.sample_rate, A_SAMPLE_RATE)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    return pcm


def iter_pcm_frames(pcm: bytes, frame_bytes: int = A_FRAME_BYTES) -> Iterable[bytes]:
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset: offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        yield chunk


def run_ab_loop(
    case_id: str,
    strategy_spec: str,
    *,
    config: Mapping[str, Any] | None = None,
    a_root: str | Path = paths.DEFAULT_A_ROOT,
    bridge: ABridge | None = None,
    realtime: bool = False,
    log: Any = print,
    quiet: bool = False,
) -> Dict[str, Any]:
    """A -> ASR_TICK -> B -> TURN_STOP -> A -> ASR_FINAL 的最小闭环。"""
    config = config or paths.load_config()
    specs = {spec.case_id: spec for spec in dataset.load_cases(a_root)}
    spec = specs[case_id]
    context = replay.spec_context(spec)
    strategy = strategies.build_strategy(strategy_spec, config)
    det = Detector("s01", 1, config, strategy, context)

    owns_bridge = bridge is None
    bridge = bridge or ABridge(session_id="s01", a_root=Path(a_root))
    pcm = load_pcm_16k(case_id, config, a_root)
    events: List[Dict[str, Any]] = []
    final_events: List[Dict[str, Any]] = []
    stop = None
    started = time.perf_counter()

    try:
        bridge.open()
        if not quiet:
            log(f"A ready: sample_rate={bridge.sample_rate} frame_ms={bridge.frame_ms}")
        bridge.begin_turn(1, context)
        for index, frame in enumerate(iter_pcm_frames(pcm, bridge.frame_bytes)):
            audio_ms = (index + 1) * bridge.frame_ms
            if realtime:
                delay = started + audio_ms / 1000.0 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            for event in bridge.push_frame(1, frame):
                events.append(event)
                payload = event["payload"]
                tick = Tick(
                    audio_ms=float(payload["audio_ms"]),
                    is_speech=bool(payload["is_speech"]),
                    partial_text=str(payload.get("partial_text") or ""),
                    partial_updated_ms=payload.get("partial_updated_ms"),
                    turn_id=1,
                )
                stop = det.on_tick(tick)
                if stop is not None:
                    break
            if stop is not None:
                break
        final_events = bridge.finish_turn(1)
        events.extend(final_events)
    finally:
        if owns_bridge:
            bridge.close()

    if not quiet:
        log(f"A -> B: {len([e for e in events if e['event'] == 'ASR_TICK'])} 条 ASR_TICK")
        if stop is None:
            log("B 未发出 TURN_STOP")
        else:
            log(f"B -> A: TURN_STOP decision={stop.decision_audio_ms:.0f}ms "
                f"reason={stop.reason} basis={stop.basis}")
        for event in final_events:
            if event["event"] == "ASR_FINAL":
                log(f"A -> C: ASR_FINAL status={event['payload']['status']} "
                    f"text=\"{event['payload'].get('final_text') or ''}\"")
            elif event["event"] == "PROFILE_SUGGESTION":
                log(f"A -> D: PROFILE_SUGGESTION {event['payload']}")

    return {"stop": stop, "events": events, "detector": det}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m b_module.a_bridge",
        description="A <-> 组协议转接层：跑一次 A->B->A 闭环",
    )
    parser.add_argument("--case", default="dev_01", help="样本 id")
    parser.add_argument("--strategy", default="fixed:600", help="端点策略")
    parser.add_argument("--python", dest="python_exe", default=None,
                        help="A 所在环境的解释器（默认用当前解释器）")
    parser.add_argument("--a-args", nargs="*", default=[], help="透传给 A 的 serve 参数")
    parser.add_argument("--realtime", action="store_true", help="按真实时间送帧")
    args = parser.parse_args(argv)

    try:
        result = run_ab_loop(
            args.case, args.strategy,
            bridge=ABridge(session_id="s01", python_exe=args.python_exe,
                           a_args=tuple(args.a_args or ()), a_root=paths.DEFAULT_A_ROOT),
            realtime=args.realtime,
        )
    except ABridgeError as exc:
        print(f"[A 未就绪] {exc}", file=sys.stderr)
        print("\n下一步：", file=sys.stderr)
        print("  1) 建 A 的环境（Python 3.11/3.12，别用 Anaconda base）：")
        print("     python -m venv a_module/.venv")
        print("     .\\a_module\\.venv\\Scripts\\pip install -r a_module/requirements.txt")
        print("  2) 拉模型：.\\a_module\\.venv\\Scripts\\python.exe a_module/fetch_models.py")
        print("  3) 重跑：python -m b_module.a_bridge --python a_module/.venv/Scripts/python.exe")
        print("\n如果 stderr 里出现 'OMP: Error #15'，说明用的是 Anaconda base 的 Python，"
              "它的 torch 本身起不来，必须换独立环境。", file=sys.stderr)
        return 2
    return 0 if result["stop"] is not None else 3


if __name__ == "__main__":
    raise SystemExit(main())
