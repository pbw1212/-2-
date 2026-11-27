"""端点状态机：唯一允许发 ``TURN_STOP`` 的地方。

每个用户轮次一个实例，状态只有三个：

    WAIT_SPEECH --(检测到语音)--> IN_SPEECH --(静音达阈值/整轮超上限)--> STOPPED

硬性约定（与组协议一致）：

* **每轮只发一次** ``TURN_STOP``；发出后再来的 tick 一律丢弃（返回 ``None``）；
* 迟到的旧轮次事件按 ``turn_id`` 丢弃，不报错也不影响当前轮；
* ``decision_audio_ms`` 取决策时**最新可见的**音频位置，不是“语音结束位置”；
* 只使用已经到达的信息：文本必须已经出现在 tick 里，稳定时长按
  ``partial_updated_ms`` 计算（重复文本不刷新）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from . import protocol
from .context import TurnContext
from .strategies import Strategy, TextState


@dataclass(frozen=True)
class Tick:
    """一条 ``ASR_TICK``：A 的流式进度。"""

    audio_ms: float
    is_speech: bool
    partial_text: str = ""
    partial_updated_ms: float | None = None
    turn_id: int | None = None


@dataclass(frozen=True)
class TurnStop:
    """一次端点决策的结果（内部结构，转发时转成协议事件）。"""

    session_id: str
    turn_id: int
    decision_audio_ms: float
    reason: str
    silence_ms: float
    wait_ms: float
    basis: str
    strategy: str
    state_at_stop: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "decision_audio_ms": round(self.decision_audio_ms, 2),
            "reason": self.reason,
            "silence_ms": round(self.silence_ms, 2),
            "wait_ms": round(self.wait_ms, 2),
            "basis": self.basis,
            "strategy": self.strategy,
        }

    def to_event(self) -> Dict[str, Any]:
        """转成组协议的 ``TURN_STOP`` 事件。"""
        return protocol.make_turn_stop(
            self.session_id,
            self.turn_id,
            self.decision_audio_ms,
            self.reason,
        )


class Detector:
    """单轮端点检测器。"""

    WAIT_SPEECH = "wait_speech"
    IN_SPEECH = "in_speech"
    STOPPED = "stopped"

    def __init__(
        self,
        session_id: str,
        turn_id: int,
        config: Mapping[str, Any],
        strategy: Strategy,
        context: TurnContext | None = None,
    ):
        if not isinstance(turn_id, int) or isinstance(turn_id, bool) or turn_id < 0:
            raise ValueError(f"turn_id 必须是非负整数：{turn_id!r}")
        self.session_id = session_id
        self.turn_id = turn_id
        self.config = config
        self.strategy = strategy
        self.context = context

        audio_cfg = config["audio"]
        limits = config["limits"]
        self.frame_ms = float(audio_cfg["frame_ms"])
        self.no_speech_ms = float(limits["no_speech_ms"])
        self.hard_cap_ms = float(limits["hard_cap_ms"])

        self.state = self.WAIT_SPEECH
        self.last_audio_ms: float | None = None
        self.last_voiced_ms: float | None = None

        self._text = ""
        self._text_changed_at: float | None = None
        self.text_change_count = 0

        self.result: TurnStop | None = None
        self.tick_count = 0
        self.dropped_ticks = 0
        self.history: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------ 状态
    @property
    def stopped(self) -> bool:
        return self.state == self.STOPPED

    @property
    def text(self) -> str:
        return self._text

    def text_state(self, audio_ms: float) -> TextState:
        stable = None if self._text_changed_at is None else max(audio_ms - self._text_changed_at, 0.0)
        return TextState(self._text, self._text_changed_at, stable)

    # ------------------------------------------------------------------ 主流程
    def on_tick(self, tick: Tick) -> TurnStop | None:
        """喂一条 ``ASR_TICK``；本轮结束时返回 ``TurnStop``，否则 ``None``。"""
        if self.stopped:
            self.dropped_ticks += 1
            return None
        if tick.turn_id is not None and tick.turn_id != self.turn_id:
            # 迟到的旧轮次消息：按 turn_id 丢弃
            self.dropped_ticks += 1
            return None

        audio_ms = float(tick.audio_ms)
        if self.last_audio_ms is not None and audio_ms < self.last_audio_ms:
            raise ValueError(f"audio_ms 必须单调不减：{audio_ms} < {self.last_audio_ms}")
        self.last_audio_ms = audio_ms
        self.tick_count += 1

        self._update_text(tick, audio_ms)

        if tick.is_speech:
            self.last_voiced_ms = audio_ms
            self.state = self.IN_SPEECH
        elif self.state == self.WAIT_SPEECH:
            if audio_ms >= self.no_speech_ms:
                return self._stop(audio_ms, "no_speech", wait_ms=self.no_speech_ms,
                                  basis="limit:no_speech")
        else:
            silence = audio_ms - float(self.last_voiced_ms or 0.0)
            decision = self.strategy.decide(self.context, self.text_state(audio_ms))
            if silence >= decision.wait_ms:
                return self._stop(audio_ms, decision.reason, wait_ms=decision.wait_ms,
                                  basis=decision.basis, silence_ms=silence)

        # 硬上限：兜底，宁可报 timeout 也不让一轮永远不结束
        if audio_ms >= self.hard_cap_ms:
            return self._stop(audio_ms, "timeout", wait_ms=self.hard_cap_ms,
                              basis="limit:hard_cap",
                              silence_ms=audio_ms - float(self.last_voiced_ms or 0.0))
        return None

    def _update_text(self, tick: Tick, audio_ms: float) -> None:
        text = tick.partial_text or ""
        if text == self._text:
            return  # 重复文本不刷新稳定计时
        self._text = text
        self.text_change_count += 1
        stamp = tick.partial_updated_ms
        if stamp is None or float(stamp) > audio_ms:
            stamp = audio_ms
        self._text_changed_at = max(float(stamp), 0.0)

    def _stop(self, audio_ms: float, reason: str, *, wait_ms: float, basis: str,
              silence_ms: float | None = None) -> TurnStop:
        if silence_ms is None:
            silence_ms = audio_ms - float(self.last_voiced_ms or 0.0)
        result = TurnStop(
            session_id=self.session_id,
            turn_id=self.turn_id,
            decision_audio_ms=audio_ms,
            reason=reason,
            silence_ms=silence_ms,
            wait_ms=wait_ms,
            basis=basis,
            strategy=self.strategy.name,
            state_at_stop=self.state,
        )
        self.result = result
        self.state = self.STOPPED
        self.history.append(result.to_dict())
        return result


def run_ticks(
    ticks,
    *,
    session_id: str,
    turn_id: int,
    config: Mapping[str, Any],
    strategy: Strategy,
    context: TurnContext | None = None,
) -> tuple[TurnStop | None, Detector]:
    """把一串 tick 灌进一个 Detector，返回 (最后一次决策, detector)。"""
    detector = Detector(session_id, turn_id, config, strategy, context)
    stop: TurnStop | None = None
    for tick in ticks:
        result = detector.on_tick(tick)
        if result is not None and stop is None:
            stop = result
    return stop, detector
