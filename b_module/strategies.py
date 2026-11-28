"""端点策略：决定“这一轮还要再等多久”。

三档策略对应 B 方案里的三级对照，每级只加入一种新证据：

=====================  =====================================  ====================
策略                    额外使用的信息                         触发后的 reason
=====================  =====================================  ====================
``fixed``               无（固定静音时长）                      ``fixed_silence``
``adaptive_question``   D 的 ``question_type``                 ``adaptive_silence``
``adaptive_text``       临时转写文本 + 稳定窗口                 ``adaptive_silence``
=====================  =====================================  ====================

“等待上限”（默认 1500 ms）只对自适应策略生效，且属于**正常轮末**，因此 reason 记
``adaptive_silence`` 而不是 ``timeout``；``timeout`` 只用于 Detector 的整轮硬上限
（默认 30 s），下游按 error 处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol

from .context import TurnContext, VALID_QUESTION_TYPES

# 归一化时去掉的标点与空白：模型出的转写通常带少量标点/空格
_STRIP_CHARS = " \t\r\n，。！？、；：,.!?;:…—·\"'“”‘’（）()【】[]《》<>"


@dataclass(frozen=True)
class TextState:
    """Detector 记录下来的文本状态，只包含“此刻已经发生”的信息。"""

    text: str = ""
    changed_at_ms: float | None = None
    stable_ms: float | None = None

    @property
    def has_text(self) -> bool:
        return bool(normalize(self.text))


@dataclass(frozen=True)
class Decision:
    wait_ms: float
    reason: str
    basis: str

    def to_dict(self) -> Dict[str, Any]:
        return {"wait_ms": self.wait_ms, "reason": self.reason, "basis": self.basis}


class Strategy(Protocol):
    name: str

    def decide(self, context: TurnContext | None, text: TextState) -> Decision:  # pragma: no cover
        ...


def normalize(text: str) -> str:
    """去掉空格与标点，便于短答匹配。"""
    return "".join(char for char in str(text) if char not in _STRIP_CHARS)


class FixedStrategy:
    """固定静音阈值基线。"""

    name = "fixed"

    def __init__(self, threshold_ms: float):
        if threshold_ms <= 0:
            raise ValueError(f"固定阈值必须为正：{threshold_ms}")
        self.threshold_ms = float(threshold_ms)

    def decide(self, context: TurnContext | None, text: TextState) -> Decision:
        return Decision(self.threshold_ms, "fixed_silence", f"fixed:{self.threshold_ms:.0f}")

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"FixedStrategy({self.threshold_ms:.0f}ms)"


class AdaptiveQuestionStrategy:
    """自适应一：只按 D 的问题类型选档，不碰文本。"""

    name = "adaptive_question"

    def __init__(self, thresholds: Mapping[str, float], fallback_ms: float = 900.0,
                 max_wait_ms: float | None = None):
        # 只取问题类型三个合法键；配置里的 comment / fallback_ms 等说明字段一律忽略
        self.thresholds = {
            str(key): float(value)
            for key, value in thresholds.items()
            if key in VALID_QUESTION_TYPES
        }
        self.fallback_ms = float(fallback_ms)
        self.max_wait_ms = max_wait_ms

    def _clamp(self, wait_ms: float) -> float:
        if self.max_wait_ms is None:
            return wait_ms
        return min(wait_ms, float(self.max_wait_ms))

    def decide(self, context: TurnContext | None, text: TextState) -> Decision:
        question_type = context.question_type if context is not None else "unknown"
        if question_type in self.thresholds:
            wait = self._clamp(self.thresholds[question_type])
            return Decision(wait, "adaptive_silence", f"question:{question_type}")
        # 上下文缺失或类型非法 -> 回退固定档，绝不猜
        return Decision(self._clamp(self.fallback_ms), "adaptive_silence", "question:fallback")


class AdaptiveTextStrategy:
    """自适应二：在自适应一之上用临时文本调整等待。

    只做两件有明确线索的事（对应 B 方案的两张表）：

    * **明确短答**：是非/定向轮里文本已经稳定，且命中短答词或很短（≤ ``short_answer_max_chars``）
      且没有续说线索 -> 缩短到 ``short_answer_ms``；
    * **明显未完**：出现续说线索或句尾是连接成分 -> 拉长到 ``continuation_ms``。

    开放式轮次即使已识别出单个字段也不走短答档（沿用开放式等待），避免把
    “我想约康复科……”的半句提前截断。
    """

    name = "adaptive_text"

    def __init__(self, config: Mapping[str, Any], question_thresholds: Mapping[str, float],
                 fallback_ms: float = 900.0, max_wait_ms: float | None = None):
        self.short_answer_ms = float(config["short_answer_ms"])
        self.continuation_ms = float(config["continuation_ms"])
        self.stability_ms = float(config["stability_ms"])
        self.max_wait_ms = max_wait_ms if max_wait_ms is not None else config.get("max_wait_ms")
        self.short_answer_max_chars = int(config.get("short_answer_max_chars", 6))
        self.short_answer_patterns = tuple(str(item) for item in config.get("short_answer_patterns", ()))
        self.continuation_patterns = tuple(str(item) for item in config.get("continuation_patterns", ()))
        self.continuation_tail_chars = tuple(str(item) for item in config.get("continuation_tail_chars", ()))
        self._question = AdaptiveQuestionStrategy(question_thresholds, fallback_ms, self.max_wait_ms)

    def _clamp(self, wait_ms: float) -> float:
        if self.max_wait_ms is None:
            return wait_ms
        return min(wait_ms, float(self.max_wait_ms))

    _TARGETED_FILLERS = ("嗯", "呃", "啊", "额", "那个", "就是", "然后")
    _INCOMPLETE_TAILS = ("月", "周", "星期", "礼拜", "挂", "约", "找", "看", "改", "换")

    def _is_short_answer(self, text: str, context: TurnContext | None) -> bool:
        if context is None or context.question_type not in ("targeted", "yes_no"):
            return False
        if context.question_type == "targeted":
            if text in self._TARGETED_FILLERS:
                return False
            if any(text.endswith(cue) for cue in self._INCOMPLETE_TAILS):
                return False
            if len(text) == 1 and text not in self.short_answer_patterns:
                return False
        if any(text.endswith(pattern) for pattern in self.short_answer_patterns):
            return True
        return len(text) <= self.short_answer_max_chars

    def _is_continuing(self, text: str) -> bool:
        if any(pattern in text for pattern in self.continuation_patterns):
            return True
        return any(text.endswith(char) for char in self.continuation_tail_chars)

    def decide(self, context: TurnContext | None, text: TextState) -> Decision:
        base = self._question.decide(context, text)
        if text.stable_ms is None or text.stable_ms < self.stability_ms:
            return base

        normalized = normalize(text.text)
        if not normalized:
            return base

        continuation = self._is_continuing(normalized)
        if not continuation and self._is_short_answer(normalized, context):
            return Decision(self._clamp(self.short_answer_ms), "adaptive_silence",
                            "text:short_answer")
        if continuation:
            return Decision(self._clamp(self.continuation_ms), "adaptive_silence",
                            "text:continuation")
        return Decision(base.wait_ms, base.reason, f"{base.basis}|text:neutral")


def build_strategy(spec: str, config: Mapping[str, Any]) -> Strategy:
    """由 ``fixed`` / ``fixed:600`` / ``adaptive_question`` / ``adaptive_text`` 构造策略。"""
    strategies = config["strategies"]
    limits = config.get("limits", {})
    max_wait = limits.get("max_wait_ms")

    raw = str(spec).strip().lower()
    if not raw:
        raise ValueError("策略不能为空")

    if raw.startswith("fixed"):
        thresholds = list(strategies["fixed"]["thresholds_ms"])
        if ":" in raw:
            _, value = raw.split(":", 1)
            threshold = float(value)
        else:
            threshold = float(strategies["fixed"]["default_ms"])
        if thresholds and threshold not in thresholds:
            # 允许扫描非预设档位，但提示一下，避免和对照表口径不一致
            pass
        return FixedStrategy(threshold)

    if raw in ("adaptive_question", "adaptive", "adaptive1"):
        return AdaptiveQuestionStrategy(
            strategies["adaptive_question"],
            fallback_ms=float(strategies["adaptive_question"].get("fallback_ms", 900.0)),
            max_wait_ms=max_wait,
        )

    if raw in ("adaptive_text", "adaptive2"):
        return AdaptiveTextStrategy(
            strategies["adaptive_text"],
            strategies["adaptive_question"],
            fallback_ms=float(strategies["adaptive_question"].get("fallback_ms", 900.0)),
            max_wait_ms=max_wait,
        )

    raise ValueError(f"未知策略：{spec!r}")


def available_strategies(config: Mapping[str, Any]) -> list[str]:
    """给 CLI 提示用的策略清单（固定档展开成每档一个）。"""
    names = [f"fixed:{int(item)}" for item in config["strategies"]["fixed"]["thresholds_ms"]]
    names.extend(["adaptive_question", "adaptive_text"])
    return names
