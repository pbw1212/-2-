"""模拟的临时转写时间线（A 的流式输出未就绪时的替代）。

真实链路上 ``partial_text`` 由 A 的 Zipformer 每 160 ms 左右吐一次，且**允许回改**。
B 的阶段一先在时间线上复现这种行为：同一时刻只暴露“当时已经到达的文本”，
任何策略都只使用已经发生的信息——这样换成 A 的真实事件后，端点逻辑一行不用改。

时间线条目格式（``data/text_timeline.json``）：

    {"dev_01": [[0, ""], [340, "我"], [600, "我想"], [1180, "我想约康付"], [1340, "我想约康复科"]]}

约定：
* 时间是**该条音频内的毫秒**，与 ``audio_ms`` 同一时间轴；
* 只写“文本发生变化”的时刻，重复文本不必重复写；
* 文本必须单调（同一时刻只能有一个值），允许反向回改（如识别纠错）。
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from . import paths

DEFAULT_TIMELINE_PATH = paths.DATA_DIR / "text_timeline.json"


@dataclass(frozen=True)
class TextEvent:
    at_ms: float
    text: str


class TextTimeline:
    """一条时间线：任意 audio_ms 时刻“当前可见”的临时文本。"""

    def __init__(self, events: Iterable[TextEvent] = ()):
        ordered = sorted(events, key=lambda item: item.at_ms)
        cleaned: List[TextEvent] = []
        for event in ordered:
            if event.at_ms < 0:
                raise ValueError(f"时间线不允许负时刻：{event.at_ms}")
            if cleaned and cleaned[-1].at_ms == event.at_ms:
                if cleaned[-1].text != event.text:
                    raise ValueError(f"同一时刻出现两个文本：{event.at_ms}")
                continue
            # 与上一条文本相同则不是“变化”，去掉
            if cleaned and cleaned[-1].text == event.text:
                continue
            cleaned.append(event)
        self._events: Tuple[TextEvent, ...] = tuple(cleaned)
        self._times: List[float] = [event.at_ms for event in self._events]

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)

    @property
    def events(self) -> Tuple[TextEvent, ...]:
        return self._events

    def text_at(self, audio_ms: float) -> str:
        """该时刻可见的完整临时文本；尚未产生时为 ``""``。"""
        if not self._events:
            return ""
        index = bisect_right(self._times, float(audio_ms)) - 1
        return self._events[index].text if index >= 0 else ""

    def changed_at(self, audio_ms: float) -> float | None:
        """最近一次“文本真的变了”的时刻；尚未变化过返回 ``None``。"""
        if not self._events:
            return None
        index = bisect_right(self._times, float(audio_ms)) - 1
        return self._events[index].at_ms if index >= 0 else None

    @classmethod
    def from_entries(cls, entries: Sequence[Sequence]) -> "TextTimeline":
        events: List[TextEvent] = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f"时间线条目需要 [ms, text]：{entry!r}")
            at_ms, text = entry
            events.append(TextEvent(float(at_ms), str(text)))
        return cls(events)


def load_timelines(path: str | Path | None = None) -> Dict[str, TextTimeline]:
    """读取时间线文件；文件不存在时返回空表（全部样本退化为“无文本”）。"""
    path = Path(path) if path else DEFAULT_TIMELINE_PATH
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"时间线文件应是 {{case_id: [[ms, text], ...]}}：{path}")
    timelines: Dict[str, TextTimeline] = {}
    for case_id, entries in raw.items():
        if case_id.startswith("_") or not isinstance(entries, list):
            continue  # 允许 "_comment" 之类的说明字段
        timelines[str(case_id)] = TextTimeline.from_entries(entries)
    return timelines


_EMPTY = TextTimeline()


def timeline_for(case_id: str, timelines: Dict[str, TextTimeline] | None = None) -> TextTimeline:
    """取某条样本的时间线；没有则返回空时间线（策略自动退化为按问题类型）。"""
    if timelines is None:
        timelines = _cached_timelines()
    return timelines.get(case_id, _EMPTY)


@lru_cache(maxsize=1)
def _cached_timelines() -> Dict[str, TextTimeline]:
    return load_timelines()


def clear_cache() -> None:
    _cached_timelines.cache_clear()
