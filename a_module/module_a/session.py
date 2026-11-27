from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from .asr import DEFAULT_BIAS_WEIGHT, Zipformer, ZipformerStream
from .audio import BYTES_PER_SAMPLE, DEFAULT_VAD_DBFS, activity, duration_ms

if TYPE_CHECKING:
    from .speaker import SpeakerMatcher


@dataclass
class Turn:
    id: str
    stream: ZipformerStream
    standby: bool
    speaker_pcm: bytearray | None
    started: float = field(default_factory=perf_counter)
    audio_ms: float = 0.0
    partial: str = ""

    def event(self, kind: str, **data) -> dict:
        return {
            "type": kind,
            "turn_id": self.id,
            "audio_ms": round(self.audio_ms, 2),
            "elapsed_ms": round((perf_counter() - self.started) * 1000, 2),
            **data,
        }


class ModuleA:
    """Single audio stream. Call the methods in order from the group coordinator."""

    def __init__(
        self,
        recognizer: Zipformer,
        *,
        vad_dbfs: float = DEFAULT_VAD_DBFS,
        wake_word: str | None = None,
        speaker: SpeakerMatcher | None = None,
        final_decoder: str = "beam",
        hotwords: dict[str, tuple[str, ...]] | None = None,
        bias_weight: float = DEFAULT_BIAS_WEIGHT,
    ):
        self.recognizer = recognizer
        self.vad_dbfs = vad_dbfs
        self.wake_word = wake_word.strip() if wake_word else ""
        self.speaker = speaker
        if final_decoder not in ("beam", "hotwords"):
            raise ValueError(f"Invalid final decoder: {final_decoder}")
        if final_decoder == "hotwords" and hotwords is None:
            raise ValueError("Hotword decoder requires a lexicon")
        self.final_decoder = final_decoder
        self.hotwords = hotwords or {}
        self.bias_weight = bias_weight
        self._active: Turn | None = None
        self._used_ids: set[str] = set()

    def start_turn(
        self, turn_id: str, *, standby: bool = False, identify: bool = False, context: str = "general",
    ) -> dict:
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must be a non-empty string")
        if not isinstance(standby, bool) or not isinstance(identify, bool):
            raise ValueError("standby and identify must be booleans")
        if self._active:
            raise ValueError("Finish or cancel the active turn first")
        if turn_id in self._used_ids:
            raise ValueError("turn_id has already been used")
        if standby and (not self.wake_word or identify):
            raise ValueError("Standby needs a wake word and no speaker request")
        if identify and self.speaker is None:
            raise ValueError("Speaker matching is not configured")
        phrases = ()
        if not standby and self.final_decoder == "hotwords":
            if not isinstance(context, str) or context not in self.hotwords:
                raise ValueError(f"找不到这个问题对应的词表 {context}")
            phrases = self.hotwords[context]
        self._active = Turn(
            turn_id, self.recognizer.stream(phrases, weight=self.bias_weight),
            standby, bytearray() if identify else None,
        )
        self._used_ids.add(turn_id)
        return self._active.event("started", standby=standby)

    def push_audio(self, turn_id: str, pcm: bytes) -> list[dict]:
        turn = self._active
        if turn is None or turn.id != turn_id or not pcm:
            return []
        if len(pcm) % BYTES_PER_SAMPLE:
            raise ValueError("PCM must contain whole signed 16-bit samples")
        level, voiced = activity(pcm, self.vad_dbfs)
        text = turn.stream.push(pcm)
        turn.audio_ms += duration_ms(pcm)
        if turn.speaker_pcm is not None:
            turn.speaker_pcm.extend(pcm)
        events = [turn.event("audio", level_dbfs=level, voiced=voiced)]
        if turn.standby:
            if self.wake_word in text:
                self._active = None
                events.append(turn.event("wake", keyword=self.wake_word))
        elif text != turn.partial:
            turn.partial = text
            events.append(turn.event("partial", text=text))
        return events

    def finish_turn(self, turn_id: str) -> list[dict]:
        turn = self._active
        if turn is None or turn.id != turn_id:
            return []
        self._active = None
        started = perf_counter()
        text = turn.stream.finish()
        if turn.standby:
            if self.wake_word in text:
                return [turn.event("wake", keyword=self.wake_word)]
            return [turn.event("closed", status="wake_missed")]
        final = turn.event(
            "final",
            text=text,
            status="ok" if text else "no_text",
            finalize_ms=round((perf_counter() - started) * 1000, 2),
        )
        events = [final]
        if turn.speaker_pcm is not None:
            assert self.speaker is not None
            try:
                result = self.speaker.match(bytes(turn.speaker_pcm))
            except Exception as exc:
                # A speaker failure must leave the finalized speech result usable.
                result = {"status": "error", "profile_id": None, "message": str(exc)}
            events.append(turn.event("speaker", **result))
        return events

    def cancel_turn(self, turn_id: str) -> list[dict]:
        turn = self._active
        if turn is None or turn.id != turn_id:
            return []
        self._active = None
        return [turn.event("closed", status="cancelled")]
