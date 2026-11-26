"""E-module coordinator implementing half-duplex audio gating, dedup, and playback control."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from e_module.audio_player import AudioPlayer
from e_module.config import DEFAULT_PAUSE_MS, OUTPUTS_DIR
from e_module.events import (
    make_listen_start,
    make_playback_result,
    monotonic_ms,
    to_json_line,
    validate_event,
)


class SynthesizerProtocol(Protocol):
    def synthesize(
        self,
        text: str,
        output_wav: Path,
        mode: str = "tn_prosody",
        action: Optional[str] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        cacheable: Optional[bool] = None,
        use_cache: bool = True,
    ) -> Any: ...


class GateState(str, Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"
    WAIT_CONTEXT = "WAIT_CONTEXT"
    LISTENING = "LISTENING"


class EModuleService:
    """Manages synthesis, playback, microphone half-duplex gating, and event logging."""

    def __init__(
        self,
        synthesizer: SynthesizerProtocol,
        player: AudioPlayer,
        work_dir: Path = OUTPUTS_DIR / "service_runtime",
        log_path: Optional[Path] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        mode: str = "tn_prosody",
        clear_speech: bool = False,
        trim_silence: bool = False,
        enable_pipelined: bool = False,
        on_mic_pause: Optional[Callable[[int], None]] = None,
        on_mic_flush_and_open: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.synthesizer = synthesizer
        self.player = player
        self.work_dir = Path(work_dir)
        self.wav_dir = self.work_dir / "wavs"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else (self.work_dir / "e_events.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.pause_ms = pause_ms
        self.mode = mode
        self.clear_speech = clear_speech
        self.trim_silence = trim_silence
        self.enable_pipelined = enable_pipelined
        self.on_mic_pause = on_mic_pause
        self.on_mic_flush_and_open = on_mic_flush_and_open

        self.state: GateState = GateState.IDLE
        self.mic_open: bool = False
        self.session_id: Optional[str] = None
        self.active_turn_id: int = -1
        self.active_reply_id: Optional[str] = None
        self.playback_finished_for_turn: bool = True

        # Ready TURN_CONTEXT waiting to open mic after playback finishes
        self._pending_context: Optional[Dict[str, Any]] = None
        # Track emitted LISTEN_START per (session_id, turn_id) to guarantee single open
        self._opened_turns: set[Tuple[str, int]] = set()
        # Dedup cache keyed by (session_id, reply_id) -> PLAYBACK_RESULT event
        self._reply_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.trace_records: List[Dict[str, Any]] = []

    def _record_log(self, kind: str, data: Dict[str, Any]) -> None:
        entry = {"kind": kind, "mono_ms": monotonic_ms(), **data}
        self.trace_records.append(entry)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _close_mic(self, turn_id: int) -> None:
        if self.mic_open:
            self.mic_open = False
            if self.on_mic_pause:
                self.on_mic_pause(turn_id)
            self._record_log("MIC_PAUSED", {"session_id": self.session_id, "turn_id": turn_id})

    def _maybe_open_listening(self) -> Optional[Dict[str, Any]]:
        """Emit LISTEN_START once when playback is finished and TURN_CONTEXT is ready."""
        if not self.playback_finished_for_turn or self._pending_context is None:
            return None

        ctx = self._pending_context
        sid = ctx["session_id"]
        tid = ctx["turn_id"]
        if tid == 0:
            # turn_id=0 is reserved for opening announcement with mic kept closed
            return None
        if (sid, tid) in self._opened_turns:
            return None

        if self.on_mic_flush_and_open:
            self.on_mic_flush_and_open(tid)
        self.mic_open = True
        self.state = GateState.LISTENING
        self._opened_turns.add((sid, tid))
        self._pending_context = None

        ev = make_listen_start(session_id=sid, turn_id=tid, mic_buffer_flushed=True)
        self._record_log("EVENT_OUT", {"envelope": ev})
        return ev

    def handle_turn_context(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process TURN_CONTEXT from D; open mic only if previous audio has finished."""
        validate_event(event)
        if event["event"] != "TURN_CONTEXT":
            raise ValueError(f"Expected TURN_CONTEXT, got {event['event']}")

        sid = event["session_id"]
        tid = event["turn_id"]
        self._record_log("EVENT_IN", {"envelope": event})

        if self.session_id is None or sid != self.session_id:
            self.session_id = sid
            self.active_turn_id = tid
            self.playback_finished_for_turn = True
            self._pending_context = None
        elif tid < self.active_turn_id:
            self._record_log(
                "STALE_IGNORED",
                {"reason": "stale_turn_context", "session_id": sid, "turn_id": tid, "active_turn_id": self.active_turn_id},
            )
            return None
        else:
            self.active_turn_id = tid

        self._pending_context = event
        if not self.playback_finished_for_turn:
            # Keep mic paused until playback completes
            return None
        return self._maybe_open_listening()

    def handle_reply_request(self, event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process REPLY_REQUEST from D.

        Returns (playback_result_event, optional_listen_start_event).
        """
        validate_event(event)
        if event["event"] != "REPLY_REQUEST":
            raise ValueError(f"Expected REPLY_REQUEST, got {event['event']}")

        sid = event["session_id"]
        tid = event["turn_id"]
        payload = event["payload"]
        reply_id = payload["reply_id"]
        reply_text = payload["reply_text"]
        action = payload.get("action")
        cacheable = payload.get("cacheable")

        self._record_log("EVENT_IN", {"envelope": event})

        if self.session_id is None:
            self.session_id = sid

        # 1. Drop stale turn messages
        if sid == self.session_id and tid < self.active_turn_id:
            self._record_log(
                "STALE_IGNORED",
                {"reason": "stale_reply_request", "session_id": sid, "turn_id": tid, "active_turn_id": self.active_turn_id, "reply_id": reply_id},
            )
            return None, None

        # 2. Deduplicate repeated reply_id by returning cached PLAYBACK_RESULT
        dedup_key = (sid, reply_id)
        if dedup_key in self._reply_results:
            cached_res = self._reply_results[dedup_key]
            self._record_log("DUPLICATE_REPLY_REUSED", {"session_id": sid, "turn_id": tid, "reply_id": reply_id})
            return cached_res, None

        # 3. Close mic and enter SPEAKING state (single active reply)
        self.active_turn_id = max(self.active_turn_id, tid)
        self.active_reply_id = reply_id
        self.playback_finished_for_turn = False
        self.state = GateState.SPEAKING
        self._close_mic(tid)

        wav_path = self.wav_dir / f"{sid}_t{tid}_{reply_id}.wav"
        t_req_mono_ms = event["emitted_mono_ms"]

        # Optional chunk-pipelined synthesis + playback path to minimize TTFA on multi-segment replies
        if self.enable_pipelined and hasattr(self.synthesizer, "synthesize_and_play_pipelined"):
            try:
                self._record_log("PLAYBACK_QUEUED", {"session_id": sid, "turn_id": tid, "reply_id": reply_id, "pipelined": True})
                pipe_res = self.synthesizer.synthesize_and_play_pipelined(
                    text=reply_text,
                    output_dir=self.wav_dir,
                    stem=f"{sid}_t{tid}_{reply_id}",
                    player=self.player,
                    mode=self.mode,
                    action=action,
                    pause_ms=self.pause_ms,
                    cacheable=cacheable,
                    use_cache=True,
                    clear_speech=self.clear_speech,
                    trim_silence=self.trim_silence or self.clear_speech,
                )
            except Exception as exc:
                self.player.stop()
                self.playback_finished_for_turn = True
                self.active_reply_id = None
                self.state = GateState.WAIT_CONTEXT
                err_res = make_playback_result(
                    session_id=sid,
                    turn_id=tid,
                    reply_id=reply_id,
                    status="failed",
                    error_code=f"synth_error:{type(exc).__name__}",
                    timing={"reply_request_mono_ms": t_req_mono_ms},
                )
                self._reply_results[dedup_key] = err_res
                self._record_log("EVENT_OUT", {"envelope": err_res})
                return err_res, None

            self.player.stop()
            self.playback_finished_for_turn = True
            self.active_reply_id = None
            self.state = GateState.WAIT_CONTEXT
            if not pipe_res.success:
                err_res = make_playback_result(
                    session_id=sid,
                    turn_id=tid,
                    reply_id=reply_id,
                    status="failed",
                    error_code=pipe_res.error_code or "device_error",
                    timing={"reply_request_mono_ms": t_req_mono_ms, "ttfa_ms": pipe_res.ttfa_ms},
                )
                self._reply_results[dedup_key] = err_res
                self._record_log("EVENT_OUT", {"envelope": err_res})
                return err_res, None

            self._record_log("PLAYBACK_START", {"session_id": sid, "turn_id": tid, "reply_id": reply_id, "mono_ms": pipe_res.playback_start_mono_ms})
            self._record_log("PLAYBACK_END", {"session_id": sid, "turn_id": tid, "reply_id": reply_id, "mono_ms": pipe_res.playback_end_mono_ms})
            ok_res = make_playback_result(
                session_id=sid,
                turn_id=tid,
                reply_id=reply_id,
                status="completed",
                timing={
                    "reply_request_mono_ms": t_req_mono_ms,
                    "playback_start_mono_ms": pipe_res.playback_start_mono_ms,
                    "playback_end_mono_ms": pipe_res.playback_end_mono_ms,
                    "ttfa_ms": pipe_res.ttfa_ms,
                    "tts_total_ms": pipe_res.total_synth_ms,
                    "audio_duration_s": pipe_res.audio_duration_s,
                    "underrun_count": pipe_res.underrun_count,
                    "cache_hit": pipe_res.cache_hit,
                    "backend": pipe_res.backend,
                    "pipelined": True,
                },
            )
            self._reply_results[dedup_key] = ok_res
            self._record_log("EVENT_OUT", {"envelope": ok_res})
            listen_ev = self._maybe_open_listening()
            return ok_res, listen_ev

        # 4. Synthesize speech offline
        try:
            synth_kwargs: Dict[str, Any] = {
                "text": reply_text,
                "output_wav": wav_path,
                "mode": self.mode,
                "action": action,
                "pause_ms": self.pause_ms,
                "cacheable": cacheable,
                "use_cache": True,
            }
            if self.clear_speech or self.trim_silence:
                synth_kwargs["clear_speech"] = self.clear_speech
                synth_kwargs["trim_silence"] = self.trim_silence
            synth_res = self.synthesizer.synthesize(**synth_kwargs)
            tts_ready_ms = monotonic_ms()
            self._record_log(
                "TTS_READY",
                {
                    "session_id": sid,
                    "turn_id": tid,
                    "reply_id": reply_id,
                    "wav_path": str(wav_path),
                    "cache_hit": synth_res.cache_hit,
                    "audio_duration_s": synth_res.audio_duration_s,
                    "tts_ms": synth_res.total_ms,
                    "rtf": synth_res.rtf,
                },
            )
        except Exception as exc:
            self.player.stop()
            self.playback_finished_for_turn = True
            self.active_reply_id = None
            self.state = GateState.WAIT_CONTEXT
            err_res = make_playback_result(
                session_id=sid,
                turn_id=tid,
                reply_id=reply_id,
                status="failed",
                error_code=f"synth_error:{type(exc).__name__}",
                timing={"reply_request_mono_ms": t_req_mono_ms},
            )
            self._reply_results[dedup_key] = err_res
            self._record_log("EVENT_OUT", {"envelope": err_res})
            return err_res, None

        # Validate generated WAV file before playing
        if not wav_path.exists() or os.path.getsize(wav_path) <= 44:
            self.player.stop()
            self.playback_finished_for_turn = True
            self.active_reply_id = None
            self.state = GateState.WAIT_CONTEXT
            err_res = make_playback_result(
                session_id=sid,
                turn_id=tid,
                reply_id=reply_id,
                status="failed",
                error_code="file_error:empty_or_missing",
                timing={"reply_request_mono_ms": t_req_mono_ms, "tts_ready_mono_ms": tts_ready_ms},
            )
            self._reply_results[dedup_key] = err_res
            self._record_log("EVENT_OUT", {"envelope": err_res})
            return err_res, None

        # 5. Play audio synchronously while mic stays closed
        self._record_log("PLAYBACK_QUEUED", {"session_id": sid, "turn_id": tid, "reply_id": reply_id})
        outcome = self.player.play(wav_path)
        if not outcome.success:
            self.player.stop()
            self.playback_finished_for_turn = True
            self.active_reply_id = None
            self.state = GateState.WAIT_CONTEXT
            err_res = make_playback_result(
                session_id=sid,
                turn_id=tid,
                reply_id=reply_id,
                status="failed",
                error_code=outcome.error_code or "device_error",
                timing={
                    "reply_request_mono_ms": t_req_mono_ms,
                    "tts_ready_mono_ms": tts_ready_ms,
                    "playback_start_mono_ms": outcome.playback_start_mono_ms,
                    "playback_end_mono_ms": outcome.playback_end_mono_ms,
                    "proxy_timing_note": outcome.proxy_timing_note,
                },
            )
            self._reply_results[dedup_key] = err_res
            self._record_log("EVENT_OUT", {"envelope": err_res})
            return err_res, None

        # 6. Confirm player stopped and emit PLAYBACK_RESULT(completed)
        self.player.stop()
        self.playback_finished_for_turn = True
        self.active_reply_id = None
        self.state = GateState.WAIT_CONTEXT

        timing_info = {
            "reply_request_mono_ms": t_req_mono_ms,
            "tts_ready_mono_ms": tts_ready_ms,
            "playback_start_mono_ms": outcome.playback_start_mono_ms,
            "playback_end_mono_ms": outcome.playback_end_mono_ms,
            "tts_total_ms": synth_res.total_ms,
            "audio_duration_s": synth_res.audio_duration_s,
            "rtf": synth_res.rtf,
            "cache_hit": synth_res.cache_hit,
            "backend": outcome.backend,
            "proxy_timing_note": outcome.proxy_timing_note,
        }
        self._record_log("PLAYBACK_START", {"session_id": sid, "turn_id": tid, "reply_id": reply_id, "mono_ms": outcome.playback_start_mono_ms})
        self._record_log("PLAYBACK_END", {"session_id": sid, "turn_id": tid, "reply_id": reply_id, "mono_ms": outcome.playback_end_mono_ms})

        ok_res = make_playback_result(
            session_id=sid,
            turn_id=tid,
            reply_id=reply_id,
            status="completed",
            timing=timing_info,
        )
        self._reply_results[dedup_key] = ok_res
        self._record_log("EVENT_OUT", {"envelope": ok_res})

        listen_ev = self._maybe_open_listening()
        return ok_res, listen_ev

    def dispatch_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Dispatch an incoming JSON event and return list of emitted events."""
        validate_event(event)
        ev_name = event["event"]
        emitted: List[Dict[str, Any]] = []
        if ev_name == "TURN_CONTEXT":
            listen_ev = self.handle_turn_context(event)
            if listen_ev is not None:
                emitted.append(listen_ev)
        elif ev_name == "REPLY_REQUEST":
            res_ev, listen_ev = self.handle_reply_request(event)
            if res_ev is not None:
                emitted.append(res_ev)
            if listen_ev is not None:
                emitted.append(listen_ev)
        else:
            self._record_log("IGNORED_EVENT", {"event": ev_name, "turn_id": event["turn_id"]})
        return emitted

    def dump_protocol_events(self, out_path: Path) -> None:
        """Write only schema_version=1 protocol envelopes in chronological order."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in self.trace_records:
                if "envelope" in rec:
                    f.write(to_json_line(rec["envelope"]) + "\n")
