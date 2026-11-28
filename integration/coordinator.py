"""Full-system event-driven coordinator integrating Modules A, B, C, D, E and LocalLLM."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
D_SRC_ROOT = PROJECT_ROOT / "d_module" / "02_D源代码"
A_PKG_ROOT = PROJECT_ROOT / "a_module"
for _p in (PROJECT_ROOT, D_SRC_ROOT, A_PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from b_module import detector as b_detector  # noqa: E402
from b_module import paths as b_paths  # noqa: E402
from b_module import protocol as b_protocol  # noqa: E402
from b_module import strategies as b_strategies  # noqa: E402
from b_module.a_bridge import AProtocolTranslator  # noqa: E402
from b_module.context import TurnContext as BTurnContext  # noqa: E402
from b_module.context import from_event as b_context_from_event  # noqa: E402

from c_module.nlu import CModule  # noqa: E402

from dmodule import events as d_ev  # noqa: E402
from dmodule.config import (  # noqa: E402
    PERIOD_DISPLAY,
    PolicyConfig,
    default_path as d_default_path,
    load_lexicon,
    load_policy,
    load_profiles,
)
from dmodule.policy import DialogManager, _md  # noqa: E402
from dmodule.store import RegistrationStore  # noqa: E402

from e_module.audio_player import AudioPlayer, MockAudioPlayer, create_default_player  # noqa: E402
from e_module.config import DEFAULT_FAST_GL_ITERS, DEFAULT_PAUSE_MS, OUTPUTS_DIR  # noqa: E402
from e_module.e_service import EModuleService  # noqa: E402
from e_module.events import make_reply_request, validate_event as e_validate_event  # noqa: E402

from integration.local_llm import LocalLLM  # noqa: E402

DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "integration" / "runtime"
DEFAULT_OPENING_GREETING = "您好，这里是离线复诊登记助手，请问您要预约复诊还是咨询流程？"


class InProcessABackend:
    """Wrap A's ModuleA with B's AProtocolTranslator using schema_version=1 events."""

    def __init__(
        self,
        session_id: str,
        *,
        vad_dbfs: float = -45.0,
        final_decoder: str = "hotwords",
        speaker_db_path: Optional[Path] = None,
        enable_speaker: bool = False,
        threads: int = 2,
    ) -> None:
        from module_a.asr import DEFAULT_ASR_MODEL, DEFAULT_HOTWORDS, Zipformer, load_hotwords
        from module_a.session import ModuleA
        from module_a.speaker import DEFAULT_MODEL as DEFAULT_CAM_MODEL, SpeakerMatcher

        self.session_id = session_id
        self.translator = AProtocolTranslator(session_id=session_id)
        self.recognizer = Zipformer(DEFAULT_ASR_MODEL, threads=threads)
        self.hotwords = load_hotwords(DEFAULT_HOTWORDS) if final_decoder == "hotwords" else {}
        self.speaker = None
        if enable_speaker and DEFAULT_CAM_MODEL.is_file():
            db = speaker_db_path or (DEFAULT_RUNTIME_DIR / "speaker_profiles.sqlite3")
            if db.is_file():
                self.speaker = SpeakerMatcher(DEFAULT_CAM_MODEL, db, vad_dbfs=vad_dbfs)
        self.module_a = ModuleA(
            self.recognizer,
            vad_dbfs=vad_dbfs,
            speaker=self.speaker,
            final_decoder=final_decoder,
            hotwords=self.hotwords if final_decoder == "hotwords" else None,
        )
        self._active_turn_id: Optional[int] = None
        self._active_a_turn: Optional[str] = None

    def set_vad_threshold(self, vad_dbfs: float) -> None:
        self.module_a.vad_dbfs = float(vad_dbfs)

    def start_turn(self, turn_id: int, context: BTurnContext, *, identify: bool = False) -> None:
        self._active_turn_id = turn_id
        a_turn = self.translator.map_turn_id(turn_id)
        self._active_a_turn = a_turn
        self.translator.reset()
        ctx_name = self.translator.context_name(context)
        if self.hotwords and ctx_name not in self.hotwords:
            ctx_name = "general"
        can_identify = bool(identify and self.speaker is not None)
        self.module_a.start_turn(a_turn, standby=False, identify=can_identify, context=ctx_name)

    def push_frame(self, pcm: bytes) -> List[Dict[str, Any]]:
        if self._active_turn_id is None or self._active_a_turn is None:
            return []
        raw_events = self.module_a.push_audio(self._active_a_turn, pcm)
        out: List[Dict[str, Any]] = []
        for ev_item in raw_events:
            out.extend(self.translator.observe(ev_item, self._active_turn_id, self._active_a_turn))
        return out

    def finish_turn(self, *, stop_reason: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._active_turn_id is None or self._active_a_turn is None:
            return []
        tid = self._active_turn_id
        a_tid = self._active_a_turn
        self._active_turn_id = None
        self._active_a_turn = None
        raw_events = self.module_a.finish_turn(a_tid)
        out: List[Dict[str, Any]] = []
        for ev_item in raw_events:
            out.extend(self.translator.observe(ev_item, tid, a_tid, stop_reason=stop_reason))
        return out

    def cancel_turn(self) -> None:
        if self._active_a_turn is not None:
            self.module_a.cancel_turn(self._active_a_turn)
        self._active_turn_id = None
        self._active_a_turn = None


class FullSystemCoordinator:
    """Orchestrates A -> B -> A -> C(+LLM) -> D(+NLG/SQLite) -> E(+Gating) closed loop."""

    def __init__(
        self,
        *,
        session_id: str = "s01",
        runtime_dir: Path = DEFAULT_RUNTIME_DIR,
        db_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
        e_trace_path: Optional[Path] = None,
        reference_date: Optional[str] = None,
        c_retriever: str = "bm25",
        b_strategy: str = "adaptive_text",
        llm_enabled: bool = True,
        polish_replies: bool = True,
        tts_backend: str = "real",
        player: Optional[AudioPlayer] = None,
        synthesizer: Optional[Any] = None,
        a_backend: Optional[Any] = None,
        llm: Optional[LocalLLM] = None,
        profile_id: Optional[str] = None,
        on_mic_pause: Optional[Callable[[int], None]] = None,
        on_mic_flush_and_open: Optional[Callable[[int], None]] = None,
        on_agent_reply: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.session_id = session_id
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path) if db_path else (self.runtime_dir / "registrations.sqlite")
        self.log_path = Path(log_path) if log_path else (self.runtime_dir / f"protocol_{session_id}.jsonl")
        self.e_trace_path = Path(e_trace_path) if e_trace_path else (self.runtime_dir / f"e_trace_{session_id}.jsonl")
        self.reference_date = reference_date or datetime.date.today().isoformat()
        self.polish_replies = bool(polish_replies)
        self.on_agent_reply = on_agent_reply

        # Clear fresh protocol log if explicitly initializing a new session run? Preserve append safety if desired.
        self.protocol_events: List[Dict[str, Any]] = []

        # 1. LLM layer (local Qwen2.5-0.5B GGUF with --llm off fallback)
        self.llm = llm if llm is not None else LocalLLM(enabled=llm_enabled)

        # 2. C Module (upgraded jieba+BM25/TF-IDF + flexible clinic NLU)
        self.c_module = CModule(retriever=c_retriever, segmenter="jieba")

        # 3. D Module (DialogManager + SQLite store)
        self.lexicon = load_lexicon(d_default_path("config", "lexicon.json"))
        self.profiles = load_profiles(d_default_path("config", "profiles.json"))
        self.policy_cfg = load_policy(d_default_path("config", "policy.json"))
        self.store = RegistrationStore(str(self.db_path))
        self.d_manager = DialogManager(
            self.store,
            self.lexicon,
            self.profiles,
            self.policy_cfg,
            reference_date=self.reference_date,
        )

        # 4. B Module (protocol-compliant Detector & strategy)
        self.b_config = b_paths.load_config()
        self.b_strategy = b_strategies.build_strategy(b_strategy, self.b_config)

        # 5. E Module (Tacotron2 INT8 + Clear Speech + Half-Duplex Gate)
        self.player = player if player is not None else create_default_player(mock=(tts_backend != "real"))
        if synthesizer is not None:
            self.synthesizer = synthesizer
        elif tts_backend == "mock":
            self.synthesizer = _LightweightMockSynthesizer()
        else:
            from e_module.tts_engine import Tacotron2Engine

            self.synthesizer = Tacotron2Engine(
                quantize_mode="lstm_converted",
                vocoder_mode="fast_griffin_lim",
                gl_iters=DEFAULT_FAST_GL_ITERS,
                max_decoder_steps=1000,
            )
            self.synthesizer.warmup()

        self.e_service = EModuleService(
            synthesizer=self.synthesizer,
            player=self.player,
            work_dir=self.runtime_dir / "e_wavs",
            log_path=self.e_trace_path,
            pause_ms=DEFAULT_PAUSE_MS,
            mode="tn_psp_ec",
            clear_speech=True,
            trim_silence=True,
            enable_pipelined=(tts_backend == "real" and synthesizer is None),
            on_mic_pause=on_mic_pause,
            on_mic_flush_and_open=on_mic_flush_and_open,
        )

        # 6. A Module (lazy or caller-injected)
        self.a_backend = a_backend

        # State tracking
        self.current_context: Optional[Dict[str, Any]] = None
        self.mic_open_for_turn: Optional[int] = None
        self._preset_profile_id = profile_id
        self._closed = False

    def _record_event(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        e_validate_event(envelope)
        self.protocol_events.append(envelope)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
        return envelope

    def start(self, *, opening_greeting: Optional[str] = DEFAULT_OPENING_GREETING) -> Dict[str, Any]:
        """Start session: optional turn_id=0 greeting (mic closed), then open turn_id=1."""
        self._closed = False
        if opening_greeting:
            req0 = make_reply_request(
                session_id=self.session_id,
                turn_id=0,
                reply_id=f"r-{self.session_id}-00-01",
                reply_text=opening_greeting,
                action="greeting",
                cacheable=True,
            )
            self._record_event(req0)
            if self.on_agent_reply:
                self.on_agent_reply("greeting", opening_greeting)
            pb0, listen0 = self.e_service.handle_reply_request(req0)
            if pb0 is not None:
                self._record_event(pb0)
            if listen0 is not None:
                # Defensive guard: turn_id=0 must never open microphone
                self._record_event(listen0)

        if self._preset_profile_id:
            prof_ev = d_ev.make_event(
                "PROFILE_SUGGESTION",
                self.session_id,
                0,
                {"profile_id": self._preset_profile_id, "match_status": "matched"},
            )
            self._record_event(prof_ev)
            self.d_manager.handle(prof_ev)

        ctx_list = self.d_manager.start_session(self.session_id)
        ctx1 = ctx_list[0]
        self._broadcast_turn_context(ctx1)
        return ctx1

    def _broadcast_turn_context(self, ctx_ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.current_context = ctx_ev
        self._record_event(ctx_ev)
        listen_ev = self.e_service.handle_turn_context(ctx_ev)
        if listen_ev is not None:
            self._record_event(listen_ev)
            self.mic_open_for_turn = listen_ev["turn_id"]
        return listen_ev

    @property
    def is_closed(self) -> bool:
        return self._closed or self.d_manager.session(self.session_id).closed

    def _required_facts_for_reply(self, state: Any, action: str) -> List[str]:
        draft = state.draft
        if draft is None:
            return []
        facts: List[str] = []
        if action in ("restate_confirm", "faq_resume_confirm", "retry_save", "saved", "duplicate_ack"):
            dep = draft.value("department")
            doc = draft.value("doctor")
            dt = draft.value("date")
            per = draft.value("period")
            if dep:
                facts.append(self.lexicon.department_name(dep))
            if doc:
                facts.append(self.lexicon.doctor_name(doc))
            if dt:
                facts.append(_md(dt))
            if per:
                facts.append(PERIOD_DISPLAY.get(per, per))
            if action in ("saved", "duplicate_ack") and draft.registration_id:
                facts.append(draft.registration_id)
        return facts

    def _enhance_nlu(self, asr_final: Dict[str, Any], turn_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run C (BM25 + rules) and fall back cleanly to LocalLLM for natural speech & guidance."""
        nlu_ev = self.c_module.process(asr_final, turn_context)
        self._record_event(nlu_ev)

        asr_payload = asr_final["payload"]
        nlu_payload = nlu_ev["payload"]
        utterance = (asr_payload.get("final_text") or "").strip()

        if asr_payload.get("status") in ("no_input", "error") or not utterance:
            return nlu_ev

        needs_fallback = (
            nlu_payload.get("intent") == "unknown"
            or (nlu_payload.get("intent") == "faq" and nlu_payload.get("status") in ("no_match", "ambiguous"))
            or (nlu_payload.get("status") == "no_match" and not nlu_payload.get("slots"))
        )
        if not needs_fallback or not self.llm.is_available():
            return nlu_ev

        # Stage 1: Try strict JSON slot/intent extraction if rule engine missed natural phrasing
        extracted = self.llm.extract_nlu_fallback(utterance, turn_context)
        if extracted is not None:
            upgraded = d_ev.make_event("NLU_RESULT", self.session_id, asr_final["turn_id"], extracted)
            self._record_event(upgraded)
            return upgraded

        # Stage 2: Route true out-of-FAQ guidance / everyday elderly chat to dynamic LLM answer
        has_draft = self.d_manager.session(self.session_id).draft is not None
        answer = self.llm.answer_open_query(utterance, has_active_draft=has_draft)
        guided = d_ev.make_event(
            "NLU_RESULT",
            self.session_id,
            asr_final["turn_id"],
            {
                "intent": "faq",
                "status": "ok",
                "slots": {},
                "faq_id": "llm_guidance",
                "answer_text": answer,
            },
        )
        self._record_event(guided)
        return guided

    def _deliver_reply_then_context(self, d_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enforce half-duplex E->D completion gating before emitting next TURN_CONTEXT."""
        replies = [ev_item for ev_item in d_outputs if ev_item["event"] == "REPLY_REQUEST"]
        next_contexts = [ev_item for ev_item in d_outputs if ev_item["event"] == "TURN_CONTEXT"]
        emitted: List[Dict[str, Any]] = []
        d_state = self.d_manager.session(self.session_id)

        pending_replies = list(replies)
        last_completed = True

        while pending_replies:
            req = pending_replies.pop(0)
            req_payload = req["payload"]
            action = str(req_payload.get("action") or "ask_slot")
            raw_text = str(req_payload.get("reply_text") or "")

            if self.polish_replies and self.llm.is_available():
                facts = self._required_facts_for_reply(d_state, action)
                polished = self.llm.polish_reply(action, raw_text, facts)
                if polished and polished != raw_text:
                    req = {**req, "payload": {**req_payload, "reply_text": polished}}

            self._record_event(req)
            if self.on_agent_reply:
                self.on_agent_reply(action, str(req["payload"]["reply_text"]))

            pb_res, early_listen = self.e_service.handle_reply_request(req)
            if early_listen is not None:
                self._record_event(early_listen)
            if pb_res is None:
                continue

            self._record_event(pb_res)
            emitted.append(pb_res)
            last_completed = pb_res["payload"]["status"] == "completed"

            follow_ups = self.d_manager.handle(pb_res)
            for item in follow_ups:
                if item["event"] == "REPLY_REQUEST":
                    pending_replies.append(item)
                elif item["event"] == "TURN_CONTEXT":
                    next_contexts.append(item)

        self._closed = d_state.closed
        if not self._closed and last_completed and next_contexts:
            for ctx in next_contexts:
                listen_ev = self._broadcast_turn_context(ctx)
                emitted.append(ctx)
                if listen_ev is not None:
                    emitted.append(listen_ev)

        return emitted

    def process_asr_final(
        self,
        asr_final: Dict[str, Any],
        *,
        profile_suggestions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process ASR_FINAL (+ optional PROFILE_SUGGESTION) through C -> D -> E -> next turn."""
        if self.current_context is None:
            raise RuntimeError("Call start() before processing turns")
        if int(asr_final["turn_id"]) != int(self.current_context["turn_id"]):
            raise ValueError(
                f"Turn mismatch: ASR_FINAL turn_id={asr_final['turn_id']} != active {self.current_context['turn_id']}"
            )

        self._record_event(asr_final)
        for prof in (profile_suggestions or ()):
            self._record_event(prof)
            if prof["payload"].get("match_status") == "matched" or prof["payload"].get("profile_id"):
                self.d_manager.handle(prof)

        nlu_ev = self._enhance_nlu(asr_final, self.current_context)
        d_out = self.d_manager.handle(nlu_ev)
        return self._deliver_reply_then_context(d_out)

    def step_text(self, user_text: str) -> List[Dict[str, Any]]:
        """Convenience turn driver for terminal/text mode without mic/ASR."""
        if self.current_context is None:
            raise RuntimeError("Call start() first")
        tid = int(self.current_context["turn_id"])
        clean = (user_text or "").strip()
        asr_final = d_ev.make_event(
            "ASR_FINAL",
            self.session_id,
            tid,
            {
                "status": "ok" if clean else "no_input",
                "final_text": clean if clean else None,
                "start_audio_ms": 0,
                "end_audio_ms": 0 if not clean else 500,
            },
        )
        return self.process_asr_final(asr_final)

    def run_pcm_turn(
        self,
        pcm_frames: Iterable[bytes],
        *,
        identify: bool = False,
        on_frame: Optional[Callable[[], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run one complete voice turn: A -> ASR_TICK -> B -> TURN_STOP -> A finish -> C -> D -> E."""
        if self.a_backend is None:
            raise RuntimeError("A backend is required for audio turns")
        if self.current_context is None:
            raise RuntimeError("Call start() first")

        tid = int(self.current_context["turn_id"])
        b_ctx = b_context_from_event(self.current_context)
        det = b_detector.Detector(self.session_id, tid, self.b_config, self.b_strategy, b_ctx)

        self.a_backend.start_turn(tid, b_ctx, identify=identify)
        stop = None

        for pcm in pcm_frames:
            if on_frame:
                on_frame()
            events = self.a_backend.push_frame(pcm)
            for item in events:
                if item["event"] == "ASR_TICK":
                    p = item["payload"]
                    tick = b_detector.Tick(
                        audio_ms=float(p["audio_ms"]),
                        is_speech=bool(p["is_speech"]),
                        partial_text=str(p.get("partial_text") or ""),
                        partial_updated_ms=p.get("partial_updated_ms"),
                        turn_id=tid,
                    )
                    stop = det.on_tick(tick)
                    if stop is not None:
                        break
            if stop is not None:
                self._record_event(stop.to_event())
                break

        stop_reason = stop.reason if stop is not None else None
        final_events = self.a_backend.finish_turn(stop_reason=stop_reason)
        asr_final = next((e for e in final_events if e["event"] == "ASR_FINAL"), None)
        if asr_final is None:
            raise RuntimeError("A backend did not emit ASR_FINAL on finish_turn")
        profs = [e for e in final_events if e["event"] == "PROFILE_SUGGESTION"]
        return self.process_asr_final(asr_final, profile_suggestions=profs)

    def run_wav_turn(self, wav_path: Path, *, identify: bool = False, tail_silence_ms: int = 2000) -> List[Dict[str, Any]]:
        """Feed one WAV file (plus trailing silence so B reaches endpoint naturally) through A->B->A->C->D->E."""
        from module_a.audio import FRAME_BYTES, SAMPLE_RATE, frames, read_wav

        pcm = read_wav(Path(wav_path))
        tail_samples = int(round(SAMPLE_RATE * tail_silence_ms / 1000.0))
        padded = pcm + b"\x00" * (tail_samples * 2)
        chunked = [
            chunk if len(chunk) == FRAME_BYTES else chunk + b"\x00" * (FRAME_BYTES - len(chunk))
            for chunk in frames(padded)
        ]
        return self.run_pcm_turn(chunked, identify=identify)

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass


class _LightweightMockSynthesizer:
    """Fast valid WAV synthesizer for CI / text-only integration tests."""

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        mode: str = "tn_psp_ec",
        action: Optional[str] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        cacheable: Optional[bool] = None,
        use_cache: bool = True,
        clear_speech: bool = False,
        trim_silence: bool = False,
    ) -> Any:
        import numpy as np
        from scipy.io import wavfile
        from e_module.tts_frontend import process_text

        fe = process_text(text, mode=mode, action=action, pause_ms=pause_ms)
        sr = 22050
        samples = np.zeros(int(sr * 0.25), dtype=np.int16)
        samples[100:200] = 1500
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(output_wav), sr, samples)
        return _MockSynthMetrics(str(output_wav), sr, len(samples), fe.to_dict())


class _MockSynthMetrics:
    def __init__(self, wav_path: str, sr: int, num_samples: int, trace: Dict[str, Any]) -> None:
        self.wav_path = wav_path
        self.sample_rate = sr
        self.num_samples = num_samples
        self.audio_duration_s = round(num_samples / sr, 4)
        self.total_ms = 5.0
        self.rtf = 0.02
        self.cache_hit = False
        self.trace = trace
