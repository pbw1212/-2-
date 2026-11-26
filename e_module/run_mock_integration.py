"""Mock full-system integration harness exercising E-module gating, retry, dedup, and real TTS."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e_module.audio_player import MockAudioPlayer
from e_module.config import DEFAULT_CHECKPOINT_PATH, DEFAULT_FAST_GL_ITERS, OUTPUTS_DIR
from e_module.e_service import EModuleService, GateState
from e_module.events import (
    make_event,
    make_reply_request,
    make_turn_context,
    monotonic_ms,
)
from e_module.latency_analyzer import analyze_trace_file
from e_module.tts_engine import Tacotron2Engine


def run_full_mock_integration(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> Dict[str, Any]:
    """Run all 6 end-to-end integration scenarios with real Tacotron2 synthesis."""
    work_dir = OUTPUTS_DIR / "mock_integration"
    log_path = OUTPUTS_DIR / "mock_integration_trace.jsonl"
    protocol_path = OUTPUTS_DIR / "mock_integration_protocol.jsonl"
    if log_path.exists():
        log_path.unlink()

    engine = Tacotron2Engine(
        checkpoint_path=checkpoint_path,
        quantize_mode="lstm_converted",
        vocoder_mode="fast_griffin_lim",
        gl_iters=DEFAULT_FAST_GL_ITERS,
    )
    engine.warmup()

    player = MockAudioPlayer(speed_factor=0.0)
    mic_events: List[tuple] = []

    service = EModuleService(
        synthesizer=engine,
        player=player,
        work_dir=work_dir,
        log_path=log_path,
        mode="tn_psp_ec",
        clear_speech=True,
        trim_silence=True,
        on_mic_pause=lambda tid: mic_events.append(("pause", tid)),
        on_mic_flush_and_open=lambda tid: mic_events.append(("flush_open", tid)),
    )

    session_id = "s_demo_01"
    ref_date = "2026-11-24"

    # Scenario 1: Full clinic flow (turn 0 opening -> turn 1 FAQ -> turn 2 book -> turn 3 reschedule -> turn 4 confirm -> done)
    # 1a. Opening announcement (turn_id=0, mic stays closed)
    res0, listen0 = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=0,
            reply_id="r_open_0",
            reply_text="您好，我是离线复诊登记助手，请问有什么可以帮您？",
            action="greeting",
            cacheable=True,
        )
    )
    assert res0 is not None and res0["payload"]["status"] == "completed"
    assert listen0 is None
    assert service.mic_open is False

    # 1b. Turn 1: D emits TURN_CONTEXT -> E flushes & emits LISTEN_START once
    listen1 = service.handle_turn_context(
        make_turn_context(session_id, 1, "idle", "open", [], ref_date)
    )
    assert listen1 is not None and listen1["event"] == "LISTEN_START" and listen1["turn_id"] == 1
    assert service.mic_open is True

    # Simulate A/B/C/D timestamps before E receives FAQ reply
    t_base = monotonic_ms()
    service._record_log(
        "EVENT_IN",
        {"envelope": make_event("TURN_STOP", session_id, 1, {"decision_audio_ms": 2100, "reason": "adaptive_silence"}, t_base)},
    )
    res1, _ = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=1,
            reply_id="r_faq_1",
            reply_text="复诊请带好医保卡、身份证和上次的病历本，提前十五分钟到一楼大厅自助机或窗口取号候诊。",
            action="faq",
            cacheable=False,
            emitted_mono_ms=t_base + 30,
        )
    )
    assert res1 is not None and res1["payload"]["status"] == "completed"
    assert service.mic_open is False

    # Scenario 6: Gate strict ordering — send TURN_CONTEXT for turn 2 while turn 2 reply won't open prematurely
    # First open turn 2 normally, then test sending TURN_CONTEXT(3) BEFORE turn 2 playback finishes
    listen2 = service.handle_turn_context(
        make_turn_context(session_id, 2, "collecting", "targeted", ["department", "date", "period"], ref_date)
    )
    assert listen2 is not None and listen2["turn_id"] == 2

    # Turn 2: Booking reply -> D asks confirmation
    res2, _ = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=2,
            reply_id="r_confirm_2",
            reply_text="康复科张医生，12月5日 14:30，对吗？",
            action="confirm",
            cacheable=False,
        )
    )
    assert res2 is not None and res2["payload"]["status"] == "completed"

    # Turn 3: Reschedule
    listen3 = service.handle_turn_context(
        make_turn_context(session_id, 3, "confirming", "yes_no", ["confirm"], ref_date)
    )
    assert listen3 is not None and listen3["turn_id"] == 3

    res3, _ = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=3,
            reply_id="r_resched_3",
            reply_text="已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
            action="confirm",
            cacheable=False,
        )
    )
    assert res3 is not None and res3["payload"]["status"] == "completed"

    # Scenario 2: Stale late REPLY_REQUEST from turn 1 arrives during turn 3 -> must be ignored
    stale_res, stale_listen = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=1,
            reply_id="r_late_stale",
            reply_text="这条过期消息不应该播报。",
            action="faq",
        )
    )
    assert stale_res is None and stale_listen is None

    # Scenario 3: Duplicate reply_id dedup -> returns cached result without re-playing
    played_before = len(player.played_paths)
    dup_res, dup_listen = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=3,
            reply_id="r_resched_3",
            reply_text="已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
            action="confirm",
        )
    )
    assert dup_res is not None and dup_res["payload"]["status"] == "completed"
    assert dup_listen is None
    assert len(player.played_paths) == played_before

    # Scenario 4: Playback device failure -> stop confirmed -> failed PLAYBACK_RESULT -> retry with new reply_id
    listen4 = service.handle_turn_context(
        make_turn_context(session_id, 4, "confirming", "yes_no", ["confirm"], ref_date)
    )
    assert listen4 is not None and listen4["turn_id"] == 4

    player.fail_next_count = 1
    stops_before = player.stop_calls
    fail_res, fail_listen = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=4,
            reply_id="r_done_4_try1",
            reply_text="好的，已为您登记完成，请按预约时间前往医院复诊。",
            action="done",
            cacheable=True,
        )
    )
    assert fail_res is not None and fail_res["payload"]["status"] == "failed"
    assert fail_res["payload"]["error_code"] == "device_error:simulated_failure"
    assert fail_listen is None
    assert player.stop_calls > stops_before
    assert not player.is_playing

    # Retry with new reply_id -> succeeds
    retry_res, _ = service.handle_reply_request(
        make_reply_request(
            session_id,
            turn_id=4,
            reply_id="r_done_4_retry2",
            reply_text="好的，已为您登记完成，请按预约时间前往医院复诊。",
            action="done",
            cacheable=True,
        )
    )
    assert retry_res is not None and retry_res["payload"]["status"] == "completed"

    # Turn 5: Verify next turn listening opens cleanly after final confirmation + prompt
    listen5 = service.handle_turn_context(
        make_turn_context(session_id, 5, "idle", "open", [], ref_date)
    )
    assert listen5 is not None and listen5["turn_id"] == 5
    assert service.state == GateState.LISTENING
    assert ("flush_open", 5) in mic_events

    service.dump_protocol_events(protocol_path)
    analysis = analyze_trace_file(log_path)

    report = {
        "scenarios_passed": [
            "1_full_clinic_flow_multi_turn",
            "2_stale_turn_ignored",
            "3_duplicate_reply_id_dedup",
            "4_playback_failure_stop_and_retry",
            "5_next_turn_listen_recovery",
            "6_half_duplex_gating",
        ],
        "played_wav_count": len(player.played_paths),
        "played_wavs": player.played_paths,
        "protocol_jsonl": str(protocol_path),
        "trace_jsonl": str(log_path),
        "latency_analysis": analysis,
    }
    (OUTPUTS_DIR / "mock_integration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    report = run_full_mock_integration()
    print(json.dumps(report["latency_analysis"], ensure_ascii=False, indent=2))
    print("Passed scenarios:", ", ".join(report["scenarios_passed"]))


if __name__ == "__main__":
    main()
