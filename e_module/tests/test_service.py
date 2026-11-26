"""Unit and state-machine tests for events, player gating, dedup, stale drops, and recovery."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from scipy.io import wavfile

from e_module.audio_player import MockAudioPlayer
from e_module.config import DEFAULT_PAUSE_MS, SAMPLE_RATE
from e_module.e_service import EModuleService, GateState
from e_module.events import (
    from_json_line,
    make_reply_request,
    make_turn_context,
    to_json_line,
)
from e_module.latency_analyzer import analyze_trace_file


@dataclass
class _StubSynthResult:
    wav_path: str
    sample_rate: int = SAMPLE_RATE
    num_samples: int = 2205
    audio_duration_s: float = 0.1
    total_ms: float = 5.0
    rtf: float = 0.05
    cache_hit: bool = False


class _StubSynthesizer:
    def __init__(self, fail_on_substring: Optional[str] = None) -> None:
        self.calls: list[str] = []
        self.fail_on_substring = fail_on_substring

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        mode: str = "tn_prosody",
        action: Optional[str] = None,
        pause_ms: int = DEFAULT_PAUSE_MS,
        cacheable: Optional[bool] = None,
        use_cache: bool = True,
    ) -> _StubSynthResult:
        if self.fail_on_substring and self.fail_on_substring in text:
            raise RuntimeError("simulated synthesis error")
        self.calls.append(text)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.zeros(2205, dtype=np.int16)
        wavfile.write(str(output_wav), SAMPLE_RATE, pcm)
        return _StubSynthResult(wav_path=str(output_wav))


def test_event_serialization_and_expected_slot_derivation():
    ev_single = make_turn_context("s01", 1, "collecting", "targeted", ["date"], "2026-11-24")
    assert ev_single["payload"]["expected_slot"] == "date"

    ev_multi = make_turn_context("s01", 2, "collecting", "open", ["department", "date"], "2026-11-24")
    assert ev_multi["payload"]["expected_slot"] is None

    line = to_json_line(ev_single)
    parsed = from_json_line(line)
    assert parsed == ev_single


def test_invalid_events_rejected():
    bad_ctx = {
        "schema_version": 1,
        "event": "TURN_CONTEXT",
        "session_id": "s01",
        "turn_id": 1,
        "emitted_mono_ms": 100,
        "payload": {
            "dialog_phase": "collecting",
            "question_type": "targeted",
            "expected_slots": ["date"],
            "expected_slot": "department",  # inconsistent
            "reference_date": "2026-11-24",
        },
    }
    with pytest.raises(ValueError):
        from_json_line(json.dumps(bad_ctx))


def test_gating_no_early_mic_open_during_playback(tmp_path: Path):
    synth = _StubSynthesizer()
    mic_log: list[tuple] = []

    class _CheckingPlayer(MockAudioPlayer):
        service_ref: Optional[EModuleService] = None

        def play(self, wav_path: Path):
            assert self.service_ref is not None
            # Mic MUST remain closed during playback and TURN_CONTEXT arriving mid-playback must NOT open mic
            assert self.service_ref.mic_open is False
            late_listen = self.service_ref.handle_turn_context(
                make_turn_context("s01", 2, "confirming", "yes_no", ["confirm"], "2026-11-24")
            )
            assert late_listen is None
            assert self.service_ref.mic_open is False
            return super().play(wav_path)

    player = _CheckingPlayer()
    svc = EModuleService(
        synthesizer=synth,
        player=player,
        work_dir=tmp_path,
        on_mic_pause=lambda tid: mic_log.append(("pause", tid)),
        on_mic_flush_and_open=lambda tid: mic_log.append(("flush_open", tid)),
    )
    player.service_ref = svc

    # Open turn 1
    l1 = svc.handle_turn_context(make_turn_context("s01", 1, "idle", "open", [], "2026-11-24"))
    assert l1 is not None and svc.mic_open is True

    # Reply for turn 1 -> inside play(), TURN_CONTEXT(turn=2) arrives early; mic stays closed until play ends
    res1, l2 = svc.handle_reply_request(make_reply_request("s01", 1, "r1", "请说日期。"))
    assert res1 is not None and res1["payload"]["status"] == "completed"
    assert l2 is not None and l2["turn_id"] == 2
    assert svc.mic_open is True
    assert ("pause", 1) in mic_log
    assert ("flush_open", 2) in mic_log

    # Sending TURN_CONTEXT(2) again must NOT emit duplicate LISTEN_START
    assert svc.handle_turn_context(make_turn_context("s01", 2, "confirming", "yes_no", ["confirm"], "2026-11-24")) is None


def test_stale_turn_and_duplicate_reply_id_dedup(tmp_path: Path):
    synth = _StubSynthesizer()
    player = MockAudioPlayer()
    svc = EModuleService(synthesizer=synth, player=player, work_dir=tmp_path)

    svc.handle_turn_context(make_turn_context("s01", 2, "collecting", "targeted", ["date"], "2026-11-24"))
    # Stale turn 1 request ignored
    r_stale, l_stale = svc.handle_reply_request(make_reply_request("s01", 1, "r_old", "过期消息"))
    assert r_stale is None and l_stale is None
    assert len(synth.calls) == 0

    # First execution of r2
    r2_a, _ = svc.handle_reply_request(make_reply_request("s01", 2, "r2", "请确认十二月五日。"))
    assert r2_a is not None and r2_a["payload"]["status"] == "completed"
    assert len(synth.calls) == 1

    # Duplicate r2 returns cached response without re-synthesizing or re-playing
    r2_b, _ = svc.handle_reply_request(make_reply_request("s01", 2, "r2", "请确认十二月五日。"))
    assert r2_b == r2_a
    assert len(synth.calls) == 1
    assert len(player.played_paths) == 1


def test_synth_and_playback_failure_recovery(tmp_path: Path):
    synth = _StubSynthesizer(fail_on_substring="BOOM")
    player = MockAudioPlayer()
    svc = EModuleService(synthesizer=synth, player=player, work_dir=tmp_path)

    svc.handle_turn_context(make_turn_context("s01", 1, "idle", "open", [], "2026-11-24"))

    # Synth failure stops player and notifies D
    err_synth, _ = svc.handle_reply_request(make_reply_request("s01", 1, "r_bad_synth", "BOOM触发异常"))
    assert err_synth is not None
    assert err_synth["payload"]["status"] == "failed"
    assert "synth_error" in err_synth["payload"]["error_code"]
    assert player.stop_calls >= 1

    # Playback failure stops player and allows retry with new reply_id
    player.fail_next_count = 1
    err_play, _ = svc.handle_reply_request(make_reply_request("s01", 1, "r_bad_play", "正常文本"))
    assert err_play is not None
    assert err_play["payload"]["status"] == "failed"
    assert err_play["payload"]["error_code"] == "device_error:simulated_failure"
    assert not player.is_playing

    ok_retry, _ = svc.handle_reply_request(make_reply_request("s01", 1, "r_good_retry", "正常文本"))
    assert ok_retry is not None and ok_retry["payload"]["status"] == "completed"

    # Analyze generated log
    report = analyze_trace_file(svc.log_path)
    assert report["turn_count"] >= 1
    assert svc.state in (GateState.WAIT_CONTEXT, GateState.LISTENING)
