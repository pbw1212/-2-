"""End-to-end integration tests covering A+B+C+D+E+LocalLLM unified closed loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b_module import paths as b_paths  # noqa: E402
from b_module import strategies as b_strategies  # noqa: E402
from b_module.a_bridge import AProtocolTranslator  # noqa: E402
from b_module.context import TurnContext  # noqa: E402
from b_module.detector import Detector, Tick  # noqa: E402

from e_module.audio_player import MockAudioPlayer  # noqa: E402
from integration.coordinator import FullSystemCoordinator  # noqa: E402
from integration.local_llm import LocalLLM  # noqa: E402


def _make_coord(tmp_path: Path, *, llm_enabled: bool = False, profile_id=None, player=None) -> FullSystemCoordinator:
    return FullSystemCoordinator(
        session_id="test-s01",
        runtime_dir=tmp_path,
        reference_date="2026-11-24",
        llm_enabled=llm_enabled,
        polish_replies=False,
        tts_backend="mock",
        player=player or MockAudioPlayer(),
        profile_id=profile_id,
    )


def test_full_registration_natural_date_and_sqlite_save(tmp_path: Path):
    mic_events = []
    coord = _make_coord(tmp_path)
    coord.e_service.on_mic_pause = lambda tid: mic_events.append(("pause", tid))
    coord.e_service.on_mic_flush_and_open = lambda tid: mic_events.append(("open", tid))

    coord.start(opening_greeting="您好，欢迎使用离线复诊登记助手。")
    # turn_id=0 opening must NOT open microphone; turn_id=1 opens once
    assert mic_events == [("open", 1)]

    coord.step_text("我想预约心内科，12月5号上午")
    assert not coord.is_closed
    assert coord.current_context["payload"]["dialog_phase"] == "confirming"
    assert coord.current_context["payload"]["question_type"] == "yes_no"

    coord.step_text("对的")
    assert coord.is_closed
    rows = coord.store.list_registrations()
    assert len(rows) == 1
    assert rows[0]["department"] == "cardio"
    assert rows[0]["visit_date"] == "2026-12-05"
    assert rows[0]["period"] == "morning"
    assert coord.store.has_duplicates() == []

    # Verify protocol invariants: LISTEN_START(t) emitted once per turn & never while SPEAKING
    events = [json.loads(line) for line in coord.log_path.read_text(encoding="utf-8").splitlines()]
    listen_turns = [e["turn_id"] for e in events if e["event"] == "LISTEN_START"]
    assert 0 not in listen_turns
    assert len(listen_turns) == len(set(listen_turns))
    coord.close()


def test_profile_inheritance_and_doctor_skip(tmp_path: Path):
    coord = _make_coord(tmp_path, profile_id="p001")
    coord.start(opening_greeting=None)
    coord.step_text("我下周四下午来复诊")
    # p001 inherited rehab + d_rehab_zhang -> straight to confirming!
    assert coord.current_context["payload"]["dialog_phase"] == "confirming"
    # Switch department -> clears incompatible rehab doctor and asks doctor slot!
    coord.step_text("改成骨科")
    assert coord.current_context["payload"]["expected_slot"] == "doctor"
    # Elderly patient says 不指定 -> SKIPPED cleanly -> moves to confirming
    coord.step_text("不指定")
    assert coord.current_context["payload"]["dialog_phase"] == "confirming"
    coord.step_text("好的")
    rows = coord.store.list_registrations()
    assert len(rows) == 1
    assert rows[0]["department"] == "ortho"
    assert rows[0]["doctor"] is None
    assert rows[0]["visit_date"] == "2026-12-03"
    coord.close()


def test_ambiguous_weekday_clarification_never_hardcodes_friday(tmp_path: Path):
    replies = []
    coord = _make_coord(tmp_path)
    coord.on_agent_reply = lambda action, txt: replies.append((action, txt))
    coord.start(opening_greeting=None)

    # 2026-11-24 is Tuesday; bare "周三" -> ambiguous [2026-11-25, 2026-12-02]
    coord.step_text("预约康复科周三上午")
    last_action, last_text = replies[-1]
    assert last_action == "ask_date"
    assert "周五" not in last_text
    assert "周三" in last_text

    coord.step_text("下周三")
    assert coord.current_context["payload"]["dialog_phase"] == "confirming"
    coord.step_text("对")
    rows = coord.store.list_registrations()
    assert len(rows) == 1
    assert rows[0]["visit_date"] == "2026-12-02"
    coord.close()


def test_expanded_faq_without_ai_boilerplate_and_resume(tmp_path: Path):
    replies = []
    coord = _make_coord(tmp_path)
    coord.on_agent_reply = lambda action, txt: replies.append((action, txt))
    coord.start(opening_greeting=None)

    coord.step_text("预约康复科12月5号下午")
    coord.step_text("复诊要带什么？")
    faq_action, faq_reply = replies[-1]
    assert faq_action == "faq_resume_confirm"
    assert "在本模拟流程中" not in faq_reply
    assert "身份证" in faq_reply
    assert "康复科" in faq_reply

    # Also test a newly added elderly FAQ topic
    coord.step_text("没带实体医保卡能刷电子医保码吗？")
    _, elec_reply = replies[-1]
    assert "电子凭证" in elec_reply or "二维码" in elec_reply

    coord.step_text("对")
    assert coord.store.count_registrations() == 1
    coord.close()


def test_cancel_writes_nothing(tmp_path: Path):
    coord = _make_coord(tmp_path)
    coord.start(opening_greeting=None)
    coord.step_text("康复科12月5号上午")
    coord.step_text("算了不约了")
    assert coord.is_closed
    assert coord.store.count_registrations() == 0
    coord.close()


def test_playback_failure_retries_and_no_early_mic_open(tmp_path: Path):
    player = MockAudioPlayer(fail_next_count=1)
    coord = _make_coord(tmp_path, player=player)
    coord.start(opening_greeting=None)
    out = coord.step_text("康复科12月5号下午")
    pb_events = [e for e in out if e["event"] == "PLAYBACK_RESULT"]
    assert [e["payload"]["status"] for e in pb_events] == ["failed", "completed"]
    assert pb_events[0]["payload"]["reply_id"] != pb_events[1]["payload"]["reply_id"]
    assert any(e["event"] == "LISTEN_START" for e in out)
    coord.close()


def test_b_targeted_strategy_avoids_filler_truncation_and_maps_timeout():
    cfg = b_paths.load_config()
    strat = b_strategies.build_strategy("adaptive_text", cfg)
    ctx_doctor = TurnContext("s01", 2, "targeted", ["doctor"])

    # Hesitation filler or incomplete surname should NOT cut off early at 450 ms
    d_filler = strat.decide(ctx_doctor, b_strategies.TextState(text="嗯", changed_at_ms=200, stable_ms=400))
    d_partial = strat.decide(ctx_doctor, b_strategies.TextState(text="张", changed_at_ms=200, stable_ms=400))
    d_complete = strat.decide(ctx_doctor, b_strategies.TextState(text="张医生", changed_at_ms=200, stable_ms=400))
    assert d_filler.basis != "text:short_answer"
    assert d_partial.basis != "text:short_answer"
    assert d_complete.basis == "text:short_answer"

    translator = AProtocolTranslator("s01")
    err_ev = translator.observe(
        {"type": "final", "status": "ok", "text": "噪声误字", "audio_ms": 30000.0},
        1,
        stop_reason="timeout",
    )[0]
    assert err_ev["payload"]["status"] == "error"
    assert err_ev["payload"]["final_text"] is None
    assert err_ev["payload"]["error_code"] == "endpoint_timeout"


@pytest.mark.skipif(not LocalLLM().is_available(), reason="Qwen2.5-0.5B GGUF model not present")
def test_local_llm_open_triage_preserves_draft_and_fallback_extracts(tmp_path: Path):
    replies = []
    coord = _make_coord(tmp_path, llm_enabled=True)
    coord.on_agent_reply = lambda action, txt: replies.append((action, txt))
    coord.start(opening_greeting=None)

    coord.step_text("帮我约骨科，后天下午")
    assert coord.current_context["payload"]["dialog_phase"] == "confirming"

    # Out-of-FAQ guidance query handled dynamically without reprompt kick-out
    coord.step_text("我膝盖有点疼不知道复查该找谁问下注意事项")
    assert not coord.is_closed
    assert coord.d_manager.session("test-s01").draft is not None

    coord.step_text("对的")
    assert coord.store.count_registrations() == 1
    coord.close()
