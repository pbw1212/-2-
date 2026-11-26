"""Unit tests for the E-module Chinese clinic TTS frontend."""

import re
import pytest

from e_module.tts_frontend import (
    digits_to_spoken,
    int_to_chinese,
    normalize_text,
    process_text,
)


def test_digits_and_integers():
    assert digits_to_spoken("2026") == "二零二六"
    assert int_to_chinese(0) == "零"
    assert int_to_chinese(5) == "五"
    assert int_to_chinese(12) == "十二"
    assert int_to_chinese(15) == "十五"
    assert int_to_chinese(20) == "二十"
    assert int_to_chinese(105) == "一百零五"
    assert int_to_chinese(1204) == "一千二百零四"


def test_iso_and_cn_dates():
    norm, rules, _ = normalize_text("改到2026-12-04")
    assert "二零二六年十二月四日" in norm
    assert "iso_date" in rules

    norm2, rules2, _ = normalize_text("12月5日")
    assert norm2 == "十二月五日"
    assert "cn_month_day" in rules2


def test_time_normalization_and_existing_period():
    norm, rules, _ = normalize_text("12月5日 14:30")
    assert norm == "十二月五日 下午两点半"
    assert "time_hhmm" in rules

    norm_am, _, _ = normalize_text("08:05")
    assert norm_am == "上午八点零五分"

    norm_noon, _, _ = normalize_text("12:00")
    assert norm_noon == "中午十二点整"

    # Should NOT duplicate period when text already says 上午/下午
    norm_no_dup, _, _ = normalize_text("上午08:05到11:30，下午13:30到16:30")
    assert "上午上午" not in norm_no_dup
    assert "下午下午" not in norm_no_dup
    assert "上午八点零五分到十一点半,下午一点半到四点半" == norm_no_dup


def test_room_floor_percentage_decimal():
    norm, _, _ = normalize_text("请在3号诊室门口候诊，102室二楼，提前15分钟，准确率98.5%")
    assert "三号诊室" in norm
    assert "一零二室" in norm
    assert "十五分钟" in norm
    assert "百分之九十八点五" in norm


def test_english_period_mapping():
    norm, _, _ = normalize_text("您预约的是morning还是afternoon？")
    assert "上午" in norm
    assert "下午" in norm
    assert norm.endswith("?")


def test_punctuation_mapping_to_ascii():
    norm, rules, _ = normalize_text("您好！请问您想约哪天？康复科、骨科。")
    assert norm == "您好!请问您想约哪天?康复科,骨科."
    assert "punct_map" in rules


def test_neutral_tone_and_medical_polyphones():
    res = process_text("请您复诊时重说一下大夫姓名，对吗？", mode="tn_only")
    pinyin = res.segments[0].pinyin
    assert "fu4 zhen3" in pinyin
    assert "chong2 shuo1" in pinyin
    assert "dai4 fu5" in pinyin
    assert "dui4 ma5" in pinyin


def test_yi_bu_and_third_tone_sandhi():
    res = process_text("可以不用一楼，不对，一会儿就好。", mode="tn_only")
    pinyin = res.segments[0].pinyin
    assert "ke2 yi3" in pinyin
    assert "bu2 yong4" in pinyin
    assert "yi1 lou2" in pinyin
    assert "bu2 dui4" in pinyin


def test_confirm_semantic_grouping_and_pauses():
    res = process_text(
        "康复科张医生，12月5日 14:30，对吗？",
        mode="tn_prosody",
        action="confirm",
        pause_ms=260,
    )
    assert len(res.segments) == 3
    assert "康复科张医生" in res.segments[0].text
    assert res.segments[0].pause_after_ms == 260
    assert "十二月五日" in res.segments[1].text
    assert "下午两点半" in res.segments[1].text
    assert res.segments[1].pause_after_ms == 260
    assert res.segments[2].text.startswith("对吗")
    assert res.segments[2].pause_after_ms == 0
    for seg in res.segments:
        assert "#3" in seg.pinyin


def test_confirm_reschedule_grouping():
    res = process_text(
        "已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
        mode="tn_prosody",
        action="confirm",
        pause_ms=280,
    )
    assert len(res.segments) >= 3
    assert any("二零二六年十二月四日" in s.text for s in res.segments)
    assert res.segments[-1].text.startswith("对吗")
    assert res.segments[-1].pause_after_ms == 0


def test_long_faq_splitting_prevents_overlong_segments():
    long_text = "复诊请带好医保卡、身份证和上次的病历本，提前十五分钟到一楼大厅自助机或窗口取号候诊。"
    res = process_text(long_text, mode="tn_prosody", action="faq", pause_ms=260)
    assert len(res.segments) >= 3
    for seg in res.segments:
        # Each chunk kept short enough to avoid Tacotron2 attention drift
        cn_chars = re.findall(r"[\u4e00-\u9fff]", seg.text)
        assert len(cn_chars) <= 18
        assert "#1" in seg.pinyin or len(cn_chars) <= 3
        assert "#3" in seg.pinyin
    assert res.segments[-1].pause_after_ms == 0


def test_ablation_modes_behavior():
    text = "康复科张医生，12月5日 14:30，对吗？"
    raw = process_text(text, mode="baseline_raw")
    basic = process_text(text, mode="basic_pinyin")
    tn_only = process_text(text, mode="tn_only")
    prosody = process_text(text, mode="tn_prosody", action="confirm")

    # baseline_raw replicates character splitting
    assert len(raw.segments) > 5
    # basic_pinyin fixes line splitting into 1 segment without TN
    assert len(basic.segments) == 1
    assert "12" in basic.segments[0].pinyin
    # tn_only normalizes digits and tones without prosody markers
    assert len(tn_only.segments) == 1
    assert "#1" not in tn_only.segments[0].pinyin
    assert "#3" not in tn_only.segments[0].pinyin
    assert "shi2 er4 yue4" in tn_only.segments[0].pinyin
    # tn_prosody splits and marks #1/#3
    assert len(prosody.segments) == 3
    assert "#3" in prosody.segments[0].pinyin


def test_unknown_expression_fallback_recorded():
    norm, _, fallbacks = normalize_text("请联系★窗口处理")
    assert "★" in fallbacks
    res = process_text("请联系★窗口处理", mode="tn_prosody")
    assert "★" in res.fallback_spans
    assert len(res.segments) >= 1
    assert "chuang1 kou3" in res.segments[0].pinyin


def test_empty_input_raises():
    with pytest.raises(ValueError):
        process_text("   ", mode="tn_prosody")
