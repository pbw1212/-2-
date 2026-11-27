"""S7 文本时间线测试。"""

from __future__ import annotations

import json

import pytest

from b_module import textsim


def test_text_at_before_first_event():
    timeline = textsim.TextTimeline.from_entries([[300, "我"]])
    assert timeline.text_at(0) == ""
    assert timeline.text_at(299) == ""
    assert timeline.changed_at(0) is None


def test_text_at_exact_boundary():
    timeline = textsim.TextTimeline.from_entries([[300, "我"], [600, "我想"]])
    assert timeline.text_at(300) == "我"
    assert timeline.text_at(599) == "我"
    assert timeline.text_at(600) == "我想"


def test_changed_at_tracks_latest_change():
    timeline = textsim.TextTimeline.from_entries([[300, "我"], [600, "我想"]])
    assert timeline.changed_at(450) == 300
    assert timeline.changed_at(900) == 600


def test_revision_rewrites_text():
    """识别回改：后面的条目可以改写已有文本。"""
    timeline = textsim.TextTimeline.from_entries(
        [[300, "我想约"], [500, "我想约康付"], [620, "我想约康复科"]]
    )
    assert timeline.text_at(550) == "我想约康付"
    assert timeline.text_at(700) == "我想约康复科"


def test_duplicate_text_is_not_a_change():
    timeline = textsim.TextTimeline.from_entries([[300, "我"], [400, "我"], [500, "我想"]])
    assert len(timeline) == 2
    assert timeline.changed_at(450) == 300


def test_same_timestamp_conflict_rejected():
    with pytest.raises(ValueError):
        textsim.TextTimeline.from_entries([[300, "我"], [300, "我想"]])


def test_negative_time_rejected():
    with pytest.raises(ValueError):
        textsim.TextTimeline.from_entries([[-20, "我"]])


def test_empty_timeline():
    timeline = textsim.TextTimeline()
    assert len(timeline) == 0
    assert bool(timeline) is False
    assert timeline.text_at(9999) == ""
    assert timeline.changed_at(9999) is None


def test_load_timelines_file(tmp_path):
    path = tmp_path / "tl.json"
    path.write_text(json.dumps({"_comment": "说明", "dev_01": [[0, ""], [300, "我想"]]},
                               ensure_ascii=False), encoding="utf-8")
    timelines = textsim.load_timelines(path)
    assert list(timelines) == ["dev_01"]
    assert timelines["dev_01"].text_at(400) == "我想"


def test_missing_file_returns_empty(tmp_path):
    assert textsim.load_timelines(tmp_path / "absent.json") == {}


def test_timeline_for_unknown_case_is_empty():
    assert textsim.timeline_for("no_such_case", {}) is not None
    assert textsim.timeline_for("no_such_case", {}).text_at(500) == ""
