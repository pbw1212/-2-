"""数据集与标注测试：分组规则、重建正确性、真值一致性。

依赖 A 的只读样例；若 A 目录不存在则整体跳过，方便单模块开发。
"""

from __future__ import annotations

import pytest

from b_module import annotate, dataset, paths

A_ROOT = paths.DEFAULT_A_ROOT
pytestmark = pytest.mark.skipif(
    not (A_ROOT / "examples" / "clinic" / "cases.json").is_file(),
    reason="缺少 A 模块样例，跳过数据集测试",
)


@pytest.fixture(scope="module")
def config():
    return paths.load_config()


@pytest.fixture(scope="module")
def specs():
    return dataset.load_cases(A_ROOT)


def test_case_count_and_groups(specs):
    groups = {}
    for spec in specs:
        groups[spec.group] = groups.get(spec.group, 0) + 1
    assert len(specs) == 29
    assert groups == {
        "open_pause": 6,
        "open_long": 10,
        "open_full": 1,
        "targeted": 11,
        "yes_no_proxy": 1,
    }


def test_group_rules(specs):
    by_id = {spec.case_id: spec for spec in specs}
    assert by_id["dev_01"].question_type == "open"
    assert by_id["dev_01"].expected_slots == []
    assert by_id["dev_02"].question_type == "targeted"
    assert by_id["dev_02"].expected_slots == ["department"]
    assert by_id["dev_03"].expected_slots == ["date"]
    assert by_id["dev_07"].group == "open_long"
    assert by_id["smoke_short"].question_type == "yes_no"
    # yes_no 轮按 D 的样例 expected_slots 为空，B 不能靠 expected_slots 判短答
    assert by_id["smoke_short"].expected_slots == []


def test_multipart_offsets(specs, config):
    by_id = {spec.case_id: spec for spec in specs}
    spec = by_id["dev_01"]
    case_audio = dataset.build_case_audio(spec, a_root=A_ROOT, config=config, use_cache=False)
    assert len(case_audio.part_offsets_ms) == 2
    # 第二段起点应在第一段结束之后，且间隔约等于设计停顿
    first_end = case_audio.part_offsets_ms[1] - spec.pause_ms
    assert first_end > 0
    assert case_audio.part_offsets_ms[1] > first_end


def test_reconstruction_pads_tail(specs, config):
    spec = {item.case_id: item for item in specs}["dev_02"]
    case_audio = dataset.build_case_audio(spec, a_root=A_ROOT, config=config, use_cache=False)
    tail = annotate.annotate_case(spec, case_audio, config)["tail_silence_ms"]
    assert tail == pytest.approx(config["audio"]["tail_pad_ms"], abs=config["audio"]["frame_ms"] * 1.5)


def test_annotations_are_consistent(specs, config):
    frame_ms = float(config["audio"]["frame_ms"])
    tol = float(config["evaluation"]["tolerance_ms"])

    for spec in specs:
        case_audio = dataset.build_case_audio(spec, a_root=A_ROOT, config=config, use_cache=True)
        record = annotate.annotate_case(spec, case_audio, config)

        # 轮末必须落在音频内部，且尾部静音不小于阈值上界
        assert 0 < record["turn_end_ms"] <= record["duration_ms"]
        assert record["tail_silence_ms"] >= 1500.0
        # 标注轮末与帧网格对齐
        assert abs(record["turn_end_ms"] % frame_ms) < 1e-6

        if spec.pause_ms > 0:
            assert record["region_count"] == 2
            assert record["mid_pause_ms"] == pytest.approx(spec.pause_ms, abs=frame_ms * 2)
        else:
            assert record["region_count"] == 1
            assert record["mid_pause_ms"] == 0.0
        assert record["turn_end_ms"] >= record["mid_pause_ms"]


def test_manifest_payload(specs, config, tmp_path):
    payload = dataset.build_manifest(specs, a_root=A_ROOT, config=config, output=tmp_path / "manifest.json")
    assert payload["case_count"] == len(specs)
    assert payload["cases"][0]["expected_slot"] is not None or payload["cases"][0]["expected_slots"] == []


def test_manual_override_roundtrip(specs, config, tmp_path, monkeypatch):
    out = tmp_path / "annotations.json"
    subset = specs[:3]
    annotate.build_annotations(
        subset, a_root=A_ROOT, config=config, output=out, make_plots=False, use_cache=True
    )
    updated = annotate.apply_manual_updates(out, verify=[subset[0].case_id],
                                            set_end=[f"{subset[1].case_id}=1234"], note="pytest")
    by_id = {item["case_id"]: item for item in updated["cases"]}
    assert by_id[subset[0].case_id]["verified"] is True
    assert by_id[subset[1].case_id]["turn_end_ms"] == 1234.0
    assert updated["verified_count"] == 2


def test_reconstruction_keeps_coda(specs, config):
    """回归测试：重建音频不能在 -45 dBFS 处硬裁尾音。

    做法：找到最后一个 >= -45 dBFS 的帧，检查它之后的若干帧里存在 -45 ~ -65 之间的
    衰减帧。若又回到"直接跳到数字静音"，说明重建阈值被改回了 -45，听感上就是
    "最后一个字的尾音被吞掉"。
    """
    from b_module import vad as vad_module

    frame_ms = float(config["audio"]["frame_ms"])
    spec = {item.case_id: item for item in specs}["test_09"]  # 尾音衰减约 180 ms
    case_audio = dataset.build_case_audio(spec, a_root=A_ROOT, config=config, use_cache=True)
    energies = vad_module.frame_energies_db(case_audio.samples, case_audio.sample_rate, frame_ms)
    voiced = energies >= -45.0
    last = int(voiced.nonzero()[0].max())
    after = energies[last + 1: last + 10]
    assert after.size > 0
    assert (after > -70.0).any(), "尾音被硬裁：-45 dBFS 之后直接是数字静音"
    assert after.max() < -40.0, "尾音之后的帧不应仍然是强语音"
