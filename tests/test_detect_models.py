"""检测数据契约的单元测试（不需要显卡、不需要权重、不需要 ultralytics）。"""

from __future__ import annotations

import pytest

from whatbird.core.detect.models import (
    BIRD_CLASS_ID,
    COCO_BIRD_CLASSES,
    Detection,
    DetectionReport,
    DetectParams,
    RawBox,
    build_detection,
    sort_by_confidence,
    unique_class_names,
    xyxy_to_xywh,
)


def test_default_params_only_want_birds():
    params = DetectParams()
    assert params.classes == COCO_BIRD_CLASSES == (BIRD_CLASS_ID,)
    assert params.device == "auto"
    assert params.conf > 0


def test_params_from_dict_tolerates_unknown_and_coerces_classes():
    params = DetectParams.from_dict({"imgsz": 960, "classes": [14, 16], "device": "cpu", "unknown": 1})
    assert params.imgsz == 960
    assert params.classes == (14, 16)
    assert params.device == "cpu"


def test_params_from_dict_none_classes_means_no_filter():
    assert DetectParams.from_dict({"classes": None}).classes == ()


def test_params_to_dict_exposes_resolved_model_dir(tmp_path):
    payload = DetectParams(model_dir=tmp_path).to_dict()
    assert payload["model_dir"] == str(tmp_path)
    assert payload["classes"] == [BIRD_CLASS_ID]


def test_xyxy_to_xywh_clamps_to_image():
    assert xyxy_to_xywh(-50, -50, 9999, 9999, width=100, height=50) == (0, 0, 100, 50)
    assert xyxy_to_xywh(10, 10, 10, 10, width=100, height=50) == (10, 10, 1, 1)


def test_build_detection_maps_normalized_box_to_pixels():
    detection = build_detection(RawBox(0.1, 0.2, 0.4, 0.6, 0.87654), width=200, height=100)
    assert (detection.x, detection.y, detection.width, detection.height) == (20, 20, 60, 40)
    assert detection.x_ratio == pytest.approx(0.1)
    assert detection.height_ratio == pytest.approx(0.4)
    assert detection.confidence == pytest.approx(0.87654)
    assert detection.box == (20, 20, 80, 60)
    assert detection.to_dict()["confidence"] == 0.8765


def test_build_detection_clamps_out_of_range_box():
    detection = build_detection(RawBox(-0.5, 0.5, 1.5, 2.0, 0.5), width=100, height=100)
    assert detection.x == 0
    assert detection.x + detection.width <= 100
    assert detection.y + detection.height <= 100


def test_sort_by_confidence_is_descending_and_stable_on_ties():
    low = Detection(x=0, y=0, width=10, height=10, confidence=0.2)
    high = Detection(x=0, y=0, width=10, height=10, confidence=0.9)
    assert [item.confidence for item in sort_by_confidence([low, high])] == [0.9, 0.2]


def test_unique_class_names_sorted_and_deduped():
    assert unique_class_names(["dog", "cat", "dog", ""]) == ["cat", "dog"]


def test_report_to_dict_shape():
    report = DetectionReport(source_image="a.jpg", width=100, height=50, device="cpu")
    report.birds = [Detection(x=1, y=2, width=3, height=4, confidence=0.5)]
    payload = report.to_dict()
    assert payload["count"] == 1
    assert payload["found"] is True
    assert payload["image"] == {"width": 100, "height": 50}
    assert payload["birds"][0]["x"] == 1
    assert set(payload) == {
        "source_image",
        "image",
        "count",
        "found",
        "birds",
        "other_classes",
        "device",
        "model_name",
        "elapsed_seconds",
        "notes",
        "params",
    }


def test_report_empty_state():
    report = DetectionReport(source_image="a.jpg", width=100, height=50)
    assert report.count == 0
    assert report.found is False
    assert report.top is None
    assert report.summary_line() == "未检测到鸟"
