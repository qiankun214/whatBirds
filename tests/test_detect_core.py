"""``detect_birds`` 的业务规则测试：全部用注入的假后端，不加载模型、不下载权重。"""

from __future__ import annotations

import pytest

from whatbird.core import DetectorDependencyError, ImageInputError, detect_birds
from whatbird.core.detect.models import BIRD_CLASS_ID, DetectionReport, DetectParams, RawBox

BIRD = RawBox(0.10, 0.20, 0.30, 0.60, 0.91, BIRD_CLASS_ID, "bird")
SMALL_BIRD = RawBox(0.50, 0.50, 0.60, 0.60, 0.42, BIRD_CLASS_ID, "bird")


class FakeDetector:
    """假后端：记录收到的图与参数，返回预设的归一化框。"""

    def __init__(self, boxes, device="xpu"):
        self.boxes = list(boxes)
        self.device = device
        self.calls = []
        self.params = None

    def __call__(self, image, params):
        self.calls.append(image)
        self.params = params
        return list(self.boxes), self.device


def test_detect_birds_returns_pixel_boxes_sorted_by_confidence(jpeg_factory):
    path = jpeg_factory(name="two.jpg", size=(200, 100))
    detector = FakeDetector([SMALL_BIRD, BIRD])

    report = detect_birds(path, detector=detector)

    assert isinstance(report, DetectionReport)
    assert report.width == 200 and report.height == 100
    assert report.count == 2
    assert report.device == "xpu"
    assert [item.confidence for item in report.birds] == [0.91, 0.42]

    first = report.birds[0]
    assert (first.x, first.y, first.width, first.height) == (20, 20, 40, 40)
    assert first.x_ratio == pytest.approx(0.1)
    assert first.y_ratio == pytest.approx(0.2)
    assert first.width_ratio == pytest.approx(0.2)
    assert first.height_ratio == pytest.approx(0.4)


def test_non_bird_classes_are_filtered_out_and_explained(jpeg_factory):
    path = jpeg_factory(name="mixed.jpg", size=(100, 100))
    detector = FakeDetector([BIRD, RawBox(0.5, 0.5, 0.9, 0.9, 0.99, 16, "dog")])

    report = detect_birds(path, detector=detector)

    assert report.count == 1
    assert report.birds[0].class_name == "bird"
    assert report.other_classes == ["dog"]
    assert any("dog" in note for note in report.notes)


def test_dog_only_image_reports_no_birds_but_lists_the_class(jpeg_factory, monkeypatch):
    path = jpeg_factory(name="dog.jpg")
    detector = FakeDetector([RawBox(0.1, 0.1, 0.5, 0.5, 0.8, 16, "dog")])

    report = detect_birds(path, detector=detector)

    assert report.count == 0
    assert report.found is False
    assert report.other_classes == ["dog"]
    assert any("未检测到鸟" in note for note in report.notes)


def test_empty_result_still_returns_a_usable_report(jpeg_factory):
    path = jpeg_factory(name="empty.jpg")
    report = detect_birds(path, detector=FakeDetector([]))
    assert report.count == 0
    assert report.summary_line() == "未检测到鸟"
    assert report.to_dict()["birds"] == []


def test_detector_receives_the_resolved_params(jpeg_factory):
    path = jpeg_factory(name="params.jpg")
    detector = FakeDetector([BIRD])
    params = DetectParams(imgsz=960, conf=0.4, device="cpu")

    detect_birds(path, params=params, detector=detector)

    assert detector.params is params
    assert detector.params.imgsz == 960
    assert detector.params.conf == 0.4


def test_out_of_range_boxes_are_clamped(jpeg_factory):
    path = jpeg_factory(name="clamp.jpg", size=(100, 100))
    detector = FakeDetector([RawBox(-0.2, -0.2, 1.4, 1.4, 0.7, BIRD_CLASS_ID, "bird")])

    report = detect_birds(path, detector=detector)
    bird = report.birds[0]

    assert bird.x >= 0 and bird.y >= 0
    assert bird.x + bird.width <= 100
    assert bird.y + bird.height <= 100


def test_bytes_source_is_supported(jpeg_factory):
    data = jpeg_factory(name="bytes.jpg").read_bytes()
    report = detect_birds(data, detector=FakeDetector([BIRD]))
    assert report.count == 1


def test_missing_file_raises_chinese_image_error():
    with pytest.raises(ImageInputError) as info:
        detect_birds(r"E:\definitely\missing\bird.jpg", detector=FakeDetector([BIRD]))
    assert "不存在" in info.value.message


def test_non_jpg_is_rejected(tmp_path):
    png = tmp_path / "bird.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ImageInputError):
        detect_birds(png, detector=FakeDetector([BIRD]))


def test_backend_dependency_error_is_not_swallowed(jpeg_factory):
    path = jpeg_factory(name="dep.jpg")

    def broken(image, params):
        raise DetectorDependencyError("缺少依赖", hint="装 ultralytics")

    with pytest.raises(DetectorDependencyError) as info:
        detect_birds(path, detector=broken)
    assert "缺少依赖" in info.value.message


def test_unexpected_backend_error_becomes_chinese_dependency_error(jpeg_factory):
    path = jpeg_factory(name="boom.jpg")

    def broken(image, params):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(DetectorDependencyError) as info:
        detect_birds(path, detector=broken)
    assert "RuntimeError" in info.value.message
    assert info.value.hint


def test_cpu_fallback_is_explained_in_notes(jpeg_factory, monkeypatch):
    path = jpeg_factory(name="cpu.jpg")
    monkeypatch.setattr("whatbird.core.detect.backend.xpu_available", lambda: False)

    report = detect_birds(path, detector=FakeDetector([BIRD], device="cpu"))

    assert report.device == "cpu"
    assert any("Intel 显卡" in note for note in report.notes)
