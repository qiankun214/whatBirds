"""画框标注图的测试：纯 Pillow 操作，不需要模型。"""

from __future__ import annotations

import io

from PIL import Image

from whatbird.core import Detection, DetectionReport, DetectionStyle, render_bird_overlay
from whatbird.core.detect.overlay import overlay_data_url, render_detection_overlay


def _report(*boxes, width=400, height=300, device="xpu"):
    report = DetectionReport(
        source_image="t.jpg",
        width=width,
        height=height,
        device=device,
        model_name="yolo11n.pt",
    )
    report.birds = [Detection(x=x, y=y, width=w, height=h, confidence=c) for x, y, w, h, c in boxes]
    return report


def _image(width=400, height=300):
    return Image.new("RGB", (width, height), (120, 170, 220))


def test_overlay_changes_pixels_and_keeps_size():
    image = _image()
    out = render_detection_overlay(image, _report((40, 60, 80, 120, 0.9)))
    assert out.size == image.size
    assert list(out.getdata()) != list(image.getdata())


def test_overlay_does_not_mutate_the_input_image():
    image = _image()
    before = list(image.getdata())
    render_detection_overlay(image, _report((40, 60, 80, 120, 0.9)))
    assert list(image.getdata()) == before


def test_overlay_works_without_detections():
    out = render_detection_overlay(_image(), _report())
    assert out.size == (400, 300)


def test_overlay_survives_tiny_and_out_of_range_boxes():
    out = render_detection_overlay(_image(), _report((0, 0, 1, 1, 0.5), (390, 290, 50, 50, 0.4)))
    assert out.size == (400, 300)


def test_overlay_style_can_disable_labels_and_legend():
    plain = render_detection_overlay(_image(), _report((40, 60, 80, 120, 0.9)), style=DetectionStyle(label=False, legend=False))
    decorated = render_detection_overlay(_image(), _report((40, 60, 80, 120, 0.9)))
    assert plain.size == decorated.size
    assert list(plain.getdata()) != list(decorated.getdata())


def test_render_bird_overlay_returns_decodable_png_at_original_size(jpeg_factory):
    path = jpeg_factory(name="bird.jpg", size=(320, 240))
    payload = render_bird_overlay(path, _report((10, 10, 30, 30, 0.8), width=320, height=240))
    assert payload.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (320, 240)


def test_render_bird_overlay_respects_exif_orientation(tmp_path):
    """竖拍照片（Orientation=6）标注图必须按转正后的尺寸输出，否则框会错位。"""
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (200, 100), (10, 120, 200))
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(path, format="JPEG", exif=exif)

    payload = render_bird_overlay(path, _report(width=100, height=200))
    with Image.open(io.BytesIO(payload)) as result:
        assert result.size == (100, 200)


def test_overlay_data_url_prefix():
    assert overlay_data_url(b"abc").startswith("data:image/png;base64,")
