"""内容分析路线（传统图形学算法）测试。

核心断言：**清晰的那块在哪，对焦点就该在哪**。用合成样本就能拿到真值，
所以这些用例完全确定性，不依赖任何真实照片。
"""

from __future__ import annotations

import json

import pytest

from whatbird.core import FocusParams, analyze_focus
from whatbird.core.focus.content import analyze_content

POSITIONS = [
    ("center", (0.5, 0.5)),
    ("top-left", (0.2, 0.25)),
    ("bottom-right", (0.8, 0.75)),
    ("top-right", (0.75, 0.3)),
]

#: 允许的定位误差（像素）。实测在合成样本上是 0~1px，留足余量避免脆弱测试。
TOLERANCE = 20


@pytest.mark.parametrize("label,center", POSITIONS, ids=[item[0] for item in POSITIONS])
def test_locates_the_sharp_patch(sharp_patch_factory, label, center):
    path, want = sharp_patch_factory(name=f"{label}.jpg", center=center)

    report = analyze_focus(path)
    hit = report.content

    assert hit.found, hit.detail
    assert hit.confidence > 0.5
    distance = ((hit.x - want[0]) ** 2 + (hit.y - want[1]) ** 2) ** 0.5
    assert distance <= TOLERANCE, f"定位偏差 {distance:.1f}px（应 ≤ {TOLERANCE}）"


def test_region_covers_the_sharp_patch(sharp_patch_factory):
    path, want = sharp_patch_factory(name="region.jpg")

    report = analyze_focus(path)

    assert report.region is not None
    region = report.region
    assert region.contains(want[0], want[1], tolerance=8)
    # 清晰块占画面 96*96/(480*360) ≈ 5.3%
    assert 0.01 < region.area_ratio < 0.2


def test_main_candidate_is_the_headline_point(sharp_patch_factory):
    path, want = sharp_patch_factory(name="cands.jpg")

    report = analyze_focus(path)

    assert report.candidates, "至少应给出主对焦点"
    assert report.candidates[0].x == report.content.x
    assert report.candidates[0].y == report.content.y
    assert report.candidates[0].score == 1.0


def test_tenengrad_metric_also_locates_the_patch(sharp_patch_factory):
    path, want = sharp_patch_factory(name="tenengrad.jpg")

    report = analyze_focus(path, params=FocusParams(metric="tenengrad"))

    assert report.content.found
    distance = ((report.content.x - want[0]) ** 2 + (report.content.y - want[1]) ** 2) ** 0.5
    assert distance <= TOLERANCE


def test_blurred_high_contrast_edge_does_not_win():
    """回归防护：一团**严重失焦**的高对比大边缘，不能盖过一块**很低对比**的锐利小纹理。

    这是"直接取全局最锐像素"会翻车的典型场景，也是不做额外对比度归一化的依据。
    """
    import cv2
    import numpy as np

    width, height = 480, 360
    # 左半黑、右半白，再重度模糊 —— 高对比，但完全失焦
    base = np.zeros((height, width), np.float32)
    base[:, width // 2 :] = 255.0
    blurred = cv2.GaussianBlur(base, (0, 0), 10.0)
    image = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # 低对比（120 / 135）但锐利的棋盘块
    patch = np.zeros((96, 96), np.uint8)
    for y in range(0, 96, 4):
        for x in range(0, 96, 4):
            patch[y : y + 4, x : x + 4] = 120 if (x // 4 + y // 4) % 2 == 0 else 135
    image[130:226, 320:416] = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)

    result = analyze_content(image)

    assert result.hit.found
    distance = ((result.hit.x - 368) ** 2 + (result.hit.y - 178) ** 2) ** 0.5
    assert distance <= TOLERANCE, f"被失焦边缘带偏了 {distance:.1f}px"


def test_flat_image_reports_no_focus():
    import numpy as np

    flat = np.full((240, 320, 3), 120, dtype=np.uint8)

    result = analyze_content(flat)

    assert result.hit.found is False
    assert "清晰度" in result.hit.detail


def test_tiny_image_reports_no_focus():
    import numpy as np

    result = analyze_content(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.hit.found is False
    assert "太小" in result.hit.detail


def test_blurry_photo_is_not_confidently_focused():
    from PIL import Image, ImageFilter

    import numpy as np

    noisy = Image.effect_noise((400, 300), 50).convert("RGB")
    blurred = noisy.filter(ImageFilter.GaussianBlur(9))

    result = analyze_content(np.asarray(blurred))

    # 全图一样糊：要么明确说找不到，要么给一个很低的置信度
    assert result.hit.found is False or result.hit.confidence < 0.5


def test_large_image_maps_back_to_original_coordinates(sharp_patch_factory):
    from PIL import Image

    path, _ = sharp_patch_factory(name="small.jpg")
    big = path.parent / "big.jpg"
    Image.open(path).resize((3000, 2250), Image.LANCZOS).save(big, quality=92)

    report = analyze_focus(big)

    assert (report.width, report.height) == (3000, 2250)
    assert max(report.analyzed_width, report.analyzed_height) == 1024
    # 清晰块在正中 → 相对位置应贴近 0.5，且绝对坐标必须落在原图范围内
    assert abs(report.content.x_ratio - 0.5) < 0.02
    assert abs(report.content.y_ratio - 0.5) < 0.02
    assert 0 <= report.content.x < 3000
    assert 0 <= report.content.y < 2250


def test_result_is_deterministic(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="deterministic.jpg")

    first = analyze_focus(path)
    second = analyze_focus(path)

    assert (first.content.x, first.content.y) == (second.content.x, second.content.y)
    assert first.content.confidence == second.content.confidence


def test_report_is_json_serialisable(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="json.jpg")

    payload = analyze_focus(path).to_dict()

    text = json.dumps(payload, ensure_ascii=False)
    assert "content" in payload and "exif" in payload
    assert payload["image"]["width"] == 480
    assert "normalize_ratio" not in payload["params"]
    assert len(text) > 0


def test_disabling_content_route_is_honoured(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="off.jpg")

    report = analyze_focus(path, params=FocusParams(use_content=False))

    assert report.content.found is False
    assert "关闭" in report.content.detail
