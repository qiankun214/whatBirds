"""EXIF 路线测试。

标准标签（SubjectArea / SubjectLocation）用 Pillow 就能合成出来，所以这一层
可以完全离线、确定性地验证。厂商 MakerNote 需要真实相机原图，见
``tests/test_focus_samples.py``。
"""

from __future__ import annotations

import json

import pytest

from whatbird.core import analyze_focus
from whatbird.core.focus.exif_focus import (
    apply_orientation,
    orientation_size,
)


# ------------------------------------------------------------------ 适配层
@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_apply_orientation_matches_pillow(orientation):
    """8 种朝向的映射必须和 ``ImageOps.exif_transpose`` 逐像素一致。

    直接拿 Pillow 的实际结果当基准：标一个亮点、转正、再看亮点跑到哪 ——
    比手写期望值可靠得多（手写的那一版就写错了 4 个）。
    """
    import numpy as np
    from PIL import Image, ImageOps

    width, height, x, y = 100, 80, 10, 20
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[y, x] = 255
    image = Image.fromarray(canvas)
    exif = Image.Exif()
    exif[0x0112] = orientation
    image.info["exif"] = exif.tobytes()

    upright = np.asarray(ImageOps.exif_transpose(image))
    found_y, found_x = np.unravel_index(int(np.argmax(upright)), upright.shape)

    assert apply_orientation(x, y, width, height, orientation) == (int(found_x), int(found_y))
    assert orientation_size(width, height, orientation) == upright.shape[::-1]


def test_apply_orientation_identity_for_unknown_value():
    assert apply_orientation(10, 20, 100, 80, 0) == (10, 20)


@pytest.mark.parametrize("orientation,size", [(1, (100, 80)), (3, (100, 80)), (6, (80, 100)), (8, (80, 100))])
def test_orientation_size(orientation, size):
    assert orientation_size(100, 80, orientation) == size


# ------------------------------------------------------------------ 标准标签
def test_subject_area_rectangle_center(exif_factory):
    # 矩形 (100, 50, 200, 100) → 中心 (200, 100)
    path = exif_factory(name="rect.jpg", size=(400, 300), subject_area=(100, 50, 200, 100))

    report = analyze_focus(path)

    assert report.exif.found
    assert (report.exif.x, report.exif.y) == (200, 100)
    assert "SubjectArea" in report.exif.detail
    assert report.exif.source_tags["tag"] == "Exif.Photo.SubjectArea"


def test_subject_area_single_point(exif_factory):
    path = exif_factory(name="point.jpg", size=(400, 300), subject_area=(120, 240))

    report = analyze_focus(path)

    assert report.exif.found
    assert (report.exif.x, report.exif.y) == (120, 240)
    assert "点" in report.exif.detail


def test_subject_area_circle(exif_factory):
    path = exif_factory(name="circle.jpg", size=(400, 300), subject_area=(300, 90, 60))

    report = analyze_focus(path)

    assert report.exif.found
    assert (report.exif.x, report.exif.y) == (300, 90)
    assert "圆" in report.exif.detail


def test_subject_area_normalised_values(exif_factory):
    """有些机型写 0~1 的比例而不是像素。"""
    path = exif_factory(name="norm.jpg", size=(400, 300), subject_area=(0.25, 0.5))

    report = analyze_focus(path)

    assert report.exif.found
    assert (report.exif.x, report.exif.y) == (100, 150)


def test_subject_area_scaled_by_pixel_dimension(exif_factory):
    """EXIF 标称尺寸与实际存储尺寸不同时，要按比例换算。"""
    path = exif_factory(
        name="scaled.jpg",
        size=(400, 300),
        subject_area=(100, 100, 200, 100),
        pixel_dimension=(800, 600),
    )

    report = analyze_focus(path)

    assert report.exif.found
    # 中心 (200,150) 在 800x600 的标称坐标系里 → 实际 400x300 的一半
    assert (report.exif.x, report.exif.y) == (100, 75)


def test_subject_location_legacy_tag(exif_factory):
    path = exif_factory(name="legacy.jpg", size=(400, 300), subject_location=(80, 60))

    report = analyze_focus(path)

    assert report.exif.found
    assert (report.exif.x, report.exif.y) == (80, 60)
    assert "SubjectLocation" in report.exif.detail


def test_subject_area_preferred_over_legacy(exif_factory):
    path = exif_factory(
        name="both.jpg", size=(400, 300), subject_area=(200, 150), subject_location=(10, 10)
    )

    report = analyze_focus(path)

    assert (report.exif.x, report.exif.y) == (200, 150)


def test_out_of_range_subject_area_is_rejected(exif_factory):
    """离谱的坐标宁可不用，也不给一个明显错误的点。"""
    path = exif_factory(name="bad.jpg", size=(400, 300), subject_area=(999999, 999999))

    report = analyze_focus(path)

    assert report.exif.found is False
    assert report.exif.detail


# ------------------------------------------------------------------ 朝向
def test_orientation_is_applied_to_exif_point(exif_factory):
    """竖拍（Orientation=6）时，EXIF 坐标必须跟转正后的图对齐。"""
    path = exif_factory(
        name="rot.jpg", size=(400, 300), orientation=6, subject_area=(40, 60)
    )

    report = analyze_focus(path)

    assert report.exif.found
    # 转正后尺寸变成 300x400；存储方向 (40,60) → (300-1-60, 40) = (239, 40)
    assert (report.width, report.height) == (300, 400)
    assert (report.exif.x, report.exif.y) == (239, 40)


# ------------------------------------------------------------------ 降级与隐私
def test_missing_exif_gives_a_chinese_reason(jpeg_factory):
    path = jpeg_factory(name="noexif.jpg", size=(320, 240))

    report = analyze_focus(path)

    assert report.exif.found is False
    assert "没有可用的 EXIF 对焦信息" in report.exif.detail
    assert report.notes, "没有 EXIF 时应在 notes 里说明"


def test_gps_is_never_leaked(exif_factory):
    """隐私回归：EXIF 里的 GPS 绝不能出现在任何对外输出里。"""
    path = exif_factory(name="located.jpg", size=(400, 300), subject_area=(100, 100), gps=True)

    report = analyze_focus(path)
    payload = report.to_dict()

    assert "GPS" not in json.dumps(payload, ensure_ascii=False)
    assert set(report.exif.source_tags) <= {"make", "model", "tag", "raw_value"}
    # 整份报告里不该出现任何 GPS 相关的键
    assert not any("gps" in key.lower() for key in payload)


def test_disabling_exif_route_is_honoured(exif_factory):
    from whatbird.core import FocusParams

    path = exif_factory(name="off.jpg", size=(400, 300), subject_area=(100, 100))

    report = analyze_focus(path, params=FocusParams(use_exif=False))

    assert report.exif.found is False
    assert "关闭" in report.exif.detail


# ------------------------------------------------------------------ 两条路线的一致性
def test_agree_when_exif_point_is_inside_focus_region(sharp_patch_factory):
    """构造一张"清晰块在正中 + EXIF 也指向正中"的图，两条路线应当互相印证。"""
    from PIL import Image

    path, want = sharp_patch_factory(name="agree.jpg")
    # 把清晰块的中心写进 EXIF（坐标是存储方向的像素）
    tagged = path.parent / "agree-tagged.jpg"
    image = Image.open(path)
    exif = Image.Exif()
    exif[0x8769] = {0x9214: (want[0], want[1])}
    image.save(tagged, format="JPEG", quality=95, exif=exif)

    report = analyze_focus(tagged)

    assert report.exif.found and report.content.found
    assert report.agree is True
    assert "一致" in report.summary_line()


def test_agree_is_false_when_routes_disagree(sharp_patch_factory):
    from PIL import Image

    path, _ = sharp_patch_factory(name="disagree.jpg")
    tagged = path.parent / "disagree-tagged.jpg"
    image = Image.open(path)
    exif = Image.Exif()
    exif[0x8769] = {0x9214: (5, 5)}  # 指到角落，和正中的清晰块不符
    image.save(tagged, format="JPEG", quality=95, exif=exif)

    report = analyze_focus(tagged)

    assert report.exif.found and report.content.found
    assert report.agree is False
    assert "不一致" in report.summary_line()
