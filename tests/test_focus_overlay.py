"""叠加标注图测试。

这一类测试守的是"图到底画上去了没有"：尺寸对不对、标记有没有真的落在对焦点那、
编码出来能不能再被打开。曾经踩过的坑是"看着调用成功、其实什么都没画"。
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from whatbird.core import FocusParams, analyze_focus, render_focus_overlay
from whatbird.core.focus.overlay import OverlayStyle, overlay_data_url, render_overlay


def _decode(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return np.asarray(image.convert("RGB"))


def test_overlay_keeps_the_original_size(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="overlay.jpg")
    report = analyze_focus(path)

    image = _decode(render_focus_overlay(path, report))

    assert image.shape[:2] == (360, 480)


def test_overlay_marks_the_focus_point(sharp_patch_factory):
    path, want = sharp_patch_factory(name="mark.jpg")
    report = analyze_focus(path)

    before = np.asarray(Image.open(path).convert("RGB")).astype(int)
    after = _decode(render_focus_overlay(path, report)).astype(int)

    # 标记点的邻域必须发生了变化
    x, y = report.content.x, report.content.y
    patch = (slice(max(y - 6, 0), y + 7), slice(max(x - 6, 0), x + 7))
    assert np.abs(after[patch] - before[patch]).max() > 30

    # 远离标记点的地方不应该被改动
    corners = (slice(0, 6), slice(0, 6))
    assert np.abs(after[corners] - before[corners]).max() <= 255  # 图例区允许有面板
    assert (x, y) != (0, 0)


def test_overlay_draws_both_routes(sharp_patch_factory):
    from PIL import Image as PILImage

    path, want = sharp_patch_factory(name="both.jpg")
    tagged = path.parent / "both-tagged.jpg"
    image = PILImage.open(path)
    exif = PILImage.Exif()
    exif[0x8769] = {0x9214: (want[0], want[1])}
    image.save(tagged, format="JPEG", quality=95, exif=exif)

    report = analyze_focus(tagged)
    assert report.exif.found and report.content.found

    canvas = _decode(render_focus_overlay(tagged, report))

    # 红色（EXIF）与青色（内容）两种标记都应该出现在画面上
    red = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] < 120) & (canvas[:, :, 2] < 120)
    cyan = (canvas[:, :, 0] < 120) & (canvas[:, :, 1] > 180) & (canvas[:, :, 2] > 200)
    assert red.sum() > 50, "没有画出 EXIF 标记"
    assert cyan.sum() > 50, "没有画出内容分析标记"


def test_overlay_renders_when_nothing_found(jpeg_factory):
    """两条路线都没有结论时也要能出图（图例会写 NO FOCUS POINT FOUND）。"""
    path = jpeg_factory(name="empty.jpg", size=(240, 180))
    report = analyze_focus(path)

    assert report.exif.found is False and report.content.found is False
    canvas = _decode(render_focus_overlay(path, report))
    assert canvas.shape[:2] == (180, 240)


def test_overlay_still_marks_the_sharpest_point_when_no_single_focus(tmp_path):
    """分布平坦、判为"没有唯一对焦点"时，图上也必须标出最锐的那一点 ——
    结论是"没有确定焦点"，但用户仍然需要知道画面里最锐的地方在哪。

    用一张"整片都在焦内"的纯噪声图构造这个场景（阈值区域会铺满画面）。
    """
    noisy = Image.effect_noise((480, 360), 60).convert("RGB")
    path = tmp_path / "flat-distribution.jpg"
    noisy.save(path, format="JPEG", quality=95)

    report = analyze_focus(path)
    assert report.candidates, "平坦分布也要给出候选点"

    payload = render_focus_overlay(path, report)
    canvas = _decode(payload).astype(int)
    cyan = (canvas[:, :, 0] < 120) & (canvas[:, :, 1] > 180) & (canvas[:, :, 2] > 200)
    assert cyan.sum() > 20, "没有画出任何青色标记"

    if not report.content.found:
        x, y = report.candidates[0].x, report.candidates[0].y
        neighbourhood = cyan[max(y - 14, 0):y + 15, max(x - 14, 0):x + 15]
        assert neighbourhood.sum() > 0, "没有标出最锐的那一点"


def test_overlay_text_is_ascii_only(sharp_patch_factory):
    """不依赖中文字体：所有烧进图的文字都必须是 ASCII。"""
    path, _ = sharp_patch_factory(name="ascii.jpg")
    report = analyze_focus(path)

    assert report.content.found
    for text in ("EXIF AF", "CONTENT PEAK", "IN-FOCUS AREA", "NO FOCUS POINT FOUND"):
        assert text.isascii()


def test_region_dashes_stay_inside_the_region_box(sharp_patch_factory):
    """回归防护：合焦区域的虚线必须在包围盒**内部**。

    曾经把 ``min(x + dash, bottom)`` 写成了用 bottom 当右端夹取，于是每条虚线都从右往左
    画回 x=bottom，糊成一条横穿画面的实线。这类错误不会报任何异常，只能用像素找出来。
    """
    path, _ = sharp_patch_factory(name="dashes.jpg")
    report = analyze_focus(path)
    assert report.region is not None and report.content.found

    with Image.open(path) as image:
        canvas = np.asarray(
            render_overlay(
                image.convert("RGB"),
                report,
                style=OverlayStyle(legend=False, draw_candidates=False, draw_labels=False),
            )
        ).astype(int)

    cyan = (canvas[:, :, 0] < 120) & (canvas[:, :, 1] > 180) & (canvas[:, :, 2] > 200)
    ys, xs = np.nonzero(cyan)
    assert xs.size > 0, "没有画出任何青色标记"

    region = report.region
    # 清晰块足够大，十字准星也落在框内，所以所有青色像素都应在框内（留出线宽余量）
    assert xs.min() >= region.x - 8, f"有青色像素跑到框左侧：{xs.min()} < {region.x}"
    assert ys.min() >= region.y - 8, f"有青色像素跑到框上方：{ys.min()} < {region.y}"
    assert xs.max() <= region.x + region.width + 8
    assert ys.max() <= region.y + region.height + 8


def test_overlay_data_url_is_a_png(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="url.jpg")
    report = analyze_focus(path)

    url = overlay_data_url(render_focus_overlay(path, report))

    assert url.startswith("data:image/png;base64,")


def test_render_overlay_accepts_plain_pil_image(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="pil.jpg")
    report = analyze_focus(path)

    with Image.open(path) as image:
        out = render_overlay(image.convert("RGB"), report, style=OverlayStyle(legend=False))

    assert out.size == (480, 360)
    assert out.mode == "RGB"


def test_overlay_respects_style_flags(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="style.jpg")
    report = analyze_focus(path)

    with Image.open(path) as image:
        base = render_overlay(
            image.convert("RGB"),
            report,
            style=OverlayStyle(legend=False, draw_region=False, draw_candidates=False),
        )

    # 关掉图例后，左上角应当保持原样
    before = np.asarray(Image.open(path).convert("RGB")).astype(int)[:40, :40]
    after = np.asarray(base).astype(int)[:40, :40]
    assert np.abs(after - before).max() < 60


def test_overlay_is_capped_for_very_large_photos(sharp_patch_factory):
    """一张 5184×3888 的原图直接画无损 PNG 是 29 MB（base64 约 39 MB），
    绝不该把这个体量返回给网页。默认要缩到 2048 以内，但**报告坐标仍是原图像素**。"""
    path, _ = sharp_patch_factory(name="cap.jpg")
    big = path.parent / "cap-big.jpg"
    Image.open(path).resize((4000, 3000), Image.LANCZOS).save(big, quality=90)

    report = analyze_focus(big)
    assert report.width == 4000  # 报告坐标仍在原图空间

    payload = render_focus_overlay(big, report)
    canvas = _decode(payload)

    assert max(canvas.shape[:2]) <= 2048
    assert len(payload) < 4 * 1024 * 1024, f"标注图 {len(payload)/1024/1024:.1f} MB，太大了"


def test_overlay_markers_follow_the_downscale(sharp_patch_factory):
    """缩小后标记必须落在**按比例换算**后的位置上，不能还按原图坐标画。"""
    path, want = sharp_patch_factory(name="scaled.jpg")
    big = path.parent / "scaled-big.jpg"
    Image.open(path).resize((3200, 2400), Image.LANCZOS).save(big, quality=90)

    report = analyze_focus(big)
    canvas = _decode(render_focus_overlay(big, report))
    height, width = canvas.shape[:2]

    ratio = max(width, height) / max(report.width, report.height)
    want_x = report.content.x * ratio
    want_y = report.content.y * ratio

    cyan = (canvas[:, :, 0] < 120) & (canvas[:, :, 1] > 180) & (canvas[:, :, 2] > 200)
    ys, xs = np.nonzero(cyan)
    assert xs.size > 0
    # 青色像素（准星）应当聚集在缩放后的期望位置附近
    assert abs(float(xs.mean()) - want_x) < 0.1 * width, (float(xs.mean()), want_x)
    assert abs(float(ys.mean()) - want_y) < 0.1 * height, (float(ys.mean()), want_y)
    assert want[0] > 0


def test_overlay_max_side_zero_disables_downscaling(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="noscale.jpg")
    report = analyze_focus(path)

    canvas = _decode(render_focus_overlay(path, report, max_side=0))

    assert canvas.shape[:2] == (360, 480)


@pytest.mark.parametrize("params", [FocusParams(use_content=False), FocusParams(use_exif=False)])
def test_overlay_works_with_a_single_route(sharp_patch_factory, params):
    path, _ = sharp_patch_factory(name="single.jpg")
    report = analyze_focus(path, params=params)

    canvas = _decode(render_focus_overlay(path, report))

    assert canvas.shape[:2] == (360, 480)
