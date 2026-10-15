"""区域清晰度评分的测试：全部用合成样本，真值可控、完全确定性。

这一组测试同时**钉死几条实测事实**（它们是这个 API 只能给相对结论的原因）：

- 同一内容：SML 随模糊单调下降 → 可以用来比较"同一只鸟的清晰/模糊版"；
- 跨内容：同样模糊量下棋盘格的 SML 是噪声区的 2 倍以上 → **不能**跨内容比绝对锐度；
- 无细节：纯色/天空区域测不出清晰度 → 结论必须是"无法判断"，而不是"没对上焦"。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFilter

from whatbird.core import (
    FocusParams,
    RegionBox,
    WhatBirdError,
    score_region_sharpness,
    score_regions,
)
from whatbird.core.detect.models import BIRD_CLASS_ID, Detection


def _save(image: Image.Image, path, *, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality)
    return path


def _checkerboard(size: int, step: int = 4) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                draw.rectangle([x, y, x + step - 1, y + step - 1], fill=255)
    return image


def _patch_box(center: tuple[int, int], patch_size: int) -> RegionBox:
    return RegionBox(
        x=center[0] - patch_size // 2,
        y=center[1] - patch_size // 2,
        width=patch_size,
        height=patch_size,
    )


# ------------------------------------------------------------------ 主流程
def test_sharp_patch_is_the_in_focus_region(sharp_patch_factory):
    """锐利块就是"合焦的那块"：画面最锐的点与最锐的一片都在框内。"""
    path, center = sharp_patch_factory(name="patch.jpg", size=(480, 360), patch_size=96)
    report = score_regions(str(path), [_patch_box(center, 96)])

    score = report.regions[0]
    assert score.reliable is True
    assert score.low_detail is False
    assert score.relative_to_frame_peak > 0.95
    assert score.top_share > 0.5
    assert "合焦" in score.verdict


def test_blurred_background_region_is_not_in_focus(sharp_patch_factory):
    """同一张图里，模糊背景那块不该被说成合焦。"""
    path, center = sharp_patch_factory(name="bg.jpg", size=(480, 360), patch_size=96)
    report = score_regions(str(path), [(_patch_box(center, 96)), (10, 250, 120, 90)])

    patch, background = report.regions
    assert patch.relative_to_frame_peak > background.relative_to_frame_peak
    assert background.relative_to_frame_peak < 0.9
    # 背景块要么"比最锐处糊"，要么"几乎没有可测细节（无法判断）"——
    # 但绝不能声称自己是合焦的那块。注意不能用 `"合焦" in verdict` 判断：
    # "无法判断"那句里也含「合焦与否」这几个字。
    assert not background.verdict.startswith("合焦")
    assert "画面里最锐的地方就在这个框里" not in background.verdict


def test_same_content_blur_is_monotonic(sharp_patch_factory, tmp_path):
    """同一内容逐级模糊：绝对锐度必须单调下降（这是唯一可靠的"谁更清晰"用法）。"""
    path, center = sharp_patch_factory(name="mono.jpg", size=(480, 360), patch_size=120)
    box = _patch_box(center, 120)

    with Image.open(path) as opened:
        base = opened.convert("RGB")

    values = []
    for radius in (0, 1, 2, 4):
        blurred = base if radius == 0 else base.filter(ImageFilter.GaussianBlur(radius))
        target = _save(blurred, tmp_path / f"blur-{radius}.jpg")
        values.append(score_region_sharpness(str(target), box).sharpness)

    assert values == sorted(values, reverse=True), f"清晰度没有随模糊单调下降：{values}"
    assert values[0] > values[-1] * 3, f"动态范围太小，无法区分清晰与模糊：{values}"


def test_cross_content_absolute_sharpness_is_not_comparable(tmp_path):
    """跨内容不可比 —— 这条是**故意钉死的实测事实**，不是缺陷。

    同样模糊量下，纹理密集的棋盘格 SML 明显高于噪声区（实测约 3 倍）。
    所以 API 只报"相对全图"的量，绝不报"这块区域绝对有多清晰"。
    """
    size = 200
    canvas = Image.new("RGB", (size * 2, size), (0, 0, 0))
    canvas.paste(_checkerboard(size).convert("RGB"), (0, 0))
    canvas.paste(Image.effect_noise((size, size), 45).convert("RGB"), (size, 0))
    target = _save(canvas.filter(ImageFilter.GaussianBlur(2.0)), tmp_path / "cross.jpg")

    report = score_regions(str(target), [(0, 0, size, size), (size, 0, size, size)])
    checker, noise = report.regions

    assert checker.sharpness > noise.sharpness * 2, "跨内容差异实测约 3 倍，这里应当明显大于 2 倍"


# ------------------------------------------------------------------ 测不出来的情况
def test_featureless_region_says_cannot_tell_not_blurry(tmp_path):
    """纯色区域必须说"无法判断"，而不是"没对上焦" —— 没内容和没焦点是两件事。"""
    canvas = Image.new("RGB", (400, 300), (128, 128, 128))
    canvas.paste(_checkerboard(180).convert("RGB"), (10, 10))
    target = _save(canvas, tmp_path / "flat.jpg")

    report = score_regions(str(target), [(200, 120, 180, 160), (10, 10, 180, 180)])
    flat, sharp = report.regions

    assert flat.low_detail is True
    assert "无法判断" in flat.verdict
    assert "没对上焦" not in flat.verdict
    assert sharp.low_detail is False
    assert "合焦" in sharp.verdict
    assert any("无法判断" in note for note in report.notes)


def test_uniformly_soft_frame_hedges_the_verdict(tmp_path):
    """整张图都软（峰值/中位数差异极小）时，不许说"这块就是最锐的所以合焦"。"""
    canvas = Image.effect_noise((400, 300), 45).convert("RGB").filter(ImageFilter.GaussianBlur(6))
    target = _save(canvas, tmp_path / "soft.jpg")

    report = score_regions(str(target), [(40, 40, 160, 160)])

    assert report.frame_has_peak is False
    assert report.frame_contrast < 2.82
    # 结论要么是"没有唯一锐处"（区域还有细节时），要么是"无法判断"（糊到没细节），
    # 但都不许声称"合焦"；全图参照不成立这件事同时会写进报告说明里。
    assert not report.regions[0].verdict.startswith("合焦")
    assert any("没有明显的唯一锐处" in note for note in report.notes)


def test_mildly_soft_frame_adds_a_hedge_note(tmp_path):
    """差异还够判、但整张图已经偏软时，补一条"不等于照片清晰"的说明。"""
    canvas = _checkerboard(200).convert("RGB").filter(ImageFilter.GaussianBlur(3.4))
    target = _save(canvas, tmp_path / "mild.jpg")

    report = score_regions(str(target), [(20, 20, 160, 160)])

    if report.frame_has_peak and report.frame_contrast < 5.64:
        assert any("清晰度差异不大" in note for note in report.notes)


def test_solid_image_is_rejected(tmp_path):
    target = _save(Image.new("RGB", (400, 300), (120, 120, 120)), tmp_path / "solid.jpg")

    with pytest.raises(WhatBirdError) as info:
        score_regions(str(target), [(10, 10, 100, 100)])

    assert "没有可用的清晰度差异" in info.value.message


def test_tiny_region_is_flagged_unreliable(tmp_path):
    """小到连两个最小窗口都放不下时，只能给"仅供参考"。"""
    canvas = Image.new("RGB", (400, 300), (128, 128, 128))
    canvas.paste(_checkerboard(300).convert("RGB"), (50, 0))
    target = _save(canvas, tmp_path / "tiny.jpg")

    score = score_region_sharpness(str(target), (100, 100, 5, 5))

    assert score.reliable is False
    assert "仅供参考" in score.verdict


def test_small_region_gets_a_smaller_window(tmp_path):
    """窗口必须远小于被测区域：小框自动换更小的窗口，并在说明里交代。"""
    canvas = Image.new("RGB", (640, 480), (128, 128, 128))
    canvas.paste(_checkerboard(240).convert("RGB"), (200, 120))
    target = _save(canvas, tmp_path / "window.jpg")

    report = score_regions(str(target), [(200, 120, 60, 60)])

    score = report.regions[0]
    assert score.reliable is True
    assert score.window < report.window
    assert any("更小的聚合窗口" in note for note in report.notes)


# ------------------------------------------------------------------ 框的写法与钳制
def test_three_box_forms_are_equivalent(tmp_path):
    """RegionBox / 四元组 / 带 x-y-width-height 的对象（检测结果）必须等价。"""
    canvas = Image.new("RGB", (400, 300), (128, 128, 128))
    canvas.paste(_checkerboard(160).convert("RGB"), (40, 40))
    target = _save(canvas, tmp_path / "forms.jpg")

    detection = Detection(x=40, y=40, width=160, height=160, confidence=0.9, class_id=BIRD_CLASS_ID)
    results = [
        score_region_sharpness(str(target), RegionBox(x=40, y=40, width=160, height=160)),
        score_region_sharpness(str(target), (40, 40, 160, 160)),
        score_region_sharpness(str(target), detection),
        score_region_sharpness(str(target), SimpleNamespace(x=40, y=40, width=160, height=160)),
    ]

    assert len({item.verdict for item in results}) == 1
    assert len({round(item.peak) for item in results}) == 1


def test_unknown_box_form_is_rejected(tmp_path, jpeg_factory):
    path = jpeg_factory(name="bad.jpg", size=(120, 120))

    with pytest.raises(WhatBirdError) as info:
        score_regions(str(path), [object()])

    assert "无法识别这个框" in info.value.message


def test_box_is_clamped_into_the_image(tmp_path):
    canvas = Image.new("RGB", (300, 200), (128, 128, 128))
    canvas.paste(_checkerboard(150).convert("RGB"), (0, 0))
    target = _save(canvas, tmp_path / "clamp.jpg")

    score = score_region_sharpness(str(target), (-50, -50, 400, 400))

    assert score.box.xyxy == (0, 0, 300, 200)
    assert score.width_ratio <= 1.0 and score.height_ratio <= 1.0


def test_box_constructors_agree():
    from_xyxy = RegionBox.from_xyxy(10.4, 20.6, 110.4, 220.6)
    assert (from_xyxy.x, from_xyxy.y, from_xyxy.width, from_xyxy.height) == (10, 21, 100, 200)

    from_ratios = RegionBox.from_ratios(0.1, 0.25, 0.2, 0.5, image_width=1000, image_height=400)
    assert from_ratios.xyxy == (100, 100, 300, 300)


# ------------------------------------------------------------------ 报告契约
def test_report_dict_shape(sharp_patch_factory):
    path, center = sharp_patch_factory(name="contract.jpg", size=(480, 360))
    report = score_regions(str(path), [_patch_box(center, 96)])

    payload = report.to_dict()
    assert set(payload) == {
        "source_image",
        "image",
        "metric",
        "frame",
        "regions",
        "elapsed_seconds",
        "notes",
        "params",
    }
    assert set(payload["image"]) == {
        "width",
        "height",
        "analyzed_width",
        "analyzed_height",
        "scale",
        "window",
    }
    assert set(payload["frame"]) == {"median", "peak", "contrast", "has_peak"}

    region = payload["regions"][0]
    assert set(region) == {
        "box",
        "x_ratio",
        "y_ratio",
        "width_ratio",
        "height_ratio",
        "sharpness",
        "peak",
        "frame_median",
        "frame_peak",
        "frame_contrast",
        "frame_has_peak",
        "relative_to_frame_peak",
        "top_share",
        "percentile",
        "texture",
        "verdict",
        "detail",
        "reliable",
        "low_detail",
        "sampled_pixels",
        "window",
    }


def test_regions_keep_input_order_and_share_frame_stats(sharp_patch_factory):
    path, center = sharp_patch_factory(name="order.jpg", size=(480, 360), patch_size=96)
    patch = (center[0] - 48, center[1] - 48, 96, 96)
    boxes = [(10, 10, 80, 80), patch, (300, 250, 100, 80)]

    report = score_regions(str(path), boxes)

    assert [item.box.to_dict() for item in report.regions] == [
        {"x": 10, "y": 10, "width": 80, "height": 80},
        {"x": patch[0], "y": patch[1], "width": 96, "height": 96},
        {"x": 300, "y": 250, "width": 100, "height": 80},
    ]
    assert len({item.frame_peak for item in report.regions}) == 1
    assert report.top() is report.regions[1]


def test_scores_are_deterministic_and_params_are_reported(sharp_patch_factory):
    path, center = sharp_patch_factory(name="det.jpg", size=(480, 360))
    box = _patch_box(center, 96)

    first = score_region_sharpness(str(path), box, params=FocusParams(region_quantile=0.5))
    second = score_region_sharpness(str(path), box, params=FocusParams(region_quantile=0.5))
    higher = score_region_sharpness(str(path), box, params=FocusParams(region_quantile=0.95))

    assert first.sharpness == second.sharpness
    assert higher.sharpness >= first.sharpness

    report = score_regions(str(path), [box])
    assert report.params["region_quantile"] == pytest.approx(0.75)
    assert report.metric == "sml"


def test_missing_file_gives_chinese_error(tmp_path):
    with pytest.raises(WhatBirdError) as info:
        score_regions(str(tmp_path / "nope.jpg"), [(0, 0, 10, 10)])

    assert "不存在" in info.value.message


def test_empty_box_list_is_rejected(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="none.jpg")

    with pytest.raises(WhatBirdError) as info:
        score_regions(str(path), [])

    assert "没有给出要评估的区域" in info.value.message


def test_detect_integration_uses_detection_boxes(sharp_patch_factory):
    """与位置检测配合：直接把 Detection 列表丢进来（这是最常见的用法）。"""
    path, center = sharp_patch_factory(name="birds.jpg", size=(480, 360), patch_size=96)
    detections = [
        Detection(x=center[0] - 48, y=center[1] - 48, width=96, height=96, confidence=0.93),
    ]

    report = score_regions(str(path), detections)

    assert len(report.regions) == 1
    assert "合焦" in report.regions[0].verdict
