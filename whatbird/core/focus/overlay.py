"""把对焦分析的结论画回原图，生成一张可直接给人看的标注图。

设计上只依赖 Pillow（项目原有依赖），不引入绘图库：

- **两条路线用两种颜色**：EXIF（相机声称的）用红色，内容分析（画面实际的）用青色。
  两者重合时肉眼一眼就能看出来，这正是这张图的价值。
- **每条线都先描一圈黑边再画彩色线**：照片底色不可控（天空、逆光、树叶……），
  单色细线在浅色或杂乱背景上会直接看不见。
- **标注文字只用 ASCII**：不能假设用户机器上装了某个中文字体。中文说明放在
  CLI 输出和网页文本里，不烧进图片。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .models import FocusReport

#: EXIF 路线（相机声称的对焦点）
COLOR_EXIF = (255, 59, 48, 255)
#: 内容分析路线（画面实际最清晰处）
COLOR_CONTENT = (0, 229, 255, 255)
#: 候选点
COLOR_CANDIDATE = (0, 229, 255, 170)
COLOR_EDGE = (0, 0, 0, 220)
COLOR_TEXT = (255, 255, 255, 255)
COLOR_PANEL = (0, 0, 0, 150)


@dataclass(slots=True)
class OverlayStyle:
    """标注样式。默认值在缩略图与原图尺寸下都看得清。"""

    arm: int = 34
    """十字准星的臂长（像素）。"""

    box: int = 30
    """EXIF 路线方框的半边长。"""

    radius: int = 16
    """内容分析路线圆环的半径。"""

    line: int = 2
    edge: int = 3
    font_size: int = 16
    legend: bool = True
    draw_labels: bool = True
    draw_region: bool = True
    draw_candidates: bool = True


#: 标注图最长边的默认上限。
#:
#: 分析出来的坐标始终是**原图**像素，但标注图只是给人看的：一张 5184×3888 的
#: 原图直接画成无损 PNG 有 29 MB，编成 base64 约 39 MB，网页接口完全不该返回这种体量。
#: 缩到 2048 之后仍然看得清标记，体积降到 1 MB 级。
DEFAULT_OVERLAY_MAX_SIDE = 2048


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - 只有很老的 Pillow 才走这里
        return ImageFont.load_default()


def _clamp(value: int, limit: int) -> int:
    return int(min(max(int(value), 0), max(limit - 1, 0)))


def _stroke_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int, int],
    style: OverlayStyle,
) -> None:
    """先画黑边再画彩色线，保证在任何底色上都看得见。"""
    draw.line(list(points), fill=COLOR_EDGE, width=style.line + style.edge * 2, joint="curve")
    draw.line(list(points), fill=color, width=style.line, joint="curve")


def _dashed_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    style: OverlayStyle,
    *,
    dash: int = 14,
) -> None:
    """Pillow 没有原生虚线，用短线段拼一个。"""
    left, top, right, bottom = box
    for x in range(left, right, dash * 2):
        _stroke_line(draw, [(x, top), (min(x + dash, right), top)], color, style)
        _stroke_line(draw, [(x, bottom), (min(x + dash, right), bottom)], color, style)
    for y in range(top, bottom, dash * 2):
        _stroke_line(draw, [(left, y), (left, min(y + dash, bottom))], color, style)
        _stroke_line(draw, [(right, y), (right, min(y + dash, bottom))], color, style)


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    style: OverlayStyle,
    *,
    width: int,
    height: int,
) -> None:
    font = _font(style.font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    x = min(max(xy[0], 0), max(width - text_w - 8, 0))
    y = min(max(xy[1], 0), max(height - text_h - 8, 0))
    draw.rectangle([x - 4, y - 3, x + text_w + 4, y + text_h + 4], fill=COLOR_PANEL)
    draw.text((x, y), text, font=font, fill=color)


def render_overlay(
    image: Image.Image,
    report: FocusReport,
    *,
    style: OverlayStyle | None = None,
    coordinate_scale: float = 1.0,
) -> Image.Image:
    """在 ``image`` 的副本上画出对焦结论，返回新的 RGB 图像。

    :param coordinate_scale: 把报告里的**原图坐标**映射到这张图上的比例。
        ``render_focus_overlay`` 在把大图缩小之后会传入小于 1 的值。
    """
    style = style or OverlayStyle()
    base = image.convert("RGBA")
    width, height = base.size
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # 坐标与标记尺寸都按同一比例缩放；标记尺寸设下限，缩小后仍要看得见
    ratio = max(float(coordinate_scale), 1e-6)
    marker = max(min(ratio, 1.0), 0.35)
    arm = max(int(round(style.arm * marker)), 8)
    half = max(int(round(style.box * marker)), 8)
    ring = max(int(round(style.radius * marker)), 5)
    dot = max(int(round(5 * marker)), 4)
    dash = max(int(round(14 * marker)), 5)

    def sx(value: float) -> int:
        return _clamp(round(value * ratio), width)

    def sy(value: float) -> int:
        return _clamp(round(value * ratio), height)

    # ---- 合焦区域：先画，免得盖住对焦点标记 ----
    if style.draw_region and report.region is not None and report.content.found:
        region = report.region
        box = (
            sx(region.x),
            sy(region.y),
            sx(region.x + region.width),
            sy(region.y + region.height),
        )
        if box[2] > box[0] and box[3] > box[1]:
            _dashed_rect(draw, box, COLOR_CONTENT, style, dash=dash)
            if style.draw_labels:
                _label(draw, (box[0], box[1] - style.font_size - 12), "IN-FOCUS AREA",
                       COLOR_CONTENT, style, width=width, height=height)

    # ---- 候选点 ----
    if style.draw_candidates:
        for point in report.candidates[1:]:
            x, y = sx(point.x), sy(point.y)
            draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=COLOR_CANDIDATE)

    # ---- EXIF 路线 ----
    if report.exif.found:
        x, y = sx(report.exif.x), sy(report.exif.y)
        _stroke_line(draw, [(x - arm, y), (x + arm, y)], COLOR_EXIF, style)
        _stroke_line(draw, [(x, y - arm), (x, y + arm)], COLOR_EXIF, style)
        _stroke_line(
            draw,
            [(x - half, y - half), (x + half, y - half), (x + half, y + half),
             (x - half, y + half), (x - half, y - half)],
            COLOR_EXIF,
            style,
        )
        if style.draw_labels:
            _label(draw, (x + arm + 6, y - style.font_size), "EXIF AF", COLOR_EXIF, style,
                   width=width, height=height)

    # ---- 内容分析路线 ----
    if report.content.found:
        x, y = sx(report.content.x), sy(report.content.y)
        _stroke_line(draw, [(x - arm, y), (x + arm, y)], COLOR_CONTENT, style)
        _stroke_line(draw, [(x, y - arm), (x, y + arm)], COLOR_CONTENT, style)
        draw.ellipse(
            [x - ring - style.edge, y - ring - style.edge,
             x + ring + style.edge, y + ring + style.edge],
            outline=COLOR_EDGE,
            width=style.line + style.edge,
        )
        draw.ellipse(
            [x - ring, y - ring, x + ring, y + ring],
            outline=COLOR_CONTENT,
            width=style.line,
        )
        if style.draw_labels:
            _label(draw, (x + arm + 6, y + 4), "CONTENT PEAK", COLOR_CONTENT, style,
                   width=width, height=height)
    elif report.candidates:
        # 没有唯一对焦点时，至少把"最锐的那一点"标出来：结论是"没有确定焦点"，
        # 但用户仍然需要知道画面里最锐的地方在哪。
        x, y = sx(report.candidates[0].x), sy(report.candidates[0].y)
        ring = max(ring, 7)
        draw.ellipse(
            [x - ring, y - ring, x + ring, y + ring],
            outline=COLOR_CONTENT,
            width=style.line,
        )
        if style.draw_labels:
            _label(draw, (x + ring + 8, y - style.font_size), "SHARPEST", COLOR_CONTENT,
                   style, width=width, height=height)

    # ---- 图例 ----
    if style.legend:
        lines = []
        if report.exif.found:
            lines.append(("EXIF AF  (camera metadata)", COLOR_EXIF))
        if report.content.found:
            lines.append(("CONTENT PEAK  (pixel sharpness)", COLOR_CONTENT))
        elif report.candidates:
            lines.append(("SHARPEST  (no single focus point)", COLOR_CONTENT))
        if not lines:
            lines.append(("NO FOCUS POINT FOUND", COLOR_TEXT))
        font = _font(style.font_size)
        pad = 8
        text_w = max(draw.textbbox((0, 0), text, font=font)[2] for text, _ in lines)
        text_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        panel_w = text_w + pad * 3
        panel_h = len(lines) * (text_h + 6) + pad * 2
        draw.rectangle([pad, pad, pad + panel_w, pad + panel_h], fill=COLOR_PANEL)
        for index, (text, color) in enumerate(lines):
            y = pad * 2 + index * (text_h + 6) - 3
            draw.rectangle([pad * 2, y + text_h // 2 - 4, pad * 2 + 12, y + text_h // 2 + 4],
                           fill=color)
            draw.text((pad * 2 + 20, y), text, font=font, fill=COLOR_TEXT)

    return Image.alpha_composite(base, layer).convert("RGB")


def render_focus_overlay(
    source: Any,
    report: FocusReport,
    *,
    style: OverlayStyle | None = None,
    max_side: int = DEFAULT_OVERLAY_MAX_SIDE,
) -> bytes:
    """读图 → 转正 → 画标注 → 编码成 PNG 字节。

    刻意不复用 ``image_input.prepare_image``：那条路会把图片缩放并重编码成 JPEG
    （它是给模型看的），而标注图需要保真。转正（``exif_transpose``）必须做，
    否则竖拍照片上的标记会和 :func:`analyze_focus` 的坐标错位 —— 转正与编码都复用
    :mod:`whatbird.core.image_input` 的公共实现，保证与鸟类位置检测的标注图用同一套坐标。

    :param max_side: 标注图最长边的上限（默认 2048），0 表示不缩放。
        **报告里的坐标始终是原图像素**，这里只是为了别把 29 MB 的 PNG
        编成 39 MB 的 base64 塞进网页接口 —— 一张 5184×3888 的原图直接画无损 PNG 就是这个体量。
    """
    from ..image_input import encode_png, load_upright_rgb

    _, image = load_upright_rgb(source)

    coordinate_scale = 1.0
    if max_side and max(image.size) > max_side:
        coordinate_scale = max_side / float(max(image.size))
        image = image.resize(
            (
                max(1, int(round(image.width * coordinate_scale))),
                max(1, int(round(image.height * coordinate_scale))),
            ),
            Image.LANCZOS,
        )

    annotated = render_overlay(
        image, report, style=style, coordinate_scale=coordinate_scale
    )
    return encode_png(annotated)


def overlay_data_url(payload: bytes, *, mime: str = "image/png") -> str:
    """把标注图字节编码成 data URL，供网页直接 ``<img src>`` 使用。"""
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


__all__ = [
    "COLOR_CONTENT",
    "COLOR_EXIF",
    "OverlayStyle",
    "overlay_data_url",
    "render_focus_overlay",
    "render_overlay",
]
