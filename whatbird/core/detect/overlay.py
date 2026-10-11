"""把检测到的鸟框画回原图，生成一张可直接给人看的标注图。

与 ``core/focus/overlay.py`` 同一套路数：只依赖 Pillow（项目原有依赖），不引入绘图库。

- **每个框都先描一圈黑边再画彩色框**：照片底色不可控（天空、逆光、树枝……），
  单色细线在浅色或杂乱背景上会直接看不见。
- **标注文字只用 ASCII**（``BIRD 0.87``）：不能假设用户机器上装了某个中文字体。
  中文说明放在 CLI 输出与网页文本里，不烧进图片。
- **不缩放**：标注图按原分辨率输出，方便直接当验收依据。
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .models import Detection, DetectionReport

#: 鸟框颜色（橙红，在天空、绿叶、水面这些常见底色上都比较跳）
COLOR_BOX = (255, 122, 24, 255)
#: 黑描边：保证在任何底色上都能看见
COLOR_EDGE = (0, 0, 0, 220)
COLOR_LABEL = (255, 255, 255, 255)
COLOR_PANEL = (0, 0, 0, 150)

CLOSE_LABEL_GAP = 6


@dataclass(slots=True)
class DetectionStyle:
    """标注样式。默认值在缩略图与原图尺寸下都看得清。"""

    line: int = 3
    """彩色框线宽（像素）。"""

    edge: int = 2
    """黑描边宽度；实际画的是 ``line + edge * 2`` 宽的黑线，彩色线叠在中间。"""

    font_size: int = 16
    label: bool = True
    """是否在框上方写 ``BIRD 0.87``。"""

    legend: bool = True
    """是否在左上角写检测数量与设备。"""


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - 只有很老的 Pillow 才走这里
        return ImageFont.load_default()


def _clamp(value: int, limit: int) -> int:
    return int(min(max(int(value), 0), max(limit - 1, 0)))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    style: DetectionStyle,
    *,
    width: int,
    height: int,
) -> None:
    """在 ``xy`` 处画一块带底色的标签，并保证不跑出画面。"""
    font = _font(style.font_size)
    text_w, text_h = _text_size(draw, text, font)

    x = min(max(xy[0], 0), max(width - text_w - 10, 0))
    y = min(max(xy[1], 0), max(height - text_h - 10, 0))
    draw.rectangle([x - 4, y - 3, x + text_w + 4, y + text_h + 4], fill=COLOR_PANEL)
    draw.text((x, y), text, font=font, fill=color)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    style: DetectionStyle,
) -> None:
    """黑描边 + 彩色框；宽度至少 1 像素，避免退化成看不见的一条线。"""
    left, top, right, bottom = box
    rect = [
        left,
        top,
        max(right, left + style.line),
        max(bottom, top + style.line),
    ]

    draw.rectangle(rect, outline=COLOR_EDGE, width=style.line + style.edge * 2)
    draw.rectangle(rect, outline=COLOR_BOX, width=style.line)


def render_detection_overlay(
    image: Image.Image,
    report: DetectionReport,
    *,
    style: DetectionStyle | None = None,
) -> Image.Image:
    """在 ``image`` 的副本上画出所有鸟框，返回新的 RGB 图像。

    ``report`` 里的框坐标就是**这张图**的像素坐标（:func:`detect_birds` 与
    :func:`render_bird_overlay` 都走 ``ImageOps.exif_transpose``，口径一致）；
    这里仍然会做一次钳制，避免用户手工构造的报告把框画到画面外。
    """
    style = style or DetectionStyle()
    base = image.convert("RGBA")
    width, height = base.size

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font = _font(style.font_size)
    for index, bird in enumerate(report.birds, start=1):
        left = _clamp(bird.x, width)
        top = _clamp(bird.y, height)
        right = _clamp(bird.x + bird.width, width)
        bottom = _clamp(bird.y + bird.height, height)
        box = (left, top, max(right, left + 1), max(bottom, top + 1))
        _draw_box(draw, box, style)

        if style.label:
            text = f"{bird.class_name.upper() or 'BIRD'} {bird.confidence:.2f}"
            if report.count > 1:
                text = f"#{index} {text}"
            text_w, text_h = _text_size(draw, text, font)
            # 默认贴在框上沿外侧；顶到画面上沿时改贴到框内侧
            y = box[1] - text_h - CLOSE_LABEL_GAP - 4
            if y < 0:
                y = box[1] + CLOSE_LABEL_GAP
            _draw_label(draw, (box[0], y), text, COLOR_LABEL, style, width=width, height=height)

    if style.legend:
        title = f"{report.count} BIRD(S) DETECTED" if report.birds else "NO BIRD DETECTED"
        lines = [title]
        if report.device:
            lines.append(f"DEVICE {report.device.upper()}  |  {report.model_name or '-'}")

        pad = 8
        line_h = _text_size(draw, "Ag", font)[1] + 6
        panel_w = max(_text_size(draw, text, font)[0] for text in lines) + pad * 3
        panel_h = len(lines) * line_h + pad * 2
        draw.rectangle([pad, pad, pad + panel_w, pad + panel_h], fill=COLOR_PANEL)
        for index, text in enumerate(lines):
            draw.text((pad * 2, pad * 2 + index * line_h - 3), text, font=font, fill=COLOR_LABEL)

    return Image.alpha_composite(base, layer).convert("RGB")


def render_bird_overlay(
    source: Any,
    report: DetectionReport,
    *,
    style: DetectionStyle | None = None,
) -> bytes:
    """读图 → 转正 → 画鸟框 → 编码成 PNG 字节（原分辨率，不缩放）。

    刻意不复用 ``image_input.prepare_image``：那条路会把图片缩放并重编码成 JPEG
    （它是给模型看的），而标注图需要保真。转正与编码都复用
    :mod:`whatbird.core.image_input` 的公共实现，保证与对焦点标注图完全一致。
    """
    from ..image_input import encode_png, load_upright_rgb

    _, image = load_upright_rgb(source)
    annotated = render_detection_overlay(image, report, style=style)
    return encode_png(annotated)


def overlay_data_url(payload: bytes, *, mime: str = "image/png") -> str:
    """把标注图字节编码成 data URL，供网页直接 ``<img src>`` 使用。

    独立实现（没有复用 focus 的同名函数）是刻意的：检测模块不该因为一个 base64
    编码就依赖对焦模块，将来任何一个功能被移除都不会牵连另一个。
    """
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


__all__ = [
    "COLOR_BOX",
    "COLOR_EDGE",
    "COLOR_LABEL",
    "COLOR_PANEL",
    "DetectionStyle",
    "overlay_data_url",
    "render_bird_overlay",
    "render_detection_overlay",
]