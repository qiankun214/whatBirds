"""对焦分析：给一张 jpg，找出对焦点在哪。

两条互相独立的路线，**都跑、都上报**：

- :mod:`~whatbird.core.focus.exif_focus` —— 读相机写进元数据的对焦点
  （相机*声称*它对在哪儿）；
- :mod:`~whatbird.core.focus.content` —— 用传统图形学算法分析像素的清晰度分布
  （画面*实际*最锐利处在哪）。

两者结论一致时基本可以放心；不一致时，元数据那一路可能是区域/手动对焦，
而内容那一路可能是画面里有比焦点更锐的东西（噪点、高对比边缘）——都值得看一眼。

用法::

    from whatbird.core import analyze_focus, render_focus_overlay

    report = analyze_focus(r"D:\\photos\\bird.jpg")
    print(report.exif.x, report.exif.y)        # 相机声称的对焦点
    print(report.content.x, report.content.y)  # 画面实际最清晰处
    png = render_focus_overlay(r"D:\\photos\\bird.jpg", report)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

from PIL import ImageOps

from ..errors import ImageInputError, WhatBirdError
from ..image_input import ImageSource, open_stored_image, read_image_bytes, to_rgb
from .content import ContentAnalysis, analyze_content
from .exif_focus import read_af_point, to_hit, unavailable_reason
from .models import (
    METHOD_CONTENT,
    METHOD_EXIF,
    FocusHit,
    FocusParams,
    FocusReport,
)
from .overlay import OverlayStyle, overlay_data_url, render_focus_overlay, render_overlay

TAG_ORIENTATION = 0x0112

#: 上层可以传进来的图片来源（本地路径 / 内存字节 / 已打包的 ImageSource）
Source = Union[ImageSource, str, Path, bytes, bytearray, tuple]


def analyze_focus(
    source: Source,
    *,
    params: FocusParams | None = None,
) -> FocusReport:
    """分析一张图片的对焦点。

    :param source: 本地路径、内存字节或 :class:`~whatbird.core.image_input.ImageSource`。
    :param params: 内容分析参数；``None`` 用默认值。
    """
    params = params or FocusParams()
    started = time.perf_counter()

    name, raw = read_image_bytes(source)

    with open_stored_image(name, raw) as opened:
        try:
            exif = opened.getexif()
            orientation = int(exif.get(TAG_ORIENTATION) or 1)
        except Exception:  # pragma: no cover - 畸形 EXIF
            orientation = 1
        stored_width, stored_height = opened.size

        reading = None
        reason = ""
        if params.use_exif:
            try:
                reading = read_af_point(opened, raw)
            except WhatBirdError:
                raise
            except Exception:  # pragma: no cover - EXIF 千奇百怪，绝不允许影响内容分析
                reading = None
            if reading is None:
                try:
                    reason = unavailable_reason(opened, raw)
                except Exception:  # pragma: no cover
                    reason = "该图片没有可用的 EXIF 对焦信息"

        upright = ImageOps.exif_transpose(opened)
        rgb_image = to_rgb(upright)

    width, height = rgb_image.size

    if params.use_exif:
        exif_hit = to_hit(
            reading,
            reason,
            width=width,
            height=height,
            orientation=orientation,
            stored_width=stored_width,
            stored_height=stored_height,
        )
    else:
        exif_hit = FocusHit(method=METHOD_EXIF, found=False, detail="已按参数关闭 EXIF 路线")

    content: ContentAnalysis | None = None
    if params.use_content:
        content = analyze_content(rgb_image, params=params)
        content_hit = content.hit
    else:
        content_hit = FocusHit(method=METHOD_CONTENT, found=False, detail="已按参数关闭内容分析")

    notes: list[str] = []
    if not exif_hit.found and params.use_exif:
        notes.append(exif_hit.detail)
    if not content_hit.found and params.use_content:
        notes.append(content_hit.detail)

    report = FocusReport(
        source_image=name,
        width=width,
        height=height,
        analyzed_width=content.analyzed_width if content else 0,
        analyzed_height=content.analyzed_height if content else 0,
        scale=content.scale if content else 1.0,
        elapsed_seconds=time.perf_counter() - started,
        exif=exif_hit,
        content=content_hit,
        region=content.region if content else None,
        candidates=content.candidates if content else [],
        agree=_agree(exif_hit, content_hit, content, width, height),
        notes=notes,
        params=params.to_dict(),
    )
    return report


def _agree(
    exif_hit: FocusHit,
    content_hit: FocusHit,
    content: ContentAnalysis | None,
    width: int,
    height: int,
) -> bool:
    """两条路线是否互相印证。

    有合焦区域时，看 EXIF 点是否落在区域里（带半个区域的容差 —— 相机的 AF 框
    本身就是一片区域，不是几何意义上的一个点）；没有区域时退化成"距离够近"。
    """
    if not (exif_hit.found and content_hit.found):
        return False
    region = content.region if content else None
    if region is not None:
        tolerance = max(region.width, region.height) // 2
        return region.contains(exif_hit.x, exif_hit.y, tolerance=tolerance)
    span = max(width, height) * 0.05
    return abs(exif_hit.x - content_hit.x) <= span and abs(exif_hit.y - content_hit.y) <= span


__all__ = [
    "ContentAnalysis",
    "FocusHit",
    "FocusParams",
    "FocusReport",
    "ImageInputError",
    "OverlayStyle",
    "Source",
    "analyze_content",
    "analyze_focus",
    "overlay_data_url",
    "read_af_point",
    "render_focus_overlay",
    "render_overlay",
]
