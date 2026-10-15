"""区域清晰度评分：框出来的那一块，在全图里算不算清晰。

为什么只给**相对**结论（这是本模块最重要的一句话）
--------------------------------------------------
HANDOFF §6.1 的实测结论决定了这个 API 的形状，本次又复验了一遍：

- ✅ **同一内容**比清晰：SML 随模糊单调下降（清晰→糊有约 7 倍动态范围），可以用来比较
  "同一只鸟的清晰版与模糊版"。
- ⚠️ **同一张图里跨内容**比绝对清晰度：**不行**。同样模糊量下，纹理密集的棋盘格 SML
  是噪声区域的 2.5 倍 —— 单看一个数会把"内容多"误读成"更清晰"。
- ❌ **跨照片**比：需要基准标定，本工具刻意不做。

所以这里上报的每个数字都是**同一张图内部**的相对量：

- ``relative_to_frame_peak``：框内锐度 ÷ 全图最锐处。**1.0 = 全图最锐的地方就在框里**，
  这是回答"框住的那只鸟合焦了吗"的主指标；
- ``percentile``：框内锐度落在全图清晰度分布的哪个百分位；
- ``texture``：框内灰度标准差 —— 天空、虚化背景、纯色面这类"没有细节"的区域，
  清晰度值恒接近 0，**无论合焦与否都测不出清晰度**，此时必须说"无法判断"，
  而不是把它当成"很糊"。

算法直接复用 :func:`~whatbird.core.focus.content.build_sharpness_map`：必须与
"全图最锐处在哪"用同一条管线（同样的缩放、去噪、聚合窗口），否则两者比出来的
相对值没有意义。

用法::

    from whatbird.core import score_region_sharpness, score_regions

    score = score_region_sharpness(r"D:\\photos\\bird.jpg", (120, 80, 300, 240))
    print(score.verdict, f"{score.relative_to_frame_peak:.0%}")

    # 与位置检测配合：一次建图，把每只鸟的框都评一遍
    report = score_regions(path, [bird for bird in detection.birds])
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable

from ..errors import WhatBirdError
from ..image_input import load_upright_rgb
from .content import SharpnessMap, build_sharpness_map, image_libs, odd_kernel
from .models import (
    FocusParams,
    RegionBox,
    RegionScoreReport,
    RegionSharpness,
    ratio_of,
)

#: 判定阈值：框内**最锐处** ÷ 全图最锐处
VERDICT_PEAK = 0.95
"""``relative_to_frame_peak`` ≥ 该值：画面最锐的点就在这个框里。"""

VERDICT_TOP_SHARE = 0.20
"""``top_share`` ≥ 该值：画面里最锐的那一片有相当一部分落在这个框内。"""

VERDICT_ABOVE_TYPICAL = 0.75
"""百分位 ≥ 该值：比全图大多数地方锐（但画面最锐处仍在框外 → 景深边缘 / 轻微失焦）。"""

TOP_QUANTILE = 0.99
"""``top_share`` 里"最锐的一片"取的是全图清晰度分布的这个分位以上（默认 1%）。"""

MIN_WINDOW = 3
"""聚合窗口的最小边长（与 :func:`~whatbird.core.focus.content.odd_kernel` 的下限一致）。

框小到连两个这样的窗口都放不下时，评分会被标为"不可靠" —— 那时已经没有任何
"窗口远小于区域"的余地了。
"""

HEDGE_FACTOR = 2.0
"""全图峰值/中位数低于"判据阈值 × 该倍数"时，补一条"整张图清晰度差异不大"的说明。

实测（同一张翠鸟逐级高斯模糊）：半径 0/1/2/4/8 时全图峰值/中位数是
50.76 / 37.08 / 24.42 / 12.76 / 4.91，半径 16 时降到 2.25（低于判据阈值 2.82 →
直接进"没有唯一锐处"分支）。也就是说 4.91 这种"整张图已经明显软了、但差异还够判"
的中间地带只有靠这条说明提醒用户：**区域结论说的是"框里是不是最锐的那块"，
不等于"这张照片很清晰"** —— 绝对锐度需要基准标定，本工具不做。
"""


def _as_box(item: Any) -> RegionBox:
    """把各种形态的框统一成 :class:`RegionBox`。

    刻意做宽，因为调用方来自不同层：

    - :class:`RegionBox`；
    - ``(x, y, width, height)`` 四元组/列表 —— 注意是**宽高**，不是右下角坐标；
    - 带 ``x`` / ``y`` / ``width`` / ``height`` 属性的对象 —— 检测模块的
      :class:`~whatbird.core.detect.models.Detection` 正好是这种，可以直接传进来。
    """
    if isinstance(item, RegionBox):
        return item
    if isinstance(item, (tuple, list)) and len(item) == 4:
        try:
            return RegionBox(
                x=int(item[0]), y=int(item[1]), width=int(item[2]), height=int(item[3])
            )
        except (TypeError, ValueError) as exc:
            raise WhatBirdError(
                "框的四个数必须是整数像素：x,y,width,height",
                hint=f"收到的是 {item!r}",
            ) from exc

    try:
        return RegionBox(
            x=int(getattr(item, "x")),
            y=int(getattr(item, "y")),
            width=int(getattr(item, "width")),
            height=int(getattr(item, "height")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise WhatBirdError(
            "无法识别这个框",
            hint=(
                "支持三种写法：RegionBox、(x, y, width, height) 四元组、"
                "或带 x/y/width/height 属性的对象（例如检测结果的 Detection）"
            ),
        ) from exc


def _frame_peak_threshold(params: FocusParams) -> float:
    """全图"有没有明显唯一锐处"的判据 —— 与对焦分析共用同一条标定。

    ``contrast_ref``（置信度记 1.0 对应的峰值/中位数）与 ``min_confidence`` 一起决定
    "差异小到什么程度就不算有锐处"：默认 10 与 0.45 → 10^0.45 ≈ 2.82，即
    "峰值只比中位数高 2.8 倍"。真实 OM-1 鸟片实测就是 2.16~2.26（整片树叶都在焦内），
    所以这个阈值是拿真实照片校准出来的，不是拍的。
    """
    reference = max(float(params.contrast_ref), 1.0 + 1e-6)
    confidence = min(max(float(params.min_confidence), 0.0), 1.0)
    return float(math.pow(reference, confidence))


def _window_for(box: RegionBox, base: SharpnessMap, params: FocusParams) -> int:
    """按框在分析尺度下的短边选聚合窗口：**窗口必须远小于被测区域**。

    默认窗口（短边 × ``aggregate_ratio`` = 8%）是给"在全图里找唯一锐处"调的，对小鸟框
    太粗（实测：810×1080 的图里一个 150×120 的框，窗口 61px，框只有 1.9 个窗口宽）。
    这里改为把窗口缩进框里：大框保持默认窗口，小框自动换更小的窗口。

    注意全图参照必须用**同一个窗口**重新算（:func:`score_regions` 按窗口缓存清晰度图）：
    不同窗口下清晰度是不同量纲，"框 vs 全图"必须同源。
    """
    min_side = min(box.width, box.height) * base.scale
    windows = max(float(params.min_region_windows), 1.0)
    # 留 25% 余量，避免"恰好等于 N 个窗口"这种刀锋情形来回翻面
    target = int(round(min_side / (windows * 1.25)))
    return odd_kernel(target, minimum=MIN_WINDOW, maximum=base.window)


def score_region_sharpness(
    source: Any,
    box: Any,
    *,
    params: FocusParams | None = None,
) -> RegionSharpness:
    """评一块区域的清晰度（单块区域的便捷入口）。

    :param source: 本地路径、内存字节或 :class:`~whatbird.core.image_input.ImageSource`。
    :param box: ``(x, y, width, height)`` 像素框、:class:`RegionBox`，或带
        ``x``/``y``/``width``/``height`` 的对象（例如检测出来的鸟）。
    :raises WhatBirdError: 图读不出来、图太小 / 没有任何清晰度差异。
    """
    return score_regions(source, [box], params=params).regions[0]


def score_regions(
    source: Any,
    boxes: Iterable[Any],
    *,
    params: FocusParams | None = None,
) -> RegionScoreReport:
    """一次给多个框评分 —— **只建一次清晰度图**，每个框再单独取统计量。

    与位置检测配合时用这个（一张图里的每只鸟各一个框）：建图是主要开销，
    循环里只是取子矩阵做分位数。

    :param boxes: 框的可迭代对象，元素写法见 :func:`_as_box`。
    """
    params = params or FocusParams()
    started = time.perf_counter()

    items = [_as_box(item) for item in boxes]
    if not items:
        raise WhatBirdError(
            "没有给出要评估的区域",
            hint="至少要有一个框：(x, y, width, height) 像素坐标，或用 RegionBox 构造",
        )

    name, image = load_upright_rgb(source)
    width, height = image.size

    smap = build_sharpness_map(image, params=params)
    if not smap.ok:
        raise WhatBirdError(smap.reason, hint="换一张有内容的 jpg 再试")

    # 按窗口缓存：小框要换更小的窗口，而"框 vs 全图"必须同窗口才同量纲
    maps: dict[int, SharpnessMap] = {smap.window: smap}
    tops: dict[int, tuple[Any, int]] = {}

    def context_for(box: RegionBox) -> tuple[SharpnessMap, Any, int]:
        wanted = _window_for(box, smap, params)
        picked = maps.get(wanted)
        if picked is None:
            picked = build_sharpness_map(image, params=params, window=wanted)
            maps[wanted] = picked
        top = tops.get(picked.window)
        if top is None:
            _, np = image_libs()
            cut = float(np.quantile(picked.sharpness, TOP_QUANTILE))
            mask = picked.sharpness >= cut
            top = (mask, max(int(mask.sum()), 1))
            tops[picked.window] = top
        return picked, top[0], top[1]

    regions = []
    for box in items:
        picked, top_mask, top_total = context_for(box)
        regions.append(_score_one(box, picked, params, top_mask, top_total))

    smallest_window = min(item.window for item in regions) if regions else smap.window
    frame_contrast = smap.peak / max(smap.median, 1e-9)
    frame_has_peak = frame_contrast >= _frame_peak_threshold(params)

    notes: list[str] = []
    if not frame_has_peak:
        notes.append(
            f"整张图的清晰度差异很小（峰值/中位数 {frame_contrast:.2f}）：图里没有明显的唯一锐处，"
            "「相对全图最锐处」这个参照本身不可靠，结论只能看百分位 —— 这与对焦分析判"
            "「没有唯一对焦点」是同一条判据"
        )
    if smallest_window < smap.window:
        adapted = sum(1 for item in regions if item.window < smap.window)
        notes.append(
            f"有 {adapted} 个框偏小，已按框的大小改用更小的聚合窗口"
            f"（最小 {smallest_window}px，默认 {smap.window}px）：窗口必须远小于被测区域，"
            "否则框外相邻的锐利内容会被算进来；全图参照用的是同一个窗口，仍然同量纲"
        )
    hedge = _frame_peak_threshold(params) * HEDGE_FACTOR
    if frame_has_peak and frame_contrast < hedge:
        notes.append(
            f"整张图的清晰度差异不大（全图峰值/中位数 {frame_contrast:.2f}，判据阈值 "
            f"{_frame_peak_threshold(params):.2f}）：区域结论说的是「框里是不是最锐的那块」，"
            "不等于整张照片清晰 —— 绝对锐度需要基准标定，本工具不做"
        )
    unreliable = [item for item in regions if not item.reliable]
    if unreliable:
        smallest = min(
            min(item.box.width, item.box.height) for item in unreliable
        )
        notes.append(
            f"有 {len(unreliable)} 个框偏小（最小短边 {smallest}px）：聚合成一个数时会把框外"
            f"相邻内容也算进来，分数只能当参考；默认参数下框的短边至少要占画面短边的 "
            f"{2 * params.aggregate_ratio:.0%}（聚合窗口 {smap.window}px @分析尺度）"
        )
    if any(item.low_detail for item in regions):
        notes.append(
            "有框内几乎没有细节（天空、虚化背景、纯色区域）：这种地方测不出清晰度，"
            "结论是「无法判断」而不是「没对上焦」"
        )
    notes.append(
        "所有分数都是同一张图内部的相对量：跨照片、跨内容比较都不可靠"
        "（实测依据见 docs/focus-algorithm.md）"
    )

    return RegionScoreReport(
        source_image=name,
        width=width,
        height=height,
        analyzed_width=smap.analyzed_width,
        analyzed_height=smap.analyzed_height,
        scale=smap.scale,
        window=smap.window,
        metric=params.metric,
        frame_median=smap.median,
        frame_peak=smap.peak,
        frame_contrast=frame_contrast,
        frame_has_peak=frame_has_peak,
        regions=regions,
        elapsed_seconds=time.perf_counter() - started,
        notes=notes,
        params=params.to_dict(),
    )


def try_score_regions(
    source: Any,
    boxes: Iterable[Any],
    *,
    params: FocusParams | None = None,
) -> tuple[RegionScoreReport | None, str]:
    """给"顺带评一下"的场景用：失败时返回 ``(None, 中文原因)``，而不是抛异常。

    ``score_regions`` 在**纯色 / 完全没有清晰度差异**的图上会抛
    :class:`~whatbird.core.errors.WhatBirdError`（那张图确实没法评）。但这类图对
    "检测鸟的位置"或"识别种类"都是合法输入 —— 调用方（``detect --sharpness``、
    ``/api/detect`` 的 sharpness 开关）不该因为附加的清晰度评分失败而整个请求失败。
    """
    try:
        return score_regions(source, boxes, params=params), ""
    except WhatBirdError as exc:
        return None, exc.message


def _score_one(
    box: RegionBox,
    smap: SharpnessMap,
    params: FocusParams,
    top_mask: Any,
    top_total: int,
) -> RegionSharpness:
    """把一块框映射到分析尺度，取统计量并给出结论。"""
    _, np = image_libs()

    clamped = box.clamped(image_width=smap.source_width, image_height=smap.source_height)
    scale = smap.scale

    x0 = int(round(clamped.x * scale))
    y0 = int(round(clamped.y * scale))
    x1 = int(round((clamped.x + clamped.width) * scale))
    y1 = int(round((clamped.y + clamped.height) * scale))
    x0 = min(max(x0, 0), smap.analyzed_width - 1)
    y0 = min(max(y0, 0), smap.analyzed_height - 1)
    x1 = min(max(x1, x0 + 1), smap.analyzed_width)
    y1 = min(max(y1, y0 + 1), smap.analyzed_height)

    values = smap.sharpness[y0:y1, x0:x1]
    gray = smap.gray[y0:y1, x0:x1]

    quantile = min(max(float(params.region_quantile), 0.0), 1.0)
    sharpness = float(np.quantile(values, quantile))
    peak = float(values.max())
    texture = float(gray.std())

    frame_peak = max(float(smap.peak), 1e-9)
    peak_ratio = min(max(peak / frame_peak, 0.0), 1.0)
    percentile = float((smap.sharpness <= sharpness).mean())
    top_share = float(top_mask[y0:y1, x0:x1].sum()) / float(top_total)

    min_side_analysis = min(x1 - x0, y1 - y0)
    # 窗口已经按框大小自适应过（见 _window_for），所以"不可靠"只剩一种情形：
    # 框小到连两个最小窗口（3px）都放不下。
    needed = MIN_WINDOW * max(float(params.min_region_windows), 0.0)
    reliable = min_side_analysis >= needed
    low_detail = texture < float(params.min_region_texture)

    frame_contrast = smap.peak / max(float(smap.median), 1e-9)
    frame_has_peak = frame_contrast >= _frame_peak_threshold(params)

    verdict = _verdict(
        peak_ratio=peak_ratio,
        top_share=top_share,
        percentile=percentile,
        texture=texture,
        reliable=reliable,
        low_detail=low_detail,
        min_side_analysis=min_side_analysis,
        needed=needed,
        frame_has_peak=frame_has_peak,
        frame_contrast=frame_contrast,
    )
    detail = (
        f"框内最锐处 {peak:.0f} = 全图峰值（{smap.peak:.0f}）的 {peak_ratio:.0%}"
        f" ｜ 全图最锐 1% 像素有 {top_share:.0%} 在框内"
        f" ｜ 框内 p{quantile * 100:.0f} 锐度 {sharpness:.0f}（全图中位 {smap.median:.0f}）"
        f" ｜ 该值落在全图第 {percentile:.0%} 百分位"
        f" ｜ 纹理 σ {texture:.1f} ｜ 聚合窗口 {smap.window}px"
        f" ｜ 分析尺度采样 {int(values.size)} 像素"
    )

    return RegionSharpness(
        box=clamped,
        x_ratio=ratio_of(clamped.x, smap.source_width),
        y_ratio=ratio_of(clamped.y, smap.source_height),
        width_ratio=ratio_of(clamped.width, smap.source_width),
        height_ratio=ratio_of(clamped.height, smap.source_height),
        sharpness=sharpness,
        peak=peak,
        frame_median=smap.median,
        frame_peak=smap.peak,
        frame_contrast=frame_contrast,
        frame_has_peak=frame_has_peak,
        relative_to_frame_peak=peak_ratio,
        top_share=top_share,
        percentile=percentile,
        texture=texture,
        verdict=verdict,
        detail=detail,
        reliable=reliable,
        low_detail=low_detail,
        sampled_pixels=int(values.size),
        window=smap.window,
    )


def _verdict(
    *,
    peak_ratio: float,
    top_share: float,
    percentile: float,
    texture: float,
    reliable: bool,
    low_detail: bool,
    min_side_analysis: int,
    needed: float,
    frame_has_peak: bool,
    frame_contrast: float,
) -> str:
    """把统计量翻成一句能直接给用户看的中文结论。

    顺序是刻意的：**先排除"测不出来 / 参照不成立"的情况，再谈锐不锐**。
    """
    if low_detail:
        return (
            f"无法判断：框内几乎没有可测的细节（纹理 σ {texture:.1f}）—— "
            "天空、虚化背景、纯色面这类区域无论合焦与否都测不出清晰度"
        )
    if not reliable:
        return (
            f"框太小，结论仅供参考：分析尺度下短边只有 {min_side_analysis}px"
            f"（不足 {needed:.0f}px）—— 已经没有任何「窗口远小于区域」的余地；"
            f"按锐度看，框内最锐处是全图峰值的 {peak_ratio:.0%}、位于第 {percentile:.0%} 百分位"
        )
    if not frame_has_peak:
        return (
            f"这张图里没有明显的唯一锐处（全图峰值/中位数仅 {frame_contrast:.2f}）："
            f"「相对最锐处」这个参照不成立，只能说这块区域位于全图第 {percentile:.0%} 百分位"
            " —— 不能据此判断是否合焦（可能整片都在焦内，也可能整张都没对上焦）"
        )
    if peak_ratio >= VERDICT_PEAK and top_share >= VERDICT_TOP_SHARE:
        return (
            f"合焦：画面里最锐的地方就在这个框里（框内最锐处 = 全图峰值的 {peak_ratio:.0%}，"
            f"全图最锐的 1% 像素有 {top_share:.0%} 落在框内）"
        )
    if peak_ratio >= VERDICT_PEAK:
        return (
            f"大体合焦：框内有全图最锐的点（{peak_ratio:.0%}），但那一片最锐的东西主要落在框外"
            f"（只占最锐 1% 像素的 {top_share:.0%}）→ 细节不如画面里最锐的地方"
        )
    if percentile >= VERDICT_ABOVE_TYPICAL:
        return (
            f"比全图大多数地方锐（第 {percentile:.0%} 百分位），但画面最锐的地方在框外"
            f"（框内最锐处只有全图峰值的 {peak_ratio:.0%}）→ 可能轻微失焦，或最锐的东西就在框旁边"
        )
    return (
        f"明显比全图最锐处糊（框内最锐处只有全图峰值的 {peak_ratio:.0%}，"
        f"第 {percentile:.0%} 百分位）→ 多半没对上焦"
    )


def sharpness_tag(score: RegionSharpness) -> str:
    """给标注图用的 **ASCII** 短标签。

    图里不能画中文（不能假设用户机器装了中文字体，见 :mod:`~whatbird.core.focus.overlay`），
    所以把结论压成几个英文标签；分类依据是标志位，不是去匹配 verdict 里的中文。
    """
    if score.low_detail:
        return "NO DETAIL"
    if not score.reliable:
        return "TOO SMALL"
    if not score.frame_has_peak:
        return "NO PEAK"
    if score.relative_to_frame_peak >= VERDICT_PEAK and score.top_share >= VERDICT_TOP_SHARE:
        return "IN FOCUS"
    if score.relative_to_frame_peak >= VERDICT_PEAK:
        return "NEAR FOCUS"
    if score.percentile >= VERDICT_ABOVE_TYPICAL:
        return "SOFT EDGE"
    return "SOFT"


__all__ = [
    "TOP_QUANTILE",
    "VERDICT_ABOVE_TYPICAL",
    "VERDICT_PEAK",
    "VERDICT_TOP_SHARE",
    "score_region_sharpness",
    "score_regions",
    "sharpness_tag",
    "try_score_regions",
]
