"""对焦分析的「内容」路线：只看像素，找出画面里最清晰的位置。

为什么可以这样做
----------------
相机合焦的那一小块区域成像最锐利 —— 也就是高频分量最强。所以只要把"局部有多锐"
量化出来，最强的地方就是实际合焦处。这正是传统反差对焦（contrast detection）的原理，
只是这里改成在整幅图上一次性做。

算法链路（每一步都对应一个具体的坑）
------------------------------------
1. **灰度 + 缩放**：缩小时用 ``INTER_AREA``。用 ``INTER_LINEAR`` 会自己制造高频，
   把清晰度图污染掉。
2. **高斯去噪**：噪声也是高频。不去噪的话，传感器噪点和 JPEG 块效应会被当成
   "很清晰"，结果直接跑飞。
3. **清晰度图**：SML（改进拉普拉斯）—— 对去噪后的图取拉普拉斯绝对值，再做局部求和。
   这是经典的清晰度评价函数。之所以够用、不需要额外做对比度归一化，是因为**二阶
   导数天然拒绝失焦的高对比边缘**：一片缓坡（失焦边缘）的一阶导数可能很大，但二阶
   导数接近零。实测过"严重失焦的黑白大边缘 vs 极低对比的锐利小纹理"，SML 稳定选中
   后者（见 ``tests/test_focus_content.py``）。
4. **合焦区域**：用 max(Otsu 阈值, 峰值 × 比例) 二值化，取包含峰值的连通域。
   相机的一个对焦点本来覆盖的就是一片区域（景深），不是一个几何点。
5. **对焦点 = 合焦区域内的锐度加权质心**。
   这一步是实测后的关键选择：直接取"最锐的那个像素"非常不稳，在合成样本上偏差能到
   30~50 像素（清晰块内部本来就有起伏）；而**区域内的锐度加权质心**在同一批样本上
   误差为 0，且天然抗噪。加权（而不是取几何中心）保证结果会被拉向真正最锐的地方。
6. **置信度**：峰值相对全图中位数的倍数。整幅图清晰度都差不多（全图合焦或全图失焦）
   时倍数很小，此时必须老实承认"找不到明确对焦点"，而不是硬报一个坐标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import FocusDependencyError
from .models import (
    METRIC_TENENGRAD,
    METHOD_CONTENT,
    FocusHit,
    FocusParams,
    FocusPoint,
    FocusRegion,
    ratio_of,
)

#: 小于这个尺寸就别算了：聚合窗口会大于图像本身，结果没有意义
MIN_SIDE = 32

#: 局部极大值候选的排序上限（够用就行，避免在平坦图上对几十万像素做排序）
_MAX_CANDIDATES = 500

#: 合焦区域小于整图这个比例时，认为它只是一个噪点，不作为区域上报
_MIN_REGION_RATIO = 0.0005

_LIBS: tuple[Any, Any] | None = None


def _libs() -> tuple[Any, Any]:
    """惰性导入 opencv 与 numpy。

    刻意放在函数里而不是模块顶层：``import whatbird.core`` 会被 CLI 的每个子命令
    （包括 ``config show``）触发，不该为了一个功能付 opencv 的导入开销。
    """
    global _LIBS
    if _LIBS is None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise FocusDependencyError(
                "对焦分析缺少依赖：opencv-python-headless / numpy",
                hint=(
                    "python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
                    "opencv-python-headless"
                ),
            ) from exc
        _LIBS = (cv2, np)
    return _LIBS


@dataclass(slots=True)
class ContentAnalysis:
    """内容路线的结果（坐标已换算回原图）。"""

    hit: FocusHit
    region: FocusRegion | None = None
    candidates: list[FocusPoint] = field(default_factory=list)
    analyzed_width: int = 0
    analyzed_height: int = 0
    scale: float = 1.0
    contrast: float = 0.0

    @property
    def ok(self) -> bool:
        return self.hit.found


# --------------------------------------------------------------------- 工具
def _odd(value: int, *, minimum: int = 3, maximum: int | None = None) -> int:
    """夹到 ``[minimum, maximum]`` 内并保证是奇数（OpenCV 卷积核习惯用奇数）。"""
    number = max(int(value), int(minimum))
    if maximum is not None:
        number = min(number, int(maximum))
    if number % 2 == 0:
        number = number - 1 if number - 1 >= minimum else number + 1
    return max(number, minimum)


def _otsu_threshold(values: Any, np: Any, *, bins: int = 256) -> float:
    """显式实现的 Otsu 阈值（最大类间方差）。

    不用 ``cv2.threshold`` 的 OTSU 模式，是因为那条路要求先把浮点图压成 8 位，
    白白丢精度；这里直接在浮点直方图上算，逻辑也更好读。
    """
    flat = values.reshape(-1)
    low = float(flat.min())
    high = float(flat.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return high

    hist, edges = np.histogram(flat, bins=bins, range=(low, high))
    total = float(flat.size)
    centers = (edges[:-1] + edges[1:]) / 2.0

    weight_background = np.cumsum(hist).astype(np.float64)
    weight_foreground = total - weight_background
    cumulative = np.cumsum(hist * centers)

    valid = (weight_background > 0) & (weight_foreground > 0)
    if not valid.any():  # pragma: no cover - 只有全常数图才会走到
        return high

    total_sum = float((hist * centers).sum())
    mean_background = cumulative / np.maximum(weight_background, 1.0)
    mean_foreground = (total_sum - cumulative) / np.maximum(weight_foreground, 1.0)
    between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    between = np.where(valid, between, -1.0)
    return float(centers[int(np.argmax(between))])


def _weighted_centroid(mask: Any, weights: Any, np: Any) -> tuple[float, float] | None:
    """在 ``mask`` 内按 ``weights`` 加权求质心（先返回分析尺度坐标）。"""
    masked = np.where(mask > 0, weights, 0.0)
    total = float(masked.sum())
    if total <= 0:
        return None
    height, width = masked.shape
    x = float((masked.sum(axis=0) * np.arange(width, dtype=np.float64)).sum() / total)
    y = float((masked.sum(axis=1) * np.arange(height, dtype=np.float64)).sum() / total)
    return x, y


def _clamp(value: int, original: int) -> int:
    return int(min(max(value, 0), max(original - 1, 0)))


# --------------------------------------------------------------------- 主流程
def analyze_content(rgb: Any, *, params: FocusParams | None = None) -> ContentAnalysis:
    """对一幅图像做内容分析。

    :param rgb: ``uint8`` 的 ``(H, W, 3)`` numpy 数组，或已按 EXIF 转正的 PIL 图像。
    :returns: :class:`ContentAnalysis`，坐标**已换算回原图**。
    """
    cv2, np = _libs()
    params = params or FocusParams()

    if hasattr(rgb, "mode"):  # PIL.Image
        rgb = np.asarray(rgb.convert("RGB"))

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if min(height, width) < MIN_SIDE:
        return ContentAnalysis(
            hit=FocusHit(
                method=METHOD_CONTENT,
                found=False,
                detail=f"图片太小（{width}×{height}），无法做清晰度分析",
            ),
            analyzed_width=width,
            analyzed_height=height,
        )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ---- 1. 缩放（缩小时必须用 INTER_AREA） ----
    longest = max(width, height)
    if longest > params.max_side:
        factor = params.max_side / float(longest)
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
            interpolation=cv2.INTER_AREA,
        )
    analyzed_height, analyzed_width = int(gray.shape[0]), int(gray.shape[1])
    scale = analyzed_width / float(width)

    grayf = gray.astype(np.float32)

    # ---- 2. 去噪 ----
    sigma = max(float(params.denoise_sigma), 0.0)
    if sigma > 0:
        grayf = cv2.GaussianBlur(grayf, (0, 0), sigma)

    short_side = min(analyzed_height, analyzed_width)
    window = _odd(
        round(short_side * params.aggregate_ratio),
        minimum=3,
        maximum=_odd(max(short_side // 2, 3), minimum=3),
    )

    # ---- 3. 清晰度图（局部求和，绝对量；不做比值归一化） ----
    if params.metric == METRIC_TENENGRAD:
        gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
        energy = gx * gx + gy * gy
        metric_label = "Tenengrad 梯度"
    else:
        energy = np.abs(cv2.Laplacian(grayf, cv2.CV_32F, ksize=3))
        metric_label = "SML 改进拉普拉斯"

    sharpness = cv2.boxFilter(energy, -1, (window, window), normalize=False)

    peak = float(sharpness.max())
    baseline = float(np.median(sharpness))
    if not np.isfinite(peak) or peak <= 0:
        return ContentAnalysis(
            hit=FocusHit(
                method=METHOD_CONTENT,
                found=False,
                detail="画面没有可用的清晰度差异（可能是纯色或完全均匀的图）",
            ),
            analyzed_width=analyzed_width,
            analyzed_height=analyzed_height,
            scale=scale,
        )

    eps = max(peak, 1.0) * 1e-6
    flat_index = int(np.argmax(sharpness))
    peak_y, peak_x = divmod(flat_index, analyzed_width)

    # ---- 4. 合焦区域 ----
    threshold = max(_otsu_threshold(sharpness, np), peak * float(params.peak_ratio))
    binary = (sharpness >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    label = int(labels[peak_y, peak_x])
    if label <= 0 and count > 1:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    region_analyzed = None
    mask = None
    if label > 0:
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / float(analyzed_width * analyzed_height)
        if area_ratio >= _MIN_REGION_RATIO:
            mask = (labels == label).astype(np.uint8)
            region_analyzed = FocusRegion(
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                width=int(stats[label, cv2.CC_STAT_WIDTH]),
                height=int(stats[label, cv2.CC_STAT_HEIGHT]),
                area_ratio=area_ratio,
                centroid_x=int(round(float(centroids[label][0]))),
                centroid_y=int(round(float(centroids[label][1]))),
            )

    # ---- 5. 对焦点 ----
    # 只有"合焦区域确实是**一片**区域"时，才配用它的锐度加权质心当对焦点。
    # 两种不合格的情况，都是真实照片里踩出来的：
    #   - 铺满画面（占比过大）：那是"整张都在焦内"，质心不过是画面中心；
    #   - 稀疏骨架（质心落在空隙里）：阈值化后剩下的常是树叶边缘与枝条，
    #     质心会掉进骨架中间 —— 实测落在一根明显失焦的树枝上。
    # 两种情况都退回"最锐的那个点"，也更贴合输出里"最锐处大致在…"的措辞。
    area_ratio = region_analyzed.area_ratio if region_analyzed else 0.0
    if region_analyzed is not None and area_ratio > params.max_region_for_centroid:
        region_analyzed = None
        mask = None

    focal = _weighted_centroid(mask, sharpness, np) if mask is not None else None
    if focal is not None:
        # 质心离峰值超过若干个聚合窗口，说明它们在对两个不同的地方做结论 ——
        # 真实照片里这是"稀疏骨架把质心拽到了空隙里"，此时以峰值为准。
        offset = ((focal[0] - peak_x) ** 2 + (focal[1] - peak_y) ** 2) ** 0.5
        if offset > window * max(float(params.max_centroid_offset), 0.0):
            focal = None

    if focal is None:
        focal = (float(peak_x), float(peak_y))
    focus_x, focus_y = focal

    # ---- 6. 置信度 ----
    contrast = peak / (baseline + eps)
    confidence = 0.0
    reference = max(float(params.contrast_ref), 1.0 + 1e-6)
    if contrast > 1.0:
        confidence = float(np.log10(contrast) / np.log10(reference))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if area_ratio > 0.6:
        # 整幅图都"清晰"：景深极大或全图失焦，本来就没有明确的焦点可言
        confidence *= max(0.25, 1.0 - (area_ratio - 0.6) / 0.4 * 0.75)

    found = confidence >= params.min_confidence

    def to_original(value: float, limit: int) -> int:
        return _clamp(int(round(value / scale)), limit)

    focus_x_i = to_original(focus_x, width)
    focus_y_i = to_original(focus_y, height)

    region = None
    if region_analyzed is not None:
        region = FocusRegion(
            x=to_original(region_analyzed.x, width),
            y=to_original(region_analyzed.y, height),
            width=max(1, int(round(region_analyzed.width / scale))),
            height=max(1, int(round(region_analyzed.height / scale))),
            area_ratio=region_analyzed.area_ratio,
            centroid_x=to_original(region_analyzed.centroid_x, width),
            centroid_y=to_original(region_analyzed.centroid_y, height),
        )

    # ---- 7. 候选点：非极大值抑制 ----
    # 无论 found 与否都算出来：分布平坦时，把"还有这些同样锐的地方"摆出来，
    # 比只丢一个"没找到"要有用得多。
    radius = max(1, window // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(sharpness, kernel)
    maxima = (sharpness >= dilated - eps) & (sharpness > 0)
    ys, xs = np.nonzero(maxima)
    picked: list[tuple[int, int]] = [(peak_x, peak_y)]
    if xs.size:
        order = np.argsort(sharpness[ys, xs])[::-1][:_MAX_CANDIDATES]
        for index in order:
            cx, cy = int(xs[index]), int(ys[index])
            if all((cx - px) ** 2 + (cy - py) ** 2 >= radius * radius for px, py in picked):
                picked.append((cx, cy))
                if len(picked) >= max(int(params.top_n), 1):
                    break

    candidates: list[FocusPoint] = [
        FocusPoint(
            x=focus_x_i,
            y=focus_y_i,
            score=1.0,
            x_ratio=ratio_of(focus_x_i, width),
            y_ratio=ratio_of(focus_y_i, height),
        )
    ]
    near_peak = 0
    for cx, cy in picked[1:]:
        value = float(sharpness[cy, cx]) / peak
        if value >= 0.9:
            near_peak += 1
        ox, oy = to_original(cx, width), to_original(cy, height)
        candidates.append(
            FocusPoint(
                x=ox,
                y=oy,
                score=round(value, 4),
                x_ratio=ratio_of(ox, width),
                y_ratio=ratio_of(oy, height),
            )
        )

    if found:
        area_text = f"，合焦区域占画面 {area_ratio:.1%}" if region else ""
        detail = (
            f"{metric_label}清晰度峰值位于合焦区域中心"
            f"（峰值/中位数 = {contrast:.1f}{area_text}）"
        )
    else:
        # 这种照片（典型：整片树冠都在焦内）不是"算失败"，而是"确实没有唯一焦点"。
        # 所以要把最锐的地方、以及有多少处差不多的，都交代清楚。
        detail = (
            f"整图清晰度差异极小，没有唯一的对焦点"
            f"（峰值/中位数仅 {contrast:.2f}，最锐处大致在 x={focus_x_i} y={focus_y_i}"
            + (f"，另有 {near_peak} 处清晰度与之接近" if near_peak else "")
            + "；可能整片都在焦内，或整张都没对上焦）"
        )

    hit = FocusHit(
        method=METHOD_CONTENT,
        found=found,
        x=focus_x_i,
        y=focus_y_i,
        x_ratio=ratio_of(focus_x_i, width),
        y_ratio=ratio_of(focus_y_i, height),
        confidence=confidence,
        detail=detail,
    )

    return ContentAnalysis(
        hit=hit,
        region=region,
        candidates=candidates,
        analyzed_width=analyzed_width,
        analyzed_height=analyzed_height,
        scale=scale,
        contrast=contrast,
    )
