"""对焦分析的数据契约。

设计要点：**两条互相独立的路线各出一份结论，谁也不覆盖谁**。

- ``exif`` 路线读相机写进元数据里的对焦点（相机*声称*对焦点在哪）；
- ``content`` 路线分析像素本身的清晰度分布（画面*实际*最清晰处在哪）。

两条路线都可能失败（照片没有 EXIF、画面全图失焦……），所以 :class:`FocusHit`
用 ``found`` 表示"这一路有没有结论"，用 ``detail`` 携带一句可以直接展示给用户的中文说明。

坐标系约定
----------
所有对外的 ``x`` / ``y`` 都是**按 EXIF Orientation 转正后的原图**像素坐标：
原点在左上角，``x`` 向右、``y`` 向下。同时提供 ``x_ratio`` / ``y_ratio``
（0~1 的相对位置），方便前端按比例把标记叠到缩略图上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 两条路线的标识
METHOD_EXIF = "exif"
METHOD_CONTENT = "content"

#: 支持的内容分析算子
METRIC_SML = "sml"
METRIC_TENENGRAD = "tenengrad"
METRICS: tuple[str, ...] = (METRIC_SML, METRIC_TENENGRAD)


def _ratio(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(min(max(value / total, 0.0), 1.0), 4)


@dataclass(slots=True)
class FocusParams:
    """内容分析的参数（默认值即生产使用的值，全部可在 CLI / API 上覆盖）。"""

    max_side: int = 1024
    """分析时图片最长边的上限。先缩小再算，既省时间也顺带压掉一部分噪声。"""

    denoise_sigma: float = 1.0
    """算清晰度之前的 3x3 高斯去噪强度。噪声也是高频，不去掉会被当成"很清晰"。"""

    metric: str = METRIC_SML
    """:data:`METRIC_SML` 或 :data:`METRIC_TENENGRAD`。"""

    aggregate_ratio: float = 0.08
    """聚合窗口占短边的比例；窗口越大，"对焦区域"的粒度越粗、结果越稳。"""

    peak_ratio: float = 0.5
    """合焦区域阈值 = max(Otsu 阈值, 峰值 * 该比例)。"""

    top_n: int = 5
    """上报的候选对焦点个数。"""

    max_region_for_centroid: float = 0.25
    """合焦区域占画面的比例超过该值，就不再把它的质心当对焦点。

    "对焦点"的前提是有一片**有限**的合焦区域。若阈值区域已经铺满画面
    （实测的一张 OM-1 鸟片是 43%），它的质心基本就是画面中心 —— 那不是对焦点，
    是编出来的坐标。
    """

    max_centroid_offset: float = 2.0
    """质心与峰值相距超过多少个聚合窗口时，不采用质心、改用峰值（默认 2 个窗口）。

    这是被真实照片逼出来的判据。真实照片阈值化之后，剩下的往往是**树叶边缘与枝条
    构成的稀疏骨架**，它的加权质心会落在离峰值很远的地方 —— 实测一张 OM-1 鸟片的
    质心落在了一根**明显失焦的树枝**上，而峰值在 1000 像素外的浆果上。

    直接拿"质心与峰值的距离"来判，命中失败原因而且没有中间的模糊地带：
    合成样本里那块实心清晰区的质心离峰值只有 1.4~1.8 个窗口（描述的是同一个地方），
    真实照片的骨架质心离峰值 3.3 个窗口（描述的是两个地方）。

    不用"质心处的清晰度"来判：清晰度图是窗口**求和**，质心旁边有锐的东西就会把
    它的值抬起来，实测那张落点在模糊树枝上的照片该比值仍有 0.63，判不出来。
    """

    contrast_ref: float = 10.0
    """峰值 / 中位数 达到该值时，内容分析置信度记 1.0。"""

    region_quantile: float = 0.75
    """区域评分取框内清晰度图的哪个分位数当"这块区域有多锐"（默认 0.75）。

    不用均值：框住一只鸟时，框里难免带上一点背景（树枝、天空），均值会被背景稀释；
    也不用最大值：那只是一个像素的窗口和，容易被偶发的锐点（一根细枝、一个噪点边缘）抬高。
    p75 是"这块区域里普遍有多锐"的稳健代表，实测在合成样本与真实鸟片上都能分开
    "合焦的鸟"与"失焦的鸟"。
    """

    min_region_texture: float = 3.0
    """框内灰度标准差低于该值，就认为"这个区域没有可测的细节"。

    天空、纯色背景、完全虚化的水面这类区域，清晰度值恒接近 0 —— **无论合焦与否都测不出
    清晰度**。此时必须明说"无法判断"，而不是报一个"很糊"的分数（那是把"没内容"
    误读成"没对上焦"）。

    3.0 是实测定的：纯色/天空区域 σ ≈ 1.1（含 JPEG 块噪声），**同样模糊量下有内容的
    区域 σ ≈ 5.6**（模糊的噪声），而合焦的鸟 σ ≈ 36~48。阈值必须落在 1.1 与 5.6 之间，
    取 3.0 给"纯色但带压缩噪声"留余量，又不至于把"软但仍有很多细节"的区域误判成没内容。
    """

    min_region_windows: float = 2.0
    """框的短边至少要等于几个聚合窗口，否则分数标为"不可靠"。

    聚合窗口本身就有边长，窗口内的内容会被算进这个框的分数 —— 框越小，窗口"越界"
    带进来的相邻内容越多（§4.3 踩过这个坑）。默认 2 个窗口对应一条很好记的判据：
    **框的短边至少要有画面短边的 16%**（窗口 ≈ 短边 × `aggregate_ratio` = 8%）。
    """

    min_confidence: float = 0.45
    """低于该置信度就认为"没有可信的对焦点"，``found`` 置 False。

    0.45 对应"峰值只比全图中位数高约 2.8 倍"。真实照片里若真有明确焦点，
    这个倍数通常远高于此；而"整片树叶都在焦内"这类照片只有 2~3 倍，
    此时报一个坐标就是误导 —— 应当明说没有唯一对焦点（候选点仍会列出来）。

    这个默认值是拿一张真实 OM-1 鸟片校准的：那张照片在全分辨率下
    峰值/中位数只有 2.3~2.6，主体处于全图第 98.5 百分位但并非峰值，
    说明整片树冠大致同在一个距离面上 —— 本来就没有唯一对焦点。
    """

    use_exif: bool = True
    use_content: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "max_side": self.max_side,
            "denoise_sigma": self.denoise_sigma,
            "metric": self.metric,
            "aggregate_ratio": self.aggregate_ratio,
            "peak_ratio": self.peak_ratio,
            "top_n": self.top_n,
            "max_region_for_centroid": self.max_region_for_centroid,
            "max_centroid_offset": self.max_centroid_offset,
            "contrast_ref": self.contrast_ref,
            "region_quantile": self.region_quantile,
            "min_region_texture": self.min_region_texture,
            "min_region_windows": self.min_region_windows,
            "min_confidence": self.min_confidence,
            "use_exif": self.use_exif,
            "use_content": self.use_content,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FocusParams":
        """容忍缺字段/多字段地把字典转成参数对象（向上层开放覆盖）。"""
        data = data or {}
        known = set(cls().to_dict())
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class FocusPoint:
    """一个候选对焦点。"""

    x: int
    y: int
    score: float = 0.0
    """相对清晰度，0~1（主峰为 1.0）。"""

    x_ratio: float = 0.0
    y_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "score": round(self.score, 4),
            "x_ratio": self.x_ratio,
            "y_ratio": self.y_ratio,
        }


@dataclass(slots=True)
class FocusRegion:
    """合焦区域的包围盒（清晰度超过阈值的连通域）。"""

    x: int
    y: int
    width: int
    height: int
    area_ratio: float = 0.0
    centroid_x: int = 0
    centroid_y: int = 0

    def contains(self, x: int, y: int, *, tolerance: int = 0) -> bool:
        """判断某个点是否落在（可带容差的）区域里 —— 两条路线"是否互相印证"就靠它。"""
        return (
            self.x - tolerance <= x <= self.x + self.width + tolerance
            and self.y - tolerance <= y <= self.y + self.height + tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area_ratio": round(self.area_ratio, 4),
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
        }


@dataclass(slots=True)
class RegionBox:
    """要评估的一块矩形区域（``x``/``y``/``width``/``height``，与原图同口径）。

    构造时请留意坐标语义：这里的 ``width``/``height`` 是**宽高**，不是右下角坐标。
    从检测框（``Detection.box`` 是 ``xyxy``）转过来用 :meth:`from_xyxy`，
    从 0~1 相对坐标转过来用 :meth:`from_ratios`。
    """

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "RegionBox":
        """从 ``xyxy``（左上 + 右下）构造 —— 检测框就是这个口径。"""
        left, top = int(round(min(x1, x2))), int(round(min(y1, y2)))
        right, bottom = int(round(max(x1, x2))), int(round(max(y1, y2)))
        return cls(x=left, y=top, width=max(right - left, 1), height=max(bottom - top, 1))

    @classmethod
    def from_ratios(
        cls,
        x_ratio: float,
        y_ratio: float,
        width_ratio: float,
        height_ratio: float,
        *,
        image_width: int,
        image_height: int,
    ) -> "RegionBox":
        """从 0~1 相对坐标构造（前端在缩略图上框选时用这个）。"""
        return cls.from_xyxy(
            x_ratio * image_width,
            y_ratio * image_height,
            (x_ratio + width_ratio) * image_width,
            (y_ratio + height_ratio) * image_height,
        )

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamped(self, *, image_width: int, image_height: int) -> "RegionBox":
        """钳进画面内（框可以越界，但越界的部分没有像素可评）。"""
        left = min(max(self.x, 0), max(image_width - 1, 0))
        top = min(max(self.y, 0), max(image_height - 1, 0))
        right = min(max(self.x + self.width, left + 1), image_width)
        bottom = min(max(self.y + self.height, top + 1), image_height)
        return RegionBox(x=left, y=top, width=right - left, height=bottom - top)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class RegionSharpness:
    """一块区域的清晰度评分 —— **全部是相对量，只在同一张图内部有意义**。

    跨照片比较需要基准标定，本工具刻意不做（详见 ``docs/focus-algorithm.md``）。
    """

    box: RegionBox
    x_ratio: float = 0.0
    y_ratio: float = 0.0
    width_ratio: float = 0.0
    height_ratio: float = 0.0

    sharpness: float = 0.0
    """框内清晰度图的 ``region_quantile`` 分位数（默认 p75）—— 这块区域"普遍有多锐"。"""

    peak: float = 0.0
    """框内最锐的那个像素（窗口求和值，仍是绝对量）。"""

    frame_median: float = 0.0
    frame_peak: float = 0.0
    relative_to_frame_peak: float = 0.0
    """框内**最锐处** ÷ 全图最锐处 —— 主指标。**1.0 表示全图最锐的点就在这个框里**。

    刻意用"最锐处 ÷ 最锐处"而不是"框内分位数 ÷ 全图峰值"：后者是两个不可比的量
    （一块区域的稳健分位数 vs 单个窗口的极大值），实测会把**合焦的鸟**判成"只有最锐处的
    42%、可能轻微失焦"，而且逐级模糊时该比值还会反向上升（模糊先抹掉全图峰值，
    框内分位数反而相对抬起来）。同量纲相比才有意义。
    """

    top_share: float = 0.0
    """全图最锐的那 1% 像素里，有多少比例落在这个框内。

    "最高只到峰值的 9X%" 在峰值是**一个孤立噪点**时会误判；这个占比回答的是
    "画面里那一片最锐的东西是不是就在框里"，比单点比较稳。
    """

    frame_contrast: float = 0.0
    """全图峰值 ÷ 全图中位数。太小说明整张图清晰度都差不多（全图合焦或全图失焦）。"""

    frame_has_peak: bool = True
    """全图是否存在"明显的唯一锐处"。

    为 False 时 :attr:`relative_to_frame_peak` 的参照本身不成立（一张整片树叶都在焦内的
    照片就是这种），结论必须退回到百分位，并且明说不能据此判断是否合焦。
    判据与对焦分析共用同一套标定：``min_confidence`` / ``contrast_ref``。
    """

    percentile: float = 0.0
    """框内锐度落在全图清晰度分布的哪个百分位。"""

    texture: float = 0.0
    """框内灰度标准差：判断"这个区域有没有细节可测"（天空/虚化背景接近 0）。"""

    verdict: str = ""
    """一句可直接展示的中文结论。"""

    detail: str = ""
    """数字细节（锐度、相对值、百分位、纹理、采样像素数）。"""

    reliable: bool = True
    """框是否足够大（短边 ≥ ``min_region_windows`` 个聚合窗口）；False 时分数仅供参考。"""

    low_detail: bool = False
    """框里几乎没有可测细节：此时"清晰度"本身无从谈起，结论必须是"无法判断"。"""

    sampled_pixels: int = 0
    """框在分析尺度下覆盖了多少像素（给"框太小"这类判断提供依据）。"""

    window: int = 0
    """这一块评分时用的聚合窗口边长（分析尺度像素）。

    按框的大小自适应：窗口必须远小于被测区域，否则框外相邻内容会被算进来。
    大框用默认窗口；小框会自动换更小的窗口（于是全图参照也跟着换成同一窗口下的统计量，
    保证"框 vs 全图"始终同量纲）。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "x_ratio": self.x_ratio,
            "y_ratio": self.y_ratio,
            "width_ratio": self.width_ratio,
            "height_ratio": self.height_ratio,
            "sharpness": round(self.sharpness, 3),
            "peak": round(self.peak, 3),
            "frame_median": round(self.frame_median, 3),
            "frame_peak": round(self.frame_peak, 3),
            "frame_contrast": round(self.frame_contrast, 3),
            "frame_has_peak": self.frame_has_peak,
            "relative_to_frame_peak": round(self.relative_to_frame_peak, 4),
            "top_share": round(self.top_share, 4),
            "percentile": round(self.percentile, 4),
            "texture": round(self.texture, 3),
            "verdict": self.verdict,
            "detail": self.detail,
            "reliable": self.reliable,
            "low_detail": self.low_detail,
            "sampled_pixels": self.sampled_pixels,
            "window": self.window,
        }


@dataclass(slots=True)
class RegionScoreReport:
    """一次「区域清晰度评分」的完整结果（CLI 与 Web 共用的对外结构）。"""

    source_image: str = ""
    width: int = 0
    height: int = 0
    analyzed_width: int = 0
    analyzed_height: int = 0
    scale: float = 1.0
    window: int = 0
    metric: str = METRIC_SML
    frame_median: float = 0.0
    frame_peak: float = 0.0
    frame_contrast: float = 0.0
    frame_has_peak: bool = True
    regions: list[RegionSharpness] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def top(self) -> RegionSharpness | None:
        """最锐的那一块（列表顺序与传入顺序一致，这里按分数选）。"""
        if not self.regions:
            return None
        return max(self.regions, key=lambda item: item.sharpness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "image": {
                "width": self.width,
                "height": self.height,
                "analyzed_width": self.analyzed_width,
                "analyzed_height": self.analyzed_height,
                "scale": round(self.scale, 6),
                "window": self.window,
            },
            "metric": self.metric,
            "frame": {
                "median": round(self.frame_median, 3),
                "peak": round(self.frame_peak, 3),
                "contrast": round(self.frame_contrast, 3),
                "has_peak": self.frame_has_peak,
            },
            "regions": [item.to_dict() for item in self.regions],
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "notes": list(self.notes),
            "params": dict(self.params),
        }


@dataclass(slots=True)
class FocusHit:
    """一条路线给出的结论。``found=False`` 时 ``detail`` 一定是一句可展示的中文原因。"""

    method: str
    found: bool = False
    x: int = 0
    y: int = 0
    x_ratio: float = 0.0
    y_ratio: float = 0.0
    confidence: float = 0.0
    detail: str = ""
    source_tags: dict[str, str] = field(default_factory=dict)
    """仅 exif 路线使用：来源标签的**白名单**子集（make / model / tag / raw_value）。

    这里刻意不是"整个 EXIF 字典"：EXIF 里还有 GPS 等敏感信息，
    绝不能顺手透传到 API、CLI 或前端。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "found": self.found,
            "x": self.x,
            "y": self.y,
            "x_ratio": self.x_ratio,
            "y_ratio": self.y_ratio,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "source_tags": dict(self.source_tags),
        }


@dataclass(slots=True)
class FocusReport:
    """一次对焦分析的完整结果（CLI 与 Web 共用的对外结构）。"""

    source_image: str = ""
    width: int = 0
    height: int = 0
    analyzed_width: int = 0
    analyzed_height: int = 0
    scale: float = 1.0
    elapsed_seconds: float = 0.0
    exif: FocusHit = field(default_factory=lambda: FocusHit(method=METHOD_EXIF))
    content: FocusHit = field(default_factory=lambda: FocusHit(method=METHOD_CONTENT))
    region: FocusRegion | None = None
    candidates: list[FocusPoint] = field(default_factory=list)
    agree: bool = False
    notes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def available_methods(self) -> list[str]:
        return [hit.method for hit in (self.exif, self.content) if hit.found]

    def summary_line(self) -> str:
        """一行摘要，CLI 控制台用。"""
        bits: list[str] = []
        for hit in (self.exif, self.content):
            if hit.found:
                bits.append(f"{'EXIF' if hit.method == METHOD_EXIF else '内容'} ({hit.x},{hit.y})")
        if not bits:
            return "未能定位对焦点"
        head = " ｜ ".join(bits)
        return f"{head} ｜ 两者" + ("一致" if self.agree else "不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "image": {
                "width": self.width,
                "height": self.height,
                "analyzed_width": self.analyzed_width,
                "analyzed_height": self.analyzed_height,
                "scale": round(self.scale, 6),
            },
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "agree": self.agree,
            "exif": self.exif.to_dict(),
            "content": self.content.to_dict(),
            "region": self.region.to_dict() if self.region else None,
            "candidates": [item.to_dict() for item in self.candidates],
            "notes": list(self.notes),
            "params": dict(self.params),
        }


def ratio_of(value: int, total: int) -> float:
    """把像素坐标换算成 0~1 的相对位置（前端叠加用）。"""
    return _ratio(value, total)
