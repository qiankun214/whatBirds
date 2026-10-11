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
            "max_centroid_offset": self.max_centroid_offset,            "contrast_ref": self.contrast_ref,
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
