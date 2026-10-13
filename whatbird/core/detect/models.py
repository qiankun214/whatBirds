"""鸟类位置检测的数据契约。

设计要点与 ``core/focus/models.py`` 保持一致：**这里是 core 与上层（CLI / Web / 未来 UI）
之间唯一的数据结构**，CLI 与 Web 共用同一份 ``to_dict()``，所以新增字段上层自动获得。

坐标系约定
----------
对外暴露的 ``x`` / ``y`` / ``width`` / ``height`` 都是**按 EXIF Orientation 转正后的原图**
像素坐标：原点在左上角，``x`` 向右、``y`` 向下；同时给出 0~1 的 ``*_ratio``，方便前端
按比例把框叠到缩略图上（与对焦点分析完全同一套约定）。

类别约定
--------
COCO 预训练模型里 ``bird`` 是独立类别（id 14），所以第一版**只报鸟**：
``DetectParams.classes`` 默认就是 ``(BIRD_CLASS_ID,)``。其余类别只记在
:attr:`DetectionReport.other_classes` 里，用来解释「画面里明明有框，为什么没报出来」。

这个过滤是**事后**做的（core 拿到全部类别的框之后再筛）：模型始终检测全部类别，
否则非鸟框根本回不到上层，「为什么没报鸟」就无从解释。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: COCO 数据集的鸟类类别 id（ultralytics 预训练权重的类别顺序与 COCO 一致）
BIRD_CLASS_ID = 14

#: 该类别的英文名（COCO 官方名字）
BIRD_CLASS_NAME = "bird"

#: 默认只检测鸟
COCO_BIRD_CLASSES: tuple[int, ...] = (BIRD_CLASS_ID,)

#: 默认权重：YOLO11 nano，最小最快，找位置够用（首次运行会自动下载）
DEFAULT_MODEL = "yolo11n.pt"

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300

#: 设备取值：auto 先试 Intel 显卡（XPU），不可用再退 CPU
DEVICE_AUTO = "auto"
DEVICE_XPU = "xpu"
DEVICE_CPU = "cpu"
DEVICES: tuple[str, ...] = (DEVICE_AUTO, DEVICE_XPU, DEVICE_CPU)

#: 默认权重目录（用户目录下，与 API 配置同级的隔离区，天然不会被提交到 git）
DEFAULT_MODEL_DIR = Path.home() / ".whatbird" / "models"

#: 覆盖权重目录的环境变量（与 ``WHATBIRD_CONFIG_FILE`` 同一约定）。
#: 需要它是因为 Web 界面没有 ``--model-dir`` 可传：受限环境（沙箱、只读用户目录、
#: 权重放在项目内）下得有个办法让服务端找到已有权重，而不是去联网下载。
MODEL_DIR_ENV = "WHATBIRD_MODEL_DIR"


def default_model_dir() -> Path:
    """权重目录：``WHATBIRD_MODEL_DIR`` 优先，否则 :data:`DEFAULT_MODEL_DIR`。"""
    override = os.environ.get(MODEL_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_MODEL_DIR


def _clamp(value: float, limit: float) -> float:
    return min(max(value, 0.0), limit)


def _ratio(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(_clamp(value / total, 1.0), 4)


def xyxy_to_xywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """把 ``xyxy`` 框钳到画面内并转成 ``(x, y, w, h)``（整数像素）。

    检测框常有一两个像素越界（模型自己也可能给出负坐标），越界不修正的话
    ``x + w`` 会超出图像尺寸，画到标注图上就要靠绘图库兜底 —— 不如在这里一次性修正：
    宽高至少 1 像素，保证 ``x + w <= 图像宽``、``y + h <= 图像高``。
    """
    left = int(round(_clamp(x1, max(width - 1, 0))))
    top = int(round(_clamp(y1, max(height - 1, 0))))
    right = int(round(_clamp(x2, width)))
    bottom = int(round(_clamp(y2, height)))

    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    if left + box_width > width:
        left = max(width - box_width, 0)
    if top + box_height > height:
        top = max(height - box_height, 0)
    return left, top, box_width, box_height


@dataclass(slots=True)
class Detection:
    """一只鸟（或某个被检测到的目标）的位置。"""

    x: int
    y: int
    width: int
    height: int
    x_ratio: float = 0.0
    y_ratio: float = 0.0
    width_ratio: float = 0.0
    height_ratio: float = 0.0
    confidence: float = 0.0
    class_id: int = BIRD_CLASS_ID
    class_name: str = BIRD_CLASS_NAME

    @property
    def box(self) -> tuple[int, int, int, int]:
        """``(x1, y1, x2, y2)``，画标注图直接用。"""
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x_ratio": self.x_ratio,
            "y_ratio": self.y_ratio,
            "width_ratio": self.width_ratio,
            "height_ratio": self.height_ratio,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass(slots=True)
class RawBox:
    """检测后端的原始输出：**归一化** 的 ``xyxy`` + 置信度 + 类别。

    后端（ultralytics）只负责「模型说什么」，归一化与像素换算都放在 :func:`build_detection`，
    这样测试可以用一个假后端直接喂归一化坐标，不需要为了造 640x480 的框去算像素。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    class_id: int = BIRD_CLASS_ID
    class_name: str = BIRD_CLASS_NAME


def build_detection(raw: RawBox, *, width: int, height: int) -> Detection:
    """把归一化原始框换算成对外可用的 :class:`Detection`。"""
    x, y, box_width, box_height = xyxy_to_xywh(
        raw.x1 * width,
        raw.y1 * height,
        raw.x2 * width,
        raw.y2 * height,
        width=width,
        height=height,
    )
    return Detection(
        x=x,
        y=y,
        width=box_width,
        height=box_height,
        x_ratio=_ratio(x, width),
        y_ratio=_ratio(y, height),
        width_ratio=_ratio(box_width, width),
        height_ratio=_ratio(box_height, height),
        confidence=_clamp(float(raw.confidence), 1.0),
        class_id=int(raw.class_id),
        class_name=raw.class_name or "",
    )


@dataclass(slots=True)
class DetectParams:
    """检测参数（默认值即生产使用的值，全部可在 CLI / 未来的 API 上覆盖）。"""

    model_name: str = DEFAULT_MODEL
    """权重名或本地权重路径。名字会先到 :attr:`model_dir` 里找，找不到才交给 ultralytics 去下载。"""

    model_dir: Path | None = None
    """权重目录；``None`` 表示 :data:`DEFAULT_MODEL_DIR`。存在的本地权重优先，避免重复下载。"""

    device: str = DEVICE_AUTO
    """:data:`DEVICE_AUTO` / :data:`DEVICE_XPU` / :data:`DEVICE_CPU`。"""

    imgsz: int = DEFAULT_IMGSZ
    """推理分辨率。越大越慢越能抓到远处的小鸟，640 是速度与精度的常用平衡点。"""

    conf: float = DEFAULT_CONF
    """置信度阈值：低于它的框不返回。调低会多召回，但也更可能把树叶当成鸟。"""

    iou: float = DEFAULT_IOU
    """NMS 的 IoU 阈值：重叠超过它的框会被合并，避免同一只鸟出好几个框。"""

    classes: tuple[int, ...] = COCO_BIRD_CLASSES
    """哪些类别算结果；空元组表示不过滤（非鸟目标也进结果，对应 CLI 的 ``--all-classes``）。

    这是**事后过滤**（在 core 层做）：模型始终检测全部类别，因为非鸟框要用来解释
    「画面里明明有目标，为什么没报鸟」。它也不会让推理变快 —— ultralytics 本来就是
    先推理再按类别筛，``classes`` 只是过滤 NMS 之后的列表。
    """

    max_det: int = DEFAULT_MAX_DET
    """单张图最多返回多少个框（防止密集画面下爆内存 / 刷屏）。"""

    agnostic_nms: bool = False
    """True 时跨类别做 NMS；只检测鸟的情况下两者等价，保留是为了行为可控。"""

    def resolved_model_dir(self) -> Path:
        return Path(self.model_dir).expanduser() if self.model_dir else default_model_dir()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_dir": str(self.resolved_model_dir()),
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": list(self.classes),
            "max_det": self.max_det,
            "agnostic_nms": self.agnostic_nms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DetectParams":
        """容忍缺字段/多字段地把字典转成参数对象（向上层开放覆盖）。

        ``classes`` 允许传列表，也允许传 ``None``（表示不限制类别 → 空元组）。
        """
        data = dict(data or {})
        if "classes" in data:
            value = data["classes"]
            data["classes"] = () if value is None else tuple(int(item) for item in value)
        known = set(cls().to_dict())
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class DetectionReport:
    """一次检测的完整结果（CLI 与 Web 共用的对外结构）。"""

    source_image: str = ""
    width: int = 0
    height: int = 0
    birds: list[Detection] = field(default_factory=list)
    """检测到的目标，按置信度降序。默认只含鸟；``classes`` 为空时才可能含非鸟目标。"""

    other_classes: list[str] = field(default_factory=list)
    """本次画面里出现、但不算结果的类别名（按字母序去重），用来解释「为什么没报鸟」。"""

    device: str = ""
    """实际使用的推理设备（``xpu`` / ``cpu``）——回退时这里会如实上报，不假装用了显卡。"""

    model_name: str = ""
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.birds)

    @property
    def found(self) -> bool:
        return bool(self.birds)

    @property
    def top(self) -> Detection | None:
        """置信度最高的那只（列表已按置信度降序）。"""
        return self.birds[0] if self.birds else None

    def summary_line(self) -> str:
        """一行摘要，CLI 控制台用。

        默认只有鸟算结果，所以平时就是「检测到 N 只鸟」；``--all-classes`` 下结果里
        可能混着非鸟目标，此时如实说明其中几只是鸟，不要谎报成「N 只鸟」。
        """
        if not self.birds:
            return "未检测到鸟"
        bird_count = sum(1 for item in self.birds if item.class_id == BIRD_CLASS_ID)
        if bird_count == self.count:
            head = f"检测到 {self.count} 只鸟"
        else:
            head = f"检测到 {self.count} 个目标（其中 {bird_count} 只鸟）"
        best = self.top
        if best is not None:
            head += f"（最高置信度 {best.confidence:.0%}）"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_image": self.source_image,
            "image": {"width": self.width, "height": self.height},
            "count": self.count,
            "found": self.found,
            "birds": [item.to_dict() for item in self.birds],
            "other_classes": list(self.other_classes),
            "device": self.device,
            "model_name": self.model_name,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "notes": list(self.notes),
            "params": dict(self.params),
        }


def sort_by_confidence(detections: Iterable[Detection]) -> list[Detection]:
    """按置信度降序、同分按面积降序（大框更可能是主体），保证输出稳定。"""
    return sorted(detections, key=lambda item: (-item.confidence, -item.area, item.x, item.y))


def unique_class_names(names: Sequence[str]) -> list[str]:
    return sorted({name for name in names if name})
