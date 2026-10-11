"""鸟类位置检测：给一张 jpg，用 YOLO 找出画面里每一只鸟的位置。

用法::

    from whatbird.core import detect_birds, render_bird_overlay

    report = detect_birds(r"D:\\photos\\birds.jpg")
    for bird in report.birds:
        print(bird.x, bird.y, bird.width, bird.height, f"{bird.confidence:.0%}")

    Path("out.png").write_bytes(render_bird_overlay(r"D:\\photos\\birds.jpg", report))

设计要点
--------
- **只报鸟**：预训练权重是 COCO 80 类，``bird`` 是独立类别（id 14），所以默认
  ``DetectParams.classes = (14,)``；其他类别只记进 ``report.other_classes``，
  用来解释「画面里明明有目标，为什么没报鸟」。
- **不联网、不消耗额度**：与对焦点分析一样是纯本地推理，不读方舟配置；
  只有第一次需要下载 YOLO 权重（可提前手动放到权重目录，见 ``requirements-detect.txt``）。
- **Intel 显卡优先**：``device="auto"`` 时能用 XPU 就用（本机是 Arc A770），
  不可用自动退回 CPU，并且把**实际**设备如实写进 ``report.device``。
- **可选依赖**：没装 ultralytics / torch 时本模块可以正常 import，
  只有真正调用检测才会抛 :class:`~whatbird.core.errors.DetectorDependencyError`。
  测试可以注入假后端（``detector=``），从而在没有显卡与权重的机器上跑通全部业务逻辑。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..errors import DetectorDependencyError
from ..image_input import ImageSource, load_upright_rgb
from . import backend
from .backend import Detector, YoloDetector
from .models import (
    BIRD_CLASS_ID,
    BIRD_CLASS_NAME,
    DEFAULT_MODEL,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_XPU,
    Detection,
    DetectionReport,
    DetectParams,
    RawBox,
)
from .overlay import (
    DetectionStyle,
    overlay_data_url,
    render_bird_overlay,
    render_detection_overlay,
)

#: 可注入的检测后端：``(已转正的 RGB 图, 参数) -> (归一化原始框, 实际设备)``
DetectorLike = Callable[[Any, DetectParams], tuple[list[RawBox], str]]


def detect_birds(
    source: "ImageSource | str | bytes | Any",
    *,
    params: DetectParams | None = None,
    detector: DetectorLike | None = None,
) -> DetectionReport:
    """检测一张图片里所有鸟的位置。

    :param source: 本地 jpg 路径、内存字节、:class:`~whatbird.core.image_input.ImageSource`，
        或已经读好的 PIL 图像。
    :param params: 检测参数；``None`` 用默认值（只报鸟、置信度 0.25、设备 auto）。
    :param detector: 自定义检测后端（测试注入假后端用）；``None`` 走 ultralytics。
    :raises ImageInputError: 路径不存在、不是 jpg、文件损坏。
    :raises DetectorDependencyError: 缺少 ultralytics / torch，或权重不可用。
    """
    params = params or DetectParams()
    device = backend.resolve_device(params.device)
    started = time.perf_counter()

    name, image = load_upright_rgb(source)
    width, height = image.size

    active = detector if detector is not None else backend.get_detector(params)
    try:
        raw_boxes, used_device = active(image, params)
    except DetectorDependencyError:
        raise
    except Exception as exc:  # pragma: no cover - 推理阶段的第三方异常统一换中文
        raise DetectorDependencyError(
            f"YOLO 推理失败：{type(exc).__name__}: {exc}",
            hint="可先用 --device cpu 判断是不是显卡相关的问题；也可能是权重文件损坏，删掉后重新下载",
        ) from exc

    birds, other_classes, notes = backend.build_detections(
        raw_boxes,
        width=width,
        height=height,
        classes=params.classes,
    )
    notes.extend(_extra_notes(birds=birds, device=used_device or device, params=params))

    return DetectionReport(
        source_image=name,
        width=width,
        height=height,
        birds=birds,
        other_classes=other_classes,
        device=used_device or device,
        model_name=params.model_name,
        elapsed_seconds=time.perf_counter() - started,
        notes=notes,
        params=params.to_dict(),
    )


def _extra_notes(
    *,
    birds: Sequence[Detection],
    device: str,
    params: DetectParams,
) -> list[str]:
    """补充「用户可能困惑」的两条说明：没找到鸟、显卡没用上。"""
    notes: list[str] = []
    if not birds:
        notes.append(
            "未检测到鸟：可能画面里确实没有鸟，也可能是鸟太小/太远、被遮挡，"
            "或置信度阈值偏高；可以先调低 --conf 再试"
        )
    elif device != DEVICE_XPU and params.device == DEVICE_AUTO:
        notes.append(
            f"本次使用 {device} 推理（未用上 Intel 显卡）：XPU 需要 torch 的 XPU 版本，"
            "结果不受影响，只是慢一些"
        )
    return notes


__all__ = [
    "BIRD_CLASS_ID",
    "BIRD_CLASS_NAME",
    "DEFAULT_MODEL",
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_XPU",
    "Detection",
    "DetectionReport",
    "DetectionStyle",
    "DetectParams",
    "Detector",
    "DetectorLike",
    "ImageSource",
    "RawBox",
    "YoloDetector",
    "detect_birds",
    "overlay_data_url",
    "render_bird_overlay",
    "render_detection_overlay",
]