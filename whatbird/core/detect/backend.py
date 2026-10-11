"""YOLO 检测后端：**唯一**依赖 ultralytics / torch 的地方。

三层隔离，保证「没装也能用其余功能」：

1. 本模块顶层**不 import** torch / ultralytics —— ``import whatbird.core``（CLI 每个子命令都会
   触发）不应该为了一个可选能力付出加载 PyTorch 的代价，更不该在没装检测依赖时直接崩掉；
2. 依赖缺失统一抛 :class:`~whatbird.core.errors.DetectorDependencyError`，附带按清华镜像的安装命令；
3. 权重缺失 / 下载失败抛同类型异常，并给出手动放置权重的办法（国内访问 GitHub 常失败）。

坐标口径：后端只输出**归一化**的 ``xyxy``（见 :class:`RawBox`），像素换算交给
:func:`whatbird.core.detect.models.build_detection`。这样测试用的假后端不需要构造真实尺寸的框，
真后端也不必重复写一遍像素钳制逻辑。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from ..errors import DetectorDependencyError
from .models import (
    BIRD_CLASS_ID,
    DEVICE_AUTO,
    DEVICE_CPU,
    DEVICE_XPU,
    DEVICES,
    Detection,
    DetectParams,
    RawBox,
    build_detection,
    sort_by_confidence,
    unique_class_names,
)

#: 缺失依赖时统一提示的安装源（按 Agent.md 第 2 条，普通包必须用清华镜像）
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_XPU_INDEX = "https://download.pytorch.org/whl/xpu"

INSTALL_HINT = (
    "请按顺序装检测依赖（见 requirements-detect.txt）：\n"
    f"  1) python -m pip install -i {TORCH_XPU_INDEX} torch==2.8.0 torchvision==0.23.0\n"
    f"  2) python -m pip install -i {PIP_MIRROR} --no-deps ultralytics\n"
    "第 2 步必须加 --no-deps：ultralytics 声明的是 opencv-python（含 GUI），"
    "会顶掉本项目刻意选的 opencv-python-headless"
)

#: 模型下载失败时的中文提示（含手动兜底路径）
DOWNLOAD_HINT = (
    "首次使用需要下载 YOLO 权重；若网络不可达，可手动下载后放进权重目录，"
    "或用 --model <本地权重路径> 直接指定（文件名需与 --model 给的名字一致）"
)

class Detector(Protocol):
    """检测后端接口：给一张已转正的 RGB 图，返回归一化原始框。

    上层只依赖这个协议，所以测试可以注入假后端（不加载模型、不下载权重），
    也能在没装 ultralytics 的机器上跑通 core 的全部业务逻辑。
    """

    def __call__(self, image: Any, params: DetectParams) -> tuple[list[RawBox], str]:
        """返回 ``(归一化原始框列表, 实际使用的设备)``。"""
        ...


# --------------------------------------------------------------------- 设备
def xpu_available() -> bool:
    """Intel 显卡（XPU）当前是否可用。

    单独抽成函数，方便测试用 monkeypatch 覆盖「有显卡 / 没显卡」两条分支。
    """
    try:
        import torch
    except ImportError:
        return False
    xpu = getattr(torch, "xpu", None)
    if xpu is None:  # CPU 版 torch 没有 torch.xpu
        return False
    try:
        return bool(xpu.is_available())
    except Exception:  # pragma: no cover - 驱动异常/初始化失败都按不可用处理
        return False


def resolve_device(requested: str) -> str:
    """把 ``auto`` 解析成真实设备名。

    - ``auto``：能用 Intel 显卡就用（本机是 Arc A770），否则退回 CPU；
    - 显式指定 ``xpu`` 但不可用：直接报错，而不是悄悄退回 CPU 让人以为跑了显卡。
    """
    value = (requested or DEVICE_AUTO).strip().lower()
    if value == DEVICE_AUTO:
        return DEVICE_XPU if xpu_available() else DEVICE_CPU
    if value not in DEVICES:
        raise DetectorDependencyError(
            f"不支持的推理设备：{requested}",
            hint=f"可用取值：{' / '.join(DEVICES)}；auto 会优先用 Intel 显卡，不可用时退回 CPU",
        )
    if value == DEVICE_XPU and not xpu_available():
        raise DetectorDependencyError(
            "指定了 xpu（Intel 显卡）推理，但当前环境检测不到可用的 XPU 设备",
            hint=(
                '可能是 torch 装的是 CPU 版、驱动过旧，或显卡被独占；先跑 '
                'python -c "import torch;print(torch.xpu.is_available())" 确认，'
                "或改用 --device cpu / --device auto"
            ),
        )
    return value


# --------------------------------------------------------------------- 权重
def resolve_model_path(params: DetectParams) -> str:
    """算出权重的本地路径：已存在就直连，避免每次联网查一遍。

    - ``model_name`` 指向已存在的文件：原样使用（允许 ``--model D:\\models\\yolo11n.pt``）；
    - 否则看 :attr:`DetectParams.model_dir` 里有没有同名权重：有就用本地的（离线可用）；
    - 都没有：返回「权重目录 + 文件名」交给 ultralytics，由它按官方逻辑下载到该目录。
    """
    name = (params.model_name or "").strip()
    if not name:
        raise DetectorDependencyError("没有指定 YOLO 权重", hint="用 --model 指定权重名或本地权重路径")

    candidate = Path(name).expanduser()
    if candidate.is_file():
        return str(candidate)
    return str(params.resolved_model_dir() / candidate.name)


def resolve_classes(params: DetectParams) -> list[int] | None:
    """ultralytics 的 ``classes`` 参数：``None`` 表示不限制类别。"""
    return list(params.classes) if params.classes else None

# --------------------------------------------------------------------- 后端
class YoloDetector:
    """ultralytics YOLO 后端：一次加载，多次推理（按需缓存，见 :func:`get_detector`）。"""

    def __init__(self, params: DetectParams) -> None:
        self._params = params
        self._device = resolve_device(params.device)
        self._model_path = resolve_model_path(params)
        self._model = self._load()

    # ------------------------------------------------------------- 属性
    @property
    def device(self) -> str:
        return self._device

    @property
    def model_path(self) -> str:
        return self._model_path

    def _load(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise DetectorDependencyError(
                "鸟类位置检测缺少依赖：ultralytics / torch",
                hint=INSTALL_HINT,
            ) from exc

        path = Path(self._model_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model = YOLO(self._model_path)
        except Exception as exc:  # 权重损坏、格式不支持、目录不可写……
            raise DetectorDependencyError(
                f"加载 YOLO 权重失败：{self._model_path}",
                hint=f"{DOWNLOAD_HINT}；原始错误：{type(exc).__name__}: {exc}",
            ) from exc

        if self._device == DEVICE_XPU:
            try:
                model.to(DEVICE_XPU)
            except Exception as exc:  # pragma: no cover - 加载后被抢占等
                raise DetectorDependencyError(
                    f"把模型放到 Intel 显卡（xpu）上失败：{type(exc).__name__}: {exc}",
                    hint="可先用 --device cpu 验证功能是否正常，再排查显卡驱动",
                ) from exc
        return model

    # ------------------------------------------------------------- 推理
    def __call__(self, image: Any, params: DetectParams) -> tuple[list[RawBox], str]:
        results = self._model.predict(
            source=image,
            imgsz=params.imgsz,
            conf=params.conf,
            iou=params.iou,
            max_det=params.max_det,
            classes=resolve_classes(params),
            agnostic_nms=params.agnostic_nms,
            device=self._device,
            verbose=False,
        )
        return raw_boxes_from_results(results, class_names=self.class_names()), self._device

    def class_names(self) -> dict[int, str]:
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(names)}
        return {int(key): str(value) for key, value in dict(names).items()}


# --------------------------------------------------------------------- 结果转换
def raw_boxes_from_results(
    results: Iterable[Any],
    *,
    class_names: dict[int, str] | None = None,
) -> list[RawBox]:
    """把 ultralytics 的 ``Results`` 转成归一化 ``RawBox``。

    这里**不做类别过滤**（那是 core 门面的职责）：全部类别都带出来，
    这样「画面里有狗没鸟」这种情况上层还能解释清楚。
    """
    names = class_names or {}
    boxes: list[RawBox] = []
    for result in results or []:
        found = getattr(result, "boxes", None)
        if found is None or len(found) == 0:
            continue

        shape = getattr(result, "orig_shape", None) or ()
        if len(shape) < 2:
            continue
        height, width = float(shape[0]), float(shape[1])
        if width <= 0 or height <= 0:
            continue

        for coords, confidence, class_id in zip(
            found.xyxy.tolist(),
            found.conf.tolist(),
            found.cls.tolist(),
        ):
            identifier = int(class_id)
            boxes.append(
                RawBox(
                    x1=float(coords[0]) / width,
                    y1=float(coords[1]) / height,
                    x2=float(coords[2]) / width,
                    y2=float(coords[3]) / height,
                    confidence=float(confidence),
                    class_id=identifier,
                    class_name=names.get(identifier) or str(identifier),
                )
            )
    return boxes

# --------------------------------------------------------------------- 类别过滤
def build_detections(
    raw_boxes: Sequence[RawBox],
    *,
    width: int,
    height: int,
    classes: Sequence[int] | None,
) -> tuple[list[Detection], list[str], list[str]]:
    """按「只报鸟」的规则过滤并换算成对外的 :class:`Detection`。

    返回 ``(鸟的框列表, 被过滤掉的类别名, 相关提示)``。这条产品规则刻意放在 core 层
    而不是后端：**没有显卡、没有权重**的环境也能把它完整测掉。

    非鸟类别不会进入 ``birds``，只会进 ``other_classes`` 并附一句中文说明，
    用来解释「画面里明明有目标，为什么结果里没有鸟」。
    也就是说 ``classes`` 只影响「过滤掉哪些类别名要上报」，不影响「birds 只装鸟」这条规则。
    """
    birds: list[Detection] = []
    filtered: dict[int, str] = {}

    for raw in raw_boxes:
        if raw.class_id == BIRD_CLASS_ID:
            birds.append(build_detection(raw, width=width, height=height))
        else:
            filtered[raw.class_id] = raw.class_name

    notes: list[str] = []
    if filtered:
        pairs = "、".join(f"{name}(id {identifier})" for identifier, name in sorted(filtered.items()))
        notes.append(f"画面里还有非鸟目标（按参数过滤，不计入结果）：{pairs}")
    return sort_by_confidence(birds), unique_class_names(list(filtered.values())), notes


# --------------------------------------------------------------------- 缓存
_CACHE: dict[tuple[str, str], YoloDetector] = {}
_CACHE_LOCK = threading.Lock()

#: 门面接受的检测器类型（协议是结构性的，这里只是给读者一个名字）
DetectorLike = Callable[[Any, DetectParams], tuple[list[RawBox], str]]


def get_detector(params: DetectParams) -> YoloDetector:
    """按「权重路径 + 设备」缓存后端：CLI 一次给多张图时不必反复加载模型。

    只缓存加载成功的后端；失败会直接抛异常，下一次调用重新尝试（用户可能刚把权重放好）。
    """
    key = (resolve_model_path(params), resolve_device(params.device))
    with _CACHE_LOCK:
        detector = _CACHE.get(key)
        if detector is None:
            detector = YoloDetector(params)
            _CACHE[key] = detector
        return detector


def clear_cache() -> None:
    """清空后端缓存（测试用，避免设备切换 / 假后端被缓存串味）。"""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "Detector",
    "DetectorLike",
    "DOWNLOAD_HINT",
    "INSTALL_HINT",
    "PIP_MIRROR",
    "TORCH_XPU_INDEX",
    "YoloDetector",
    "build_detections",
    "clear_cache",
    "get_detector",
    "raw_boxes_from_results",
    "resolve_classes",
    "resolve_device",
    "resolve_model_path",
    "xpu_available",
]