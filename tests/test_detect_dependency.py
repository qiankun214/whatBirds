"""可选依赖的隔离性：没装 ultralytics / torch 时，其余功能必须照常可用。"""

from __future__ import annotations

import builtins

import pytest

from whatbird.core import DetectorDependencyError, detect_birds
from whatbird.core.detect import backend
from whatbird.core.detect.models import BIRD_CLASS_ID, DetectParams


@pytest.fixture(autouse=True)
def clean_cache():
    backend.clear_cache()
    yield
    backend.clear_cache()


def test_import_whatbird_core_does_not_need_torch_or_ultralytics(monkeypatch):
    """把两个包都伪装成未安装，import whatbird.core 仍然必须成功。"""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise ImportError(f"blocked {name} for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    import importlib

    module = importlib.reload(importlib.import_module("whatbird.core"))
    assert hasattr(module, "detect_birds")


def test_missing_ultralytics_raises_dependency_error_with_mirror_hint(jpeg_factory, monkeypatch):
    path = jpeg_factory(name="dep.jpg")
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise ImportError(f"blocked {name} for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.setattr(backend, "xpu_available", lambda: False)

    with pytest.raises(DetectorDependencyError) as info:
        detect_birds(path, params=DetectParams(model_dir=path.parent))

    assert "ultralytics" in info.value.message
    assert "pypi.tuna.tsinghua.edu.cn" in info.value.hint


def test_xpu_available_is_false_when_torch_is_missing(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert backend.xpu_available() is False


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(backend, "xpu_available", lambda: False)
    assert backend.resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_xpu(monkeypatch):
    monkeypatch.setattr(backend, "xpu_available", lambda: True)
    assert backend.resolve_device("auto") == "xpu"


def test_explicit_xpu_without_device_is_an_error(monkeypatch):
    monkeypatch.setattr(backend, "xpu_available", lambda: False)
    with pytest.raises(DetectorDependencyError) as info:
        backend.resolve_device("xpu")
    assert "--device cpu" in info.value.hint


def test_unknown_device_is_rejected():
    with pytest.raises(DetectorDependencyError):
        backend.resolve_device("tpu")


def test_model_path_prefers_local_file(tmp_path):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"fake")
    params = DetectParams(model_name=str(weights), model_dir=tmp_path / "models")
    assert backend.resolve_model_path(params) == str(weights)


def test_model_path_uses_existing_file_in_model_dir(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "yolo11n.pt").write_bytes(b"fake")
    params = DetectParams(model_name="yolo11n.pt", model_dir=model_dir)
    assert backend.resolve_model_path(params) == str(model_dir / "yolo11n.pt")


def test_model_path_falls_back_to_model_dir_for_download(tmp_path):
    params = DetectParams(model_name="yolo11n.pt", model_dir=tmp_path)
    assert backend.resolve_model_path(params) == str(tmp_path / "yolo11n.pt")


def test_blank_model_name_is_rejected():
    with pytest.raises(DetectorDependencyError):
        backend.resolve_model_path(DetectParams(model_name="   "))


def test_model_dir_env_var_overrides_the_default(tmp_path, monkeypatch):
    """服务端没有 --model-dir 可传，只能靠环境变量指到已有的权重目录。"""
    monkeypatch.setenv("WHATBIRD_MODEL_DIR", str(tmp_path))

    params = DetectParams(model_name="yolo11n.pt")

    assert params.resolved_model_dir() == tmp_path
    assert backend.resolve_model_path(params) == str(tmp_path / "yolo11n.pt")


def test_explicit_model_dir_beats_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("WHATBIRD_MODEL_DIR", str(tmp_path / "from-env"))

    params = DetectParams(model_name="yolo11n.pt", model_dir=tmp_path / "explicit")

    assert params.resolved_model_dir() == tmp_path / "explicit"


def test_unwritable_model_dir_gives_chinese_hint(tmp_path, monkeypatch):
    """权重目录不可写时不能抛裸 PermissionError：CLI/Web 都会把它翻成 500。"""

    def boom(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("pathlib.Path.mkdir", boom)

    with pytest.raises(DetectorDependencyError) as info:
        backend.ensure_model_dir(tmp_path / "readonly" / "yolo11n.pt")

    assert "权重目录不可写" in info.value.message
    assert "WHATBIRD_MODEL_DIR" in info.value.hint


# --------------------------------------------------------------------- 后端参数
class _Array:
    """够用的数组替身：只需要 ``tolist()``（真身是 numpy 数组）。"""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeBoxes:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    @property
    def xyxy(self):
        return _Array([row[:4] for row in self._rows])

    @property
    def conf(self):
        return _Array([row[4] for row in self._rows])

    @property
    def cls(self):
        return _Array([row[5] for row in self._rows])


class _FakeResult:
    def __init__(self, rows, shape):
        self.boxes = _FakeBoxes(rows)
        self.orig_shape = shape


class _FakeModel:
    """假 YOLO 模型：记录 ``predict`` 收到的参数，并返回预设的像素框。"""

    names = {14: "bird", 16: "dog"}

    def __init__(self, rows, shape=(100, 200)):
        self._rows = rows
        self._shape = shape
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult(self._rows, self._shape)]


def _detector_with(model, device="cpu"):
    """绕过 ``__init__`` 构造 YoloDetector（它会 import ultralytics / 加载权重）。"""
    detector = object.__new__(backend.YoloDetector)
    detector._params = DetectParams()
    detector._device = device
    detector._model_path = "yolo11n.pt"
    detector._model = model
    return detector


def test_backend_never_pushes_the_class_filter_down_to_the_model():
    """回归：``predict(classes=[14])`` 会让非鸟框根本回不到上层。

    症状是默认参数下 ``report.other_classes`` 永远为空 —— 文档承诺的
    「解释为什么没报鸟」直接失效，只有 ``--all-classes`` 才有内容。
    真机实测过：一张有公交车和人的照片，加了 --all-classes 才会出现 bus / person。
    """
    model = _FakeModel([[10, 20, 60, 80, 0.9, 16]])  # 一个 dog（COCO id 16）
    detector = _detector_with(model)

    boxes, device = detector(None, DetectParams(classes=(BIRD_CLASS_ID,)))

    assert model.calls[0]["classes"] is None, "类别过滤不能下推给模型，否则无法解释「为什么没报鸟」"
    assert [item.class_name for item in boxes] == ["dog"]
    assert device == "cpu"


def test_backend_passes_the_documented_predict_arguments():
    model = _FakeModel([])
    detector = _detector_with(model)
    params = DetectParams(imgsz=960, conf=0.4, iou=0.5, max_det=20)

    detector(None, params)

    call = model.calls[0]
    assert call["imgsz"] == 960
    assert call["conf"] == pytest.approx(0.4)
    assert call["iou"] == pytest.approx(0.5)
    assert call["max_det"] == 20
    assert call["device"] == "cpu"
    assert call["verbose"] is False
