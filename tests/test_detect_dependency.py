"""可选依赖的隔离性：没装 ultralytics / torch 时，其余功能必须照常可用。"""

from __future__ import annotations

import builtins

import pytest

from whatbird.core import DetectorDependencyError, detect_birds
from whatbird.core.detect import backend
from whatbird.core.detect.models import DetectParams


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
