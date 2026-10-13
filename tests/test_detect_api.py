"""鸟类位置检测的 REST 接口测试。

检测要加载 YOLO 权重，CI / 新克隆的仓库里没有，所以这里**注入假后端**：
只验证 REST 层契约（参数透传、错误翻译、标注图编码），不碰真实推理。
画框本身是真跑的（Pillow，很快），所以标注图那条链路是真实验证过的。
"""

from __future__ import annotations

import base64
import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from whatbird.core import DetectorDependencyError
from whatbird.core.detect.models import (
    BIRD_CLASS_ID,
    COCO_BIRD_CLASSES,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    Detection,
    DetectionReport,
)

web_app = importlib.import_module("whatbird.web.app")

HEADERS = {"X-WhatBird-Client": "1"}
client = TestClient(web_app.app)


def _jpg_bytes(size=(480, 360)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 140, 100)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _report(width: int, height: int, birds: int = 1) -> DetectionReport:
    report = DetectionReport(
        source_image="api.jpg",
        width=width,
        height=height,
        device="cpu",
        model_name="yolo11n.pt",
        elapsed_seconds=0.42,
    )
    for index in range(birds):
        report.birds.append(
            Detection(
                x=10 + index * 20,
                y=20,
                width=40,
                height=30,
                confidence=0.9 - index * 0.1,
                class_id=BIRD_CLASS_ID,
                class_name="bird",
            )
        )
    return report


@pytest.fixture
def patched_detect(monkeypatch):
    """把 ``web_app.detect_birds`` 换成假实现，并记录收到的参数。"""

    def install(size=(480, 360), birds=1, raises=None, other_classes=None):
        calls = {"params": []}

        def fake(source, *, params=None, detector=None):
            calls["params"].append(params)
            if raises is not None:
                raise raises
            report = _report(size[0], size[1], birds=birds)
            report.other_classes = list(other_classes or [])
            return report

        monkeypatch.setattr(web_app, "detect_birds", fake)
        return calls

    return install


# ------------------------------------------------------------------ 上传接口
def test_detect_upload_returns_result_and_overlay(patched_detect):
    patched_detect(size=(480, 360), birds=1)

    response = client.post(
        "/api/detect",
        headers=HEADERS,
        files={"file": ("api.jpg", _jpg_bytes((480, 360)), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["count"] == 1
    assert payload["result"]["device"] == "cpu"

    overlay = payload["overlay"]
    assert overlay.startswith("data:image/png;base64,")
    raw = base64.b64decode(overlay.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (480, 360)


def test_detect_upload_uses_documented_defaults(patched_detect):
    calls = patched_detect()

    response = client.post(
        "/api/detect", headers=HEADERS, files={"file": ("d.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    params = calls["params"][0]
    assert params.conf == pytest.approx(DEFAULT_CONF)
    assert params.imgsz == DEFAULT_IMGSZ
    assert params.classes == COCO_BIRD_CLASSES


def test_detect_upload_honours_form_overrides(patched_detect):
    calls = patched_detect()

    response = client.post(
        "/api/detect",
        headers=HEADERS,
        files={"file": ("d.jpg", _jpg_bytes(), "image/jpeg")},
        data={"conf": "0.15", "imgsz": "960", "all_classes": "true"},
    )

    assert response.status_code == 200
    params = calls["params"][0]
    assert params.conf == pytest.approx(0.15)
    assert params.imgsz == 960
    assert params.classes == (), "all_classes 应当清空类别过滤"


def test_detect_upload_requires_client_header():
    response = client.post("/api/detect", files={"file": ("csrf.jpg", _jpg_bytes(), "image/jpeg")})

    assert response.status_code == 403


def test_detect_upload_rejects_oversized_file():
    oversized = b"\xff\xd8" + b"\x00" * (web_app.MAX_UPLOAD_BYTES + 10)

    response = client.post(
        "/api/detect", headers=HEADERS, files={"file": ("huge.jpg", oversized, "image/jpeg")}
    )

    assert response.status_code == 413
    assert "过大" in response.json()["detail"]


def test_detect_upload_rejects_corrupt_file(patched_detect):
    patched_detect()

    response = client.post(
        "/api/detect",
        headers=HEADERS,
        files={"file": ("broken.jpg", b"definitely not a jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "无法解析" in response.json()["error"]["message"]


def test_detect_upload_without_weights_returns_chinese_hint(patched_detect):
    """权重没下好时前端要看到中文提示 + 下载办法，而不是 500 或裸栈。"""
    patched_detect(raises=DetectorDependencyError("加载 YOLO 权重失败", hint="可用 tools/fetch_yolo_weight.py 换镜像下载"))

    response = client.post(
        "/api/detect", headers=HEADERS, files={"file": ("d.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "DetectorDependencyError"
    assert "权重" in error["message"]
    assert "fetch_yolo_weight" in error["hint"]


# ------------------------------------------------------------------ 路径接口
def test_detect_path_endpoint(patched_detect, jpeg_factory):
    calls = patched_detect()
    path = jpeg_factory(name="bypath.jpg")

    response = client.post("/api/detect-path", headers=HEADERS, json={"path": str(path)})

    assert response.status_code == 200
    assert calls["params"][0].classes == COCO_BIRD_CLASSES


def test_detect_path_honours_json_overrides(patched_detect, jpeg_factory):
    calls = patched_detect()
    path = jpeg_factory(name="bypath.jpg")

    response = client.post(
        "/api/detect-path",
        headers=HEADERS,
        json={"path": str(path), "conf": 0.15, "imgsz": 960, "all_classes": True},
    )

    assert response.status_code == 200
    params = calls["params"][0]
    assert params.conf == pytest.approx(0.15)
    assert params.imgsz == 960
    assert params.classes == ()


def test_detect_path_rejects_unknown_fields(jpeg_factory):
    """拼错字段不能静默忽略（否则用户以为参数生效了）。"""
    path = jpeg_factory(name="strict.jpg")

    response = client.post(
        "/api/detect-path", headers=HEADERS, json={"path": str(path), "confidence": 0.2}
    )

    assert response.status_code == 422


def test_detect_path_missing_file_gives_chinese_hint(tmp_path):
    missing = tmp_path / "nope.jpg"

    response = client.post("/api/detect-path", headers=HEADERS, json={"path": str(missing)})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "ImageInputError"
    assert "不存在" in error["message"]
    assert error["hint"]


def test_detect_path_rejects_non_jpg(tmp_path):
    png = tmp_path / "bird.png"
    Image.new("RGB", (40, 40)).save(png, format="PNG")

    response = client.post("/api/detect-path", headers=HEADERS, json={"path": str(png)})

    assert response.status_code == 400
    assert "暂不支持" in response.json()["error"]["message"]


# ------------------------------------------------------------------ 其它契约
def test_detect_endpoints_are_registered_in_openapi():
    paths = set(web_app.app.openapi()["paths"])
    assert "/api/detect" in paths
    assert "/api/detect-path" in paths


def test_detect_does_not_require_api_configuration(patched_detect):
    """位置检测是纯本地推理：没配置 API Key 也必须能用。"""
    patched_detect()

    response = client.post(
        "/api/detect", headers=HEADERS, files={"file": ("local.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200


def test_detect_reports_other_classes_from_core(patched_detect):
    """非鸟类别要在结果里如实上报，前端才能解释「为什么没报鸟」。"""
    patched_detect(other_classes=["bus", "person"])

    response = client.post(
        "/api/detect", headers=HEADERS, files={"file": ("mixed.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json()["result"]["other_classes"] == ["bus", "person"]
