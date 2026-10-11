"""对焦点分析的 REST 接口测试。

对焦分析是**纯本地算法**（不碰方舟、不消耗额度），所以这里不需要任何假客户端，
直接用真实实现跑合成图片即可。
"""

from __future__ import annotations

import base64
import importlib
import io

from fastapi.testclient import TestClient
from PIL import Image

web_app = importlib.import_module("whatbird.web.app")

HEADERS = {"X-WhatBird-Client": "1"}
client = TestClient(web_app.app)


def _jpg_bytes(size=(480, 360)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 140, 100)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_focus_upload_returns_result_and_overlay(sharp_patch_factory):
    path, want = sharp_patch_factory(name="api.jpg")

    response = client.post(
        "/api/focus",
        headers=HEADERS,
        files={"file": ("api.jpg", path.read_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]

    assert result["content"]["found"] is True
    assert result["image"]["width"] == 480
    assert "exif" in result and "content" in result

    overlay = payload["overlay"]
    assert overlay.startswith("data:image/png;base64,")
    raw = base64.b64decode(overlay.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (480, 360)


def test_focus_upload_requires_client_header(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="csrf.jpg")

    response = client.post(
        "/api/focus",
        files={"file": ("csrf.jpg", path.read_bytes(), "image/jpeg")},
    )

    assert response.status_code == 403


def test_focus_path_endpoint(sharp_patch_factory):
    path, _ = sharp_patch_factory(name="bypath.jpg")

    response = client.post("/api/focus-path", headers=HEADERS, json={"path": str(path)})

    assert response.status_code == 200
    assert response.json()["result"]["content"]["found"] is True


def test_focus_path_missing_file_gives_chinese_hint(tmp_path):
    missing = tmp_path / "nope.jpg"

    response = client.post("/api/focus-path", headers=HEADERS, json={"path": str(missing)})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "ImageInputError"
    assert "不存在" in error["message"]
    assert error["hint"]


def test_focus_path_rejects_non_jpg(tmp_path):
    png = tmp_path / "bird.png"
    Image.new("RGB", (40, 40)).save(png, format="PNG")

    response = client.post("/api/focus-path", headers=HEADERS, json={"path": str(png)})

    assert response.status_code == 400
    assert "暂不支持" in response.json()["error"]["message"]


def test_focus_upload_rejects_oversized_file():
    oversized = b"\xff\xd8" + b"\x00" * (web_app.MAX_UPLOAD_BYTES + 10)

    response = client.post(
        "/api/focus", headers=HEADERS, files={"file": ("huge.jpg", oversized, "image/jpeg")}
    )

    assert response.status_code == 413
    assert "过大" in response.json()["detail"]


def test_focus_upload_rejects_corrupt_file():
    response = client.post(
        "/api/focus",
        headers=HEADERS,
        files={"file": ("broken.jpg", b"definitely not a jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "无法解析" in response.json()["error"]["message"]


def test_focus_image_without_exif_still_succeeds():
    """没有 EXIF 的图也应该正常返回：EXIF 路线给中文说明，内容路线照常工作。"""
    response = client.post(
        "/api/focus", headers=HEADERS, files={"file": ("plain.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["exif"]["found"] is False
    assert "没有可用的 EXIF 对焦信息" in result["exif"]["detail"]


def test_focus_endpoints_are_registered_in_openapi():
    paths = set(web_app.app.openapi()["paths"])
    assert "/api/focus" in paths
    assert "/api/focus-path" in paths


def test_focus_does_not_require_api_configuration():
    """对焦分析是纯本地算法：没配置 API Key 也必须能用。"""
    response = client.post(
        "/api/focus", headers=HEADERS, files={"file": ("local.jpg", _jpg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
