"""REST 层测试：用假识别器替换 core，验证接口契约与错误映射。"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from whatbird.core import ArkConfig, BirdIdentification, ModelCallError
from whatbird.core.image_input import prepare_image_bytes

# 用 importlib 明确取子模块，避免包属性与子模块同名的歧义
web_app = importlib.import_module("whatbird.web.app")

HEADERS = {"X-WhatBird-Client": "1"}

FAKE_RESULT = {
    "is_bird": True,
    "confidence": 0.91,
    "chinese_name": "普通翠鸟",
    "english_name": "Common Kingfisher",
    "scientific_name": "Alcedo atthis",
    "taxonomy": {
        "kingdom": "动物界 Animalia",
        "phylum": "脊索动物门 Chordata",
        "class": "鸟纲 Aves",
        "order": "佛法僧目 Coraciiformes",
        "family": "翠鸟科 Alcedinidae",
        "genus": "翠鸟属 Alcedo",
        "species": "普通翠鸟 Alcedo atthis",
    },
    "identify_basis": ["背部亮蓝色", "喙长而尖"],
    "similar_species": [],
    "habitat": "溪流、池塘",
    "distribution": "欧亚非广布",
    "protection_status": "中国三有保护动物",
    "notes": "",
}


class FakeRecognizer:
    """替身：仍然走真实的图片校验，但跳过网络调用。"""

    fail_with: Exception | None = None

    def __init__(self, config=None, **kwargs):
        self.config = config or ArkConfig(api_key="sk-fake")

    def _build(self, source: str) -> BirdIdentification:
        if type(self).fail_with is not None:
            raise type(self).fail_with
        return BirdIdentification.from_model_output(
            FAKE_RESULT, source_image=source, model="fake-model", elapsed_seconds=0.5
        )

    def recognize(self, image_path: str) -> BirdIdentification:
        from whatbird.core.image_input import prepare_image

        prepare_image(image_path)  # 复用真实校验逻辑
        return self._build(image_path)

    def recognize_bytes(self, data: bytes, *, source: str = "<upload>", **kwargs) -> BirdIdentification:
        prepare_image_bytes(data, source=source)
        return self._build(source)

    def verify_connection(self) -> str:
        return "正常"


@pytest.fixture
def client(monkeypatch):
    FakeRecognizer.fail_with = None
    monkeypatch.setattr(web_app, "BirdRecognizer", FakeRecognizer)
    with TestClient(web_app.app) as test_client:
        yield test_client


# ------------------------------------------------------------------ 系统
def test_health_ok(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["configured"] is False  # 测试夹具把配置指向空目录


def test_json_responses_declare_utf8(client):
    """Windows PowerShell 5.1 等客户端依赖 charset 才能正确解码中文。"""
    response = client.get("/api/health")
    assert "charset=utf-8" in response.headers["content-type"].lower()


# ------------------------------------------------------------------ 配置
def test_get_config_is_masked(client):
    body = client.get("/api/config").json()
    assert body["api_key_masked"] is True
    assert body["config"]["configured"] is False
    assert body["config_file"]


def test_put_config_requires_client_header(client):
    assert client.put("/api/config", json={"api_key": "sk-x"}).status_code == 403


def test_put_and_reload_config(client):
    response = client.put("/api/config", json={"api_key": "sk-live-123456", "model": "ep-abc"}, headers=HEADERS)
    assert response.status_code == 200

    config = client.get("/api/config").json()["config"]
    assert config["configured"] is True
    assert config["model"] == "ep-abc"
    assert "sk-live" not in config["api_key"]  # 只回传脱敏串


def test_put_masked_key_does_not_overwrite(client):
    client.put("/api/config", json={"api_key": "sk-live-123456"}, headers=HEADERS)
    masked = client.get("/api/config").json()["config"]["api_key"]

    client.put("/api/config", json={"api_key": masked, "model": "ep-new"}, headers=HEADERS)

    from whatbird.core import load_config

    assert load_config().api_key == "sk-live-123456"
    assert load_config().model == "ep-new"


def test_put_config_rejects_invalid_values(client):
    response = client.put("/api/config", json={"api_key": "sk-x", "jpeg_quality": 999}, headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "ConfigError"


# ------------------------------------------------------------------ 识别
def test_recognize_path(client, jpeg_factory):
    path = str(jpeg_factory())
    response = client.post("/api/recognize-path", json={"path": path}, headers=HEADERS)

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["names"]["chinese"] == "普通翠鸟"
    assert result["taxonomy"]["family"] == "翠鸟科 Alcedinidae"
    assert result["taxonomy"]["genus"] == "翠鸟属 Alcedo"


def test_recognize_path_requires_header(client, jpeg_factory):
    response = client.post("/api/recognize-path", json={"path": str(jpeg_factory())})
    assert response.status_code == 403


def test_recognize_upload(client, jpeg_factory):
    data = jpeg_factory().read_bytes()
    response = client.post(
        "/api/recognize",
        files={"file": ("bird.jpg", data, "image/jpeg")},
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["result"]["meta"]["source_image"] == "bird.jpg"


def test_recognize_upload_rejects_bad_image(client):
    response = client.post(
        "/api/recognize",
        files={"file": ("fake.jpg", b"not an image", "image/jpeg")},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "ImageInputError"


def test_missing_file_maps_to_400(client):
    response = client.post("/api/recognize-path", json={"path": "E:/nope/bird.jpg"}, headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "ImageInputError"


def test_model_failure_maps_to_502(client, jpeg_factory):
    FakeRecognizer.fail_with = ModelCallError("方舟服务异常", hint="稍后重试")
    response = client.post("/api/recognize-path", json={"path": str(jpeg_factory())}, headers=HEADERS)
    assert response.status_code == 502
    assert response.json()["error"]["hint"] == "稍后重试"


# ------------------------------------------------------------------ 静态前端
def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "WhatBird" in response.text


def test_static_assets_are_served(client):
    for asset in ("/static/style.css", "/static/app.js"):
        assert client.get(asset).status_code == 200
