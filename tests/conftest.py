"""pytest 公共夹具。

关键点：**每个测试都把配置路径重定向到临时目录**，
保证测试永远不会读写用户真实的 ``~/.whatbird/config.json``。
"""

from __future__ import annotations

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """把配置重定向到本次测试的临时目录。"""
    config_file = tmp_path / "whatbird-config" / "config.json"
    monkeypatch.setenv("WHATBIRD_CONFIG_FILE", str(config_file))
    return config_file


@pytest.fixture
def jpeg_factory(tmp_path):
    """生成测试用的 jpg 图片，返回 (路径生成函数)。"""

    def _make(name: str = "bird.jpg", size: tuple[int, int] = (80, 60), color=(90, 140, 100)):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, format="JPEG")
        return path

    return _make
