"""配置层测试：隔离性、脱敏、校验与读写。"""

from __future__ import annotations

import json

import pytest

from whatbird.core import ArkConfig, ConfigError, load_config, save_config
from whatbird.core.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    default_config_file,
    mask_secret,
)


def test_default_config_file_lives_outside_project():
    """配置默认位置必须在仓库之外 —— 这是「不可能被提交」的根本保证。"""
    path = default_config_file()
    assert PROJECT_ROOT not in path.parents
    assert path != PROJECT_ROOT


def test_missing_config_returns_defaults():
    config = load_config()
    assert config.configured is False
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL


def test_save_then_load_roundtrip(tmp_path):
    target = tmp_path / "cfg" / "config.json"
    original = ArkConfig(api_key="sk-test-123456", model="ep-20250101-abcdef", temperature=0.3)

    written = save_config(original, target)

    assert written == target
    loaded = load_config(target)
    assert loaded.api_key == "sk-test-123456"
    assert loaded.model == "ep-20250101-abcdef"
    assert loaded.temperature == 0.3


def test_saved_file_is_isolated_in_ark_section(tmp_path):
    """配置内容只落在 ark 命名空间下，与其他信息隔离。"""
    target = tmp_path / "config.json"
    save_config(ArkConfig(api_key="sk-abc"), target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {"$schema-note", "version", "ark"}
    assert payload["ark"]["api_key"] == "sk-abc"


def test_save_refuses_path_inside_project():
    """严禁把密钥写进仓库目录。"""
    inside = PROJECT_ROOT / "config" / "settings.json"
    with pytest.raises(ConfigError) as excinfo:
        save_config(ArkConfig(api_key="sk-abc"), inside)
    assert "拒绝" in excinfo.value.message
    assert not inside.exists()


def test_validate_requires_api_key():
    with pytest.raises(ConfigError):
        ArkConfig(api_key="   ").validate()


def test_validate_rejects_bad_base_url():
    with pytest.raises(ConfigError):
        ArkConfig(api_key="sk-abc", base_url="ark.volces.com").validate()


def test_validate_rejects_bad_quality():
    with pytest.raises(ConfigError):
        ArkConfig(api_key="sk-abc", jpeg_quality=0).validate()


def test_masked_dict_hides_secret():
    config = ArkConfig(api_key="sk-1234567890abcdef")
    masked = config.masked_dict()
    assert masked["api_key"] != config.api_key
    assert "****" in masked["api_key"]
    assert masked["configured"] is True


def test_mask_secret_edge_cases():
    assert mask_secret("") == ""
    assert mask_secret("short") == "*****"
    assert mask_secret("sk-1234567890").endswith("7890")


def test_replace_keeps_other_fields():
    config = ArkConfig(api_key="sk-1", model="m-1")
    updated = config.replace(model="m-2", timeout=None)
    assert updated.model == "m-2"
    assert updated.timeout == config.timeout
    assert updated.api_key == "sk-1"


def test_replace_rejects_unknown_field():
    with pytest.raises(ConfigError):
        ArkConfig(api_key="sk-1").replace(nope="x")


def test_unknown_keys_are_ignored():
    """配置文件里出现未知字段（未来版本写的）不应导致崩溃。"""
    config = ArkConfig.from_dict({"api_key": "sk-x", "future_option": 42})
    assert config.api_key == "sk-x"


def test_invalid_json_raises_config_error(tmp_path):
    target = tmp_path / "config.json"
    target.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(target)


def test_saved_config_then_load_from_env(isolated_config):
    """环境变量重定向后，默认读写路径都应生效且持久。"""
    save_config(ArkConfig(api_key="sk-env"))
    assert isolated_config.exists()
    assert load_config().api_key == "sk-env"
