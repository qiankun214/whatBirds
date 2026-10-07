"""配置层测试：隔离性、脱敏、校验与读写。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from whatbird.core import ArkConfig, ConfigError, load_config, save_config
from whatbird.core.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PROJECT_ROOT,
    ConfigLocation,
    config_file_status,
    config_locations,
    default_config_file,
    git_ignores,
    mask_secret,
    probe_location,
    resolve_config_location,
    write_failure_hint,
)


def _clear_location_env(monkeypatch):
    monkeypatch.delenv("WHATBIRD_CONFIG_FILE", raising=False)
    monkeypatch.delenv("WHATBIRD_CONFIG_DIR", raising=False)


def test_all_config_candidates_stay_out_of_git(monkeypatch):
    """配置位置要么在仓库之外，要么在仓库内但被 git 忽略 —— Agent.md 第 7 条。"""
    _clear_location_env(monkeypatch)

    locations = config_locations()
    assert len(locations) >= 2

    outside = [item for item in locations if not item.inside_project]
    assert outside, "必须至少有一个仓库外的候选位置"
    for item in outside:
        assert PROJECT_ROOT not in item.path.parents
        assert item.path != PROJECT_ROOT

    inside = [item for item in locations if item.inside_project]
    assert len(inside) == 1, "项目内兜底位置只能有一个"
    assert git_ignores(inside[0].path), "项目内兜底位置必须被 .gitignore 覆盖"


def test_default_config_file_follows_env_override():
    """测试夹具用 WHATBIRD_CONFIG_FILE 重定向，必须严格生效且不做降级。"""
    assert default_config_file() == Path(os.environ["WHATBIRD_CONFIG_FILE"])


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


# ------------------------------------------------------- 位置探测与自动降级
def test_probe_reports_unwritable_when_parent_is_a_file(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("我是文件，不是目录", encoding="utf-8")

    probe = probe_location(ConfigLocation(blocker / "sub" / "config.json", "user", "坏位置"))

    assert probe["writable"] is False
    assert probe["error"]


def test_probe_reports_writable_for_normal_dir(tmp_path):
    probe = probe_location(ConfigLocation(tmp_path / "ok" / "config.json", "localappdata", "好位置"))
    assert probe["writable"] is True
    assert probe["error"] is None


def test_resolution_falls_back_to_first_writable_candidate(tmp_path, monkeypatch):
    """首选位置不可写时必须自动降级 —— 用户遇到的正是这个场景。"""
    import whatbird.core.config as config_module

    _clear_location_env(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    bad = ConfigLocation(blocker / "sub" / "config.json", "user", "首选（不可写）")
    good = ConfigLocation(tmp_path / "ok" / "config.json", "localappdata", "备用（可写）")
    monkeypatch.setattr(config_module, "config_locations", lambda: [bad, good])
    monkeypatch.setattr(config_module, "_resolved_location", None)

    assert resolve_config_location(refresh=True).path == good.path


def test_status_explains_why_it_fell_back(tmp_path, monkeypatch):
    import whatbird.core.config as config_module

    _clear_location_env(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    bad = ConfigLocation(blocker / "sub" / "config.json", "user", "首选（不可写）")
    good = ConfigLocation(tmp_path / "ok" / "config.json", "localappdata", "备用（可写）")
    monkeypatch.setattr(config_module, "config_locations", lambda: [bad, good])
    monkeypatch.setattr(config_module, "_resolved_location", None)

    status = config_file_status()

    assert status["path"] == str(good.path)
    assert status["writable"] is True
    assert status["is_fallback"] is True
    assert "首选（不可写）" in status["warning"]
    assert len(status["candidates"]) == 2


def test_status_marks_everything_unwritable(tmp_path, monkeypatch):
    import whatbird.core.config as config_module

    _clear_location_env(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    bad = ConfigLocation(blocker / "sub" / "config.json", "user", "唯一（不可写）")
    monkeypatch.setattr(config_module, "config_locations", lambda: [bad])
    monkeypatch.setattr(config_module, "_resolved_location", None)

    status = config_file_status()

    assert status["writable"] is False
    assert status["error"]


def test_status_reports_why_the_preferred_location_failed(tmp_path, monkeypatch):
    """界面要能回答「为什么降级了」，所以首选位置的失败原因必须带出来。"""
    import whatbird.core.config as config_module

    _clear_location_env(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    bad = ConfigLocation(blocker / "sub" / "config.json", "user", "首选（不可写）")
    good = ConfigLocation(tmp_path / "ok" / "config.json", "localappdata", "备用（可写）")
    monkeypatch.setattr(config_module, "config_locations", lambda: [bad, good])
    monkeypatch.setattr(config_module, "_resolved_location", None)

    status = config_file_status()

    assert status["preferred_error"]
    # OSError 的字符串里路径是转义过的，这里只校验能指认出错的位置
    assert "blocker" in status["preferred_error"]


def test_load_config_survives_location_change(tmp_path, monkeypatch):
    """位置变化（例如从项目内兜底恢复到用户目录）不能让已配置内容凭空消失。"""
    import whatbird.core.config as config_module

    _clear_location_env(monkeypatch)

    legacy = tmp_path / "legacy" / "config.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"ark": {"api_key": "sk-legacy"}}), encoding="utf-8")

    new_location = ConfigLocation(tmp_path / "new" / "config.json", "user", "新位置（可写）")
    old_location = ConfigLocation(legacy, "localappdata", "旧位置")
    monkeypatch.setattr(config_module, "config_locations", lambda: [new_location, old_location])
    monkeypatch.setattr(config_module, "_resolved_location", None)

    loaded = load_config()

    assert loaded.api_key == "sk-legacy"
    assert config_file_status()["loaded_from"] == str(legacy)


def test_explicit_path_never_falls_back(tmp_path, monkeypatch):
    """显式指定路径时必须严格用它，不能偷偷去读别处的配置。"""
    _clear_location_env(monkeypatch)
    other = tmp_path / "other" / "config.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(json.dumps({"ark": {"api_key": "sk-other"}}), encoding="utf-8")

    missing = tmp_path / "not-here" / "config.json"

    assert load_config(missing).api_key == ""


def test_project_fallback_requires_git_ignore():
    """仓库内但未被 git 忽略的位置必须判定为不可用，否则密钥可能被提交。"""
    not_ignored = ConfigLocation(
        PROJECT_ROOT / "whatbird" / "config.json", "project", "仓库内未忽略", inside_project=True
    )

    probe = probe_location(not_ignored)

    assert probe["writable"] is False
    assert probe["git_ignored"] is False
    assert "git" in (probe["error"] or "")


def test_declared_project_fallback_is_actually_ignored(monkeypatch):
    """兜底位置确实被 .gitignore 覆盖（与上一条互为对照）。"""
    _clear_location_env(monkeypatch)
    fallback = next(item for item in config_locations() if item.inside_project)
    assert git_ignores(fallback.path) is True


# ------------------------------------------------------------------ 失败提示
def test_save_failure_hint_is_actionable(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        save_config(ArkConfig(api_key="sk-x"), blocker / "sub" / "config.json")

    hint = excinfo.value.hint
    assert "普通终端" in hint
    assert "WHATBIRD_CONFIG_FILE" in hint
    assert "config doctor" in hint


def test_write_failure_hint_mentions_the_directory(tmp_path):
    hint = write_failure_hint(tmp_path / "a" / "config.json", "PermissionError: 拒绝访问")
    assert str(tmp_path / "a") in hint
