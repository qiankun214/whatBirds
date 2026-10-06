"""配置层：**只**存放火山方舟（Ark）API 接入信息，与项目其他数据完全隔离。

设计要点（对应 Agent.md 第 7 条）：

1. 真实配置写在用户主目录 ``~/.whatbird/config.json``，物理上不在仓库内，
   因此不可能被提交到 git；仓库里只放 ``examples/config.example.json`` 模板。
2. 配置内容按 ``ark`` 命名空间单独存放，不与任何业务数据混在一起。
3. 保存时做二次防御：如果目标路径落在本仓库目录内，直接拒绝写入并报错，
   避免误把密钥写进项目里。
4. 保存后尽力收紧文件权限（POSIX 下 0600），Windows 下退化为普通写权限。
5. 「一次配置后续均有效」：配置是持久化文件，所有入口（CLI / Web）默认读取同一份。

测试或多环境场景可用环境变量覆盖位置（只覆盖路径，不承载密钥）：

- ``WHATBIRD_CONFIG_DIR``   指定配置目录
- ``WHATBIRD_CONFIG_FILE``  直接指定配置文件
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError

#: 方舟 OpenAI 兼容端点（华北-北京）
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

#: Doubao-Seed-2.1 系列（控制台里也可填推理接入点 ID，形如 ep-xxxxxxxx）
DEFAULT_MODEL = "doubao-seed-2-1-lite"

CONFIG_DIR_ENV = "WHATBIRD_CONFIG_DIR"
CONFIG_FILE_ENV = "WHATBIRD_CONFIG_FILE"

#: 本仓库根目录：用于阻止把密钥写进仓库
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_SECTION = "ark"
CONFIG_VERSION = 1


def default_config_dir() -> Path:
    """配置目录（默认 ``~/.whatbird``）。"""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".whatbird"


def default_config_file() -> Path:
    """配置文件路径（默认 ``~/.whatbird/config.json``）。"""
    override = os.environ.get(CONFIG_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return default_config_dir() / "config.json"


def mask_secret(value: str, *, keep: int = 4) -> str:
    """把密钥打码成 ``abcd****wxyz``，用于展示与日志。"""
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 8}{value[-keep:]}"


def _is_inside_project(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
        root = PROJECT_ROOT.resolve()
    except OSError:  # pragma: no cover - 极少数路径解析失败的情况
        return False
    return resolved == root or root in resolved.parents


@dataclass(slots=True)
class ArkConfig:
    """火山方舟接入配置。字段含义见 ``README.md`` 与 ``config.example.json``。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 120.0
    max_tokens: int = 2048
    temperature: float = 0.1
    max_image_side: int = 1568
    jpeg_quality: int = 88

    # ---------------------------------------------------------------- 状态
    @property
    def configured(self) -> bool:
        """是否已填写 API Key。"""
        return bool(self.api_key.strip())

    # ---------------------------------------------------------------- 校验
    def validate(self) -> "ArkConfig":
        """校验配置；不合法时抛出 :class:`ConfigError`。返回自身便于链式调用。"""
        if not self.api_key.strip():
            raise ConfigError(
                "尚未配置火山方舟 API Key",
                hint="运行 `whatbird config set --api-key <你的Key>`，或在网页端配置面板填写",
            )
        if not self.model.strip():
            raise ConfigError("模型（或推理接入点 ID）不能为空", hint="例如 doubao-seed-2-1-lite 或 ep-xxxxxxxx")
        if not self.base_url.strip().startswith(("http://", "https://")):
            raise ConfigError(
                f"base_url 不是合法的 HTTP(S) 地址：{self.base_url!r}",
                hint=f"方舟默认端点为 {DEFAULT_BASE_URL}",
            )
        if self.timeout <= 0:
            raise ConfigError(f"timeout 必须为正数，当前为 {self.timeout!r}")
        if self.max_tokens <= 0:
            raise ConfigError(f"max_tokens 必须为正整数，当前为 {self.max_tokens!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature 需在 0~2 之间，当前为 {self.temperature!r}")
        if self.max_image_side < 64:
            raise ConfigError(f"max_image_side 不能小于 64，当前为 {self.max_image_side!r}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError(f"jpeg_quality 需在 1~100 之间，当前为 {self.jpeg_quality!r}")
        return self

    # ---------------------------------------------------------------- 序列化
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArkConfig":
        """从 ``ark`` 段的字典构造配置，忽略未知字段（向前兼容）。"""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known and v is not None}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """完整字典（含明文密钥）——仅用于写入用户目录的配置文件。"""
        return asdict(self)

    def masked_dict(self) -> dict[str, Any]:
        """脱敏字典——用于 API 返回、CLI 展示与日志。"""
        data = asdict(self)
        data["api_key"] = mask_secret(self.api_key)
        data["configured"] = self.configured
        return data

    def replace(self, **changes: Any) -> "ArkConfig":
        """返回替换了部分字段的新配置对象（``None`` 表示保留原值）。"""
        current = asdict(self)
        for key, value in changes.items():
            if key not in current:
                raise ConfigError(f"未知配置项：{key}")
            if value is not None:
                current[key] = value
        return ArkConfig(**current)


# -------------------------------------------------------------------- 读写
def load_config(path: str | os.PathLike[str] | None = None) -> ArkConfig:
    """读取配置；文件不存在时返回带默认值的空配置（不报错）。"""
    config_path = Path(path) if path is not None else default_config_file()
    if not config_path.exists():
        return ArkConfig()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"配置文件不是合法 JSON：{config_path}",
            hint=f"可以删除该文件后重新配置（原始错误：{exc}）",
        ) from exc
    except OSError as exc:
        raise ConfigError(f"读取配置失败：{config_path}", hint=str(exc)) from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是 JSON 对象：{config_path}")
    return ArkConfig.from_dict(raw.get(CONFIG_SECTION))


def save_config(config: ArkConfig, path: str | os.PathLike[str] | None = None) -> Path:
    """校验并保存配置，返回写入的路径。

    会拒绝把配置写进本仓库目录，避免密钥被 git 追踪。
    """
    config.validate()
    config_path = Path(path) if path is not None else default_config_file()

    if _is_inside_project(config_path):
        raise ConfigError(
            f"拒绝把 API 配置写入项目仓库内：{config_path}",
            hint="配置必须与项目隔离；默认位置为 ~/.whatbird/config.json（可用 WHATBIRD_CONFIG_FILE 覆盖到仓库外）",
        )

    payload = {
        "$schema-note": "whatbird 专用配置：仅存放火山方舟接入信息，请勿提交到 git",
        "version": CONFIG_VERSION,
        CONFIG_SECTION: config.to_dict(),
    }
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ConfigError(f"写入配置失败：{config_path}", hint=str(exc)) from exc

    _harden_permissions(config_path)
    return config_path


def _harden_permissions(path: Path) -> None:
    """尽力把配置文件权限收紧为「仅本人可读写」；失败不影响功能。"""
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows 上通常无效，忽略
        pass


def config_exists(path: str | os.PathLike[str] | None = None) -> bool:
    """配置文件是否已存在。"""
    return (Path(path) if path is not None else default_config_file()).exists()
