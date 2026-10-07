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
- ``WHATBIRD_CONFIG_FILE``  直接指定配置文件（此时不做任何降级，失败就报错）

### 位置降级（本机环境适配）

有些环境（受限沙箱、受限终端、安全软件策略）不允许进程往用户主目录写东西，
表现为 ``PermissionError: [WinError 5] 拒绝访问``，但目录权限其实是正常的。
为了不让「保存配置」直接失败，``config_locations()`` 给出一个候选链，
``resolve_config_location()`` 会**真实探测**每个位置能否建目录 + 写文件，取第一个可用的：

1. ``~/.whatbird/config.json``                用户目录（首选，隔离性最好）
2. ``%LOCALAPPDATA%\\WhatBird\\config.json``   本机应用数据目录
3. ``%APPDATA%\\WhatBird\\config.json``        漫游应用数据目录
4. ``<项目>/.whatbird/config.json``            最后兜底：位于仓库内，**只有在 git 确认已忽略它时才允许使用**

第 4 项是为了兼容「进程只能写工作区」这类受限环境；它是否可用由 ``git check-ignore``
实时判定，因此 Agent.md 第 7 条「不能被提交到 git 上」仍然由 git 本身强制保证。
无论最终用哪个位置，界面与 CLI 都会显示真实路径，并在不是首选位置时给出提示。
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
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


CONFIG_FILE_NAME = "config.json"

#: 项目内兜底目录名（只有在 git 确认忽略时才允许使用）
PROJECT_FALLBACK_DIR = ".whatbird"

_resolved_location: "ConfigLocation | None" = None


@dataclass(frozen=True)
class ConfigLocation:
    """一个候选配置位置。"""

    path: Path
    kind: str  # user / localappdata / appdata / project / explicit
    label: str
    inside_project: bool = False


def default_config_dir() -> Path:
    """首选配置目录（默认 ``~/.whatbird``）。"""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".whatbird"


def _explicit_location() -> "ConfigLocation | None":
    """环境变量显式指定的位置：不做降级，出问题就直接报错。"""
    override_file = os.environ.get(CONFIG_FILE_ENV)
    if override_file:
        return ConfigLocation(Path(override_file).expanduser(), "explicit", f"{CONFIG_FILE_ENV} 指定的位置")
    override_dir = os.environ.get(CONFIG_DIR_ENV)
    if override_dir:
        return ConfigLocation(
            Path(override_dir).expanduser() / CONFIG_FILE_NAME,
            "explicit",
            f"{CONFIG_DIR_ENV} 指定的位置",
        )
    return None


def config_locations() -> list[ConfigLocation]:
    """按优先级列出候选配置位置。"""
    explicit = _explicit_location()
    if explicit is not None:
        return [explicit]

    locations = [
        ConfigLocation(default_config_dir() / CONFIG_FILE_NAME, "user", "用户目录 ~/.whatbird"),
    ]
    for env_name, label in (
        ("LOCALAPPDATA", "本机应用数据目录"),
        ("APPDATA", "漫游应用数据目录"),
    ):
        base = os.environ.get(env_name)
        if base:
            locations.append(
                ConfigLocation(Path(base) / "WhatBird" / CONFIG_FILE_NAME, env_name.lower(), label)
            )
    locations.append(
        ConfigLocation(
            PROJECT_ROOT / PROJECT_FALLBACK_DIR / CONFIG_FILE_NAME,
            "project",
            "项目内兜底位置（已被 git 忽略）",
            inside_project=True,
        )
    )
    return locations


def git_ignores(path: Path) -> bool:
    """git 是否忽略该路径。

    这是项目内兜底位置的硬门槛：只有 git 确认忽略，才敢把密钥放进去，
    从而由 git 本身保证「不会被提交」（Agent.md 第 7 条）。
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def probe_location(location: ConfigLocation) -> dict[str, Any]:
    """真实试一次「建目录 + 写探针文件」，判断该位置能不能用。

    比 ``os.access`` 靠谱：Windows 上 ACL 正常却仍被进程级策略拦截的情况，
    只有真的写一次才能发现。
    """
    directory = location.path.parent
    writable = True
    error: str | None = None

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".whatbird-write-probe-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
        finally:
            with contextlib.suppress(OSError):
                probe.unlink()
    except OSError as exc:
        writable = False
        error = str(exc)

    git_ignored: bool | None = None
    if writable and location.inside_project:
        git_ignored = git_ignores(location.path)
        if not git_ignored:
            writable = False
            error = "位于项目仓库内，但 git 并未忽略该路径；为避免密钥被提交，此位置不可用"

    return {
        "path": str(location.path),
        "kind": location.kind,
        "label": location.label,
        "inside_project": location.inside_project,
        "writable": writable,
        "error": error,
        "git_ignored": git_ignored,
    }


def resolve_config_location(refresh: bool = False) -> ConfigLocation:
    """返回实际使用的配置位置（取第一个可写的候选），进程内缓存。"""
    global _resolved_location

    explicit = _explicit_location()
    if explicit is not None:
        return explicit

    if _resolved_location is not None and not refresh:
        return _resolved_location

    locations = config_locations()
    chosen = locations[0]  # 都不可写时保留首选，便于报出最有意义的错误
    for location in locations:
        if probe_location(location)["writable"]:
            chosen = location
            break

    _resolved_location = chosen
    return chosen


def default_config_file() -> Path:
    """实际使用的配置文件路径（含位置降级）。"""
    return resolve_config_location().path


def config_file_status() -> dict[str, Any]:
    """配置位置的完整状态，供 GUI / CLI / 排障使用。"""
    global _resolved_location

    locations = config_locations()
    probes = [probe_location(location) for location in locations]

    index = next((i for i, probe in enumerate(probes) if probe["writable"]), 0)
    chosen, chosen_probe = locations[index], probes[index]
    if _explicit_location() is None:
        _resolved_location = chosen  # 与读取路径保持一致

    preferred = probes[0]
    is_fallback = Path(preferred["path"]) != Path(chosen_probe["path"])

    warning: str | None = None
    if chosen.kind == "project":
        warning = (
            "配置保存在项目目录内（git 已确认忽略，不会被提交）。"
            "隔离性弱于用户目录，能写用户目录时建议优先用用户目录。"
        )
    elif is_fallback:
        warning = f"首选位置不可写（{preferred['label']}），已自动改用{chosen.label}。"

    return {
        "path": chosen_probe["path"],
        "exists": Path(chosen_probe["path"]).exists(),
        "writable": chosen_probe["writable"],
        "error": chosen_probe["error"],
        "kind": chosen_probe["kind"],
        "label": chosen_probe["label"],
        "is_fallback": is_fallback,
        "warning": warning,
        "preferred_path": preferred["path"],
        "candidates": probes,
    }


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


def write_failure_hint(path: Path, detail: str) -> str:
    """把写入失败的原因翻译成用户能照着做的几步。"""
    return "\n".join(
        [
            f"系统返回：{detail}",
            "可能的原因与处理办法：",
            "  1) 程序运行在受限环境（沙箱 / 受限终端 / 受限权限的终端）中，只允许写特定目录"
            " —— 请在普通终端里重新运行 whatbird serve；",
            f'  2) 目标目录权限不足 —— 可先手动创建再重试：mkdir "{path.parent}"；',
            "  3) 换一个可写位置：set WHATBIRD_CONFIG_FILE=D:\\whatbird\\config.json"
            "（该环境变量只决定位置，不承载密钥）；",
            "  4) 查看每个候选位置的可写情况：whatbird config doctor。",
        ]
    )


def save_config(config: ArkConfig, path: str | os.PathLike[str] | None = None) -> Path:
    """校验并保存配置，返回写入的路径。

    默认位置由 :func:`resolve_config_location` 探测决定（首选用户目录，不可写时自动降级）。
    显式传入路径时直接使用该路径，不做降级。

    默认情况下拒绝把配置写进本仓库目录；唯一的例外是内部探测确认「该路径已被 git 忽略」
    的项目内兜底位置 —— 此时「不会被提交」由 git 自己保证。
    """
    config.validate()

    if path is not None:
        config_path = Path(path)
        allow_project_path = False
    else:
        location = resolve_config_location()
        config_path = location.path
        allow_project_path = location.inside_project

    if _is_inside_project(config_path) and not (allow_project_path and git_ignores(config_path)):
        raise ConfigError(
            f"拒绝把 API 配置写入项目仓库内：{config_path}",
            hint="配置必须与项目隔离；默认位置为 ~/.whatbird/config.json"
            "（可用 WHATBIRD_CONFIG_FILE 指定其他仓库外的位置）",
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
        raise ConfigError(
            f"无法写入配置文件：{config_path}",
            hint=write_failure_hint(config_path, str(exc)),
        ) from exc

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
