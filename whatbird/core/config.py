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
4. ``<项目>/.whatbird/config.json``            仓库内的兜底位置，**必须显式同意才会使用**

第 4 项有三重限制，避免密钥被悄悄放进项目：

- 必须由用户显式同意（GUI 勾选框 / CLI 开关 / ``WHATBIRD_ALLOW_PROJECT_CONFIG=1``），
  默认**不会**自动降级到这里；
- 必须通过 ``git check-ignore`` 校验，「不会被提交到 git」由 git 本身保证；
- 无论是否使用它，界面与 CLI 都会显示真实路径、首选位置为何不可写，
  并且**读取**时仍会去这些位置查找，位置变化不会让已配置的内容凭空消失。
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
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

#: 显式同意把配置放进项目内兜底位置的环境变量
CONSENT_ENV = "WHATBIRD_ALLOW_PROJECT_CONFIG"

#: 位置缓存：键是「是否已获得项目内兜底的同意」
_resolved_cache: dict[bool, "ConfigLocation"] = {}


def project_consent_granted(explicit: bool | None = None) -> bool:
    """是否同意把配置写入项目内兜底位置。

    默认**不**自动降级到项目内 —— 必须由用户显式同意，来源有三：
    1. 调用方直接传入 ``explicit=True``（GUI 勾选框 / CLI 开关）；
    2. 环境变量 ``WHATBIRD_ALLOW_PROJECT_CONFIG=1``；
    3. 其他来源都不可写时，界面/命令行会明确询问。
    """
    if explicit is not None:
        return bool(explicit)
    value = os.environ.get(CONSENT_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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


def resolve_config_location(
    refresh: bool = False,
    *,
    allow_project: bool | None = None,
) -> ConfigLocation:
    """返回实际使用的配置位置（取第一个可写且被允许的候选），按同意状态缓存。

    注意：项目内兜底位置只有在 :func:`project_consent_granted` 为真时才参与写入；
    未获同意时它会退回到首选位置，于是保存会失败并给出可操作的提示，而不是偷偷降级。
    """
    explicit_location = _explicit_location()
    if explicit_location is not None:
        return explicit_location

    consent = project_consent_granted(allow_project)
    if not refresh and consent in _resolved_cache:
        return _resolved_cache[consent]

    locations = config_locations()
    usable = [
        location
        for location in locations
        if not location.inside_project or consent
    ]

    chosen = locations[0]  # 都不可写时保留首选，便于报出最有意义的错误
    for location in usable:
        if probe_location(location)["writable"]:
            chosen = location
            break

    _resolved_cache[consent] = chosen
    return chosen


def default_config_file() -> Path:
    """实际使用的配置文件路径（含位置降级）。"""
    return resolve_config_location().path


def config_read_order() -> list[Path]:
    """读取配置时的查找顺序：解析出的位置优先，其次是其他候选。

    这样即使位置发生变化（例如从项目内兜底位置恢复到用户目录），
    已经配置过的内容也不会「凭空消失」，只会提示保存到新位置。
    """
    resolved = resolve_config_location().path
    others = [location.path for location in config_locations() if location.path != resolved]
    return [resolved, *others]


def existing_config_file() -> Path | None:
    """当前实际能读到的配置文件（按 :func:`config_read_order`）。"""
    for candidate in config_read_order():
        if candidate.exists():
            return candidate
    return None


def interpreter_status() -> dict[str, Any]:
    """当前解释器的信息与「是否位于项目内」的判断。

    受限环境（沙箱 / 安全策略）常常把**项目内的解释器**限制为只能写工作区，
    因此「用户目录不可写」时，换一个项目外的 Python 往往就能解决。
    这里把这个判断做成可展示的诊断信息。
    """
    executable = Path(sys.executable)
    inside_project = _is_inside_project(executable)
    hint: str | None = None
    if inside_project:
        hint = (
            f"当前解释器位于项目内：{executable}\n"
            "受限环境常把项目内解释器限制为只能写工作区。若用户目录不可写，"
            "请改用项目外的 Python（例如系统 Python）来运行 whatbird。"
        )
    return {
        "executable": str(executable),
        "version": sys.version.split()[0],
        "inside_project": inside_project,
        "hint": hint,
    }


def migrate_config(
    *,
    allow_project: bool | None = None,
    remove_old: bool = False,
) -> dict[str, Any]:
    """把现有配置迁到当前解析出的可写位置。

    典型用途：先在受限环境里把配置写进了项目内兜底位置，
    之后换成不受限的解释器运行，用本函数把配置搬回 ``~/.whatbird/config.json``。
    """
    source = existing_config_file()
    if source is None:
        raise ConfigError(
            "没有找到任何现有配置",
            hint="先用 whatbird config set 写入一份配置，再执行迁移",
        )

    config = load_config(source)
    target = resolve_config_location(allow_project=allow_project).path

    if target == source:
        return {
            "migrated": False,
            "from": str(source),
            "path": str(target),
            "removed_old": False,
            "message": "配置已经位于当前使用的位置，无需迁移",
        }

    saved = save_config(config, allow_project=allow_project)

    removed_old = False
    if remove_old and source != saved:
        try:
            source.unlink()
            removed_old = True
        except OSError as exc:
            raise ConfigError(
                f"新位置已写入成功，但删除旧配置失败：{source}",
                hint=f"{exc}；新配置在 {saved}，可手动删除旧文件",
            ) from exc

    return {
        "migrated": True,
        "from": str(source),
        "path": str(saved),
        "removed_old": removed_old,
        "message": "配置已迁移到新位置",
    }


def config_file_status(*, allow_project: bool | None = None) -> dict[str, Any]:
    """配置位置的完整状态，供 GUI / CLI / 排障使用。"""
    locations = config_locations()
    probes = [probe_location(location) for location in locations]
    consent = project_consent_granted(allow_project)

    eligible = [
        (location, probe)
        for location, probe in zip(locations, probes)
        if probe["writable"] and (not location.inside_project or consent)
    ]
    if eligible:
        chosen, chosen_probe = eligible[0]
    else:
        chosen, chosen_probe = locations[0], probes[0]

    if _explicit_location() is None:
        _resolved_cache[consent] = chosen  # 与读取路径保持一致

    preferred = probes[0]
    is_fallback = Path(preferred["path"]) != Path(chosen_probe["path"])

    project_probe = next(
        (probe for location, probe in zip(locations, probes) if location.inside_project),
        None,
    )
    outside_writable = any(
        probe["writable"] for location, probe in zip(locations, probes) if not location.inside_project
    )
    needs_consent = bool(
        not consent and not outside_writable and project_probe and project_probe["writable"]
    )

    warning: str | None = None
    if chosen.kind == "project":
        warning = (
            "配置保存在项目目录内（已获你同意；git 已确认忽略，不会被提交）。"
            "隔离性弱于用户目录，能写用户目录时建议优先用用户目录。"
        )
    elif is_fallback:
        warning = f"首选位置不可写（{preferred['label']}），已改用{chosen.label}。"

    return {
        "path": chosen_probe["path"],
        "exists": Path(chosen_probe["path"]).exists(),
        "loaded_from": str(existing_config_file()) if existing_config_file() else None,
        "writable": chosen_probe["writable"],
        "error": chosen_probe["error"],
        "kind": chosen_probe["kind"],
        "label": chosen_probe["label"],
        "is_fallback": is_fallback,
        "needs_consent": needs_consent,
        "project_candidate": project_probe,
        "warning": warning,
        "preferred_path": preferred["path"],
        "preferred_error": preferred["error"] if not preferred["writable"] else None,
        "interpreter": interpreter_status(),
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
    """读取配置；文件不存在时返回带默认值的空配置（不报错）。

    未显式指定路径时按 :func:`config_read_order` 查找，
    因此「配置位置发生变化」不会让已经配置好的内容凭空消失。
    """
    if path is not None:
        config_path = Path(path)
    else:
        found = existing_config_file()
        if found is None:
            return ArkConfig()
        config_path = found

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


def save_config(
    config: ArkConfig,
    path: str | os.PathLike[str] | None = None,
    *,
    allow_project: bool | None = None,
) -> Path:
    """校验并保存配置，返回写入的路径。

    默认位置由 :func:`resolve_config_location` 探测决定（首选用户目录，不可写时依次尝试
    其他用户级目录）。**不会**自动把配置放进项目目录 —— 必须显式同意
    （``allow_project=True`` 或环境变量 ``WHATBIRD_ALLOW_PROJECT_CONFIG=1``）。

    显式传入路径时直接使用该路径，不做降级；此时若路径位于仓库内仍会被拒绝，
    除非它同时通过 ``git check-ignore`` 校验。
    """
    config.validate()

    if path is not None:
        config_path = Path(path)
        allow_project_path = False
    else:
        location = resolve_config_location(allow_project=allow_project)
        config_path = location.path
        allow_project_path = location.inside_project

    if _is_inside_project(config_path) and not (allow_project_path and git_ignores(config_path)):
        raise ConfigError(
            f"拒绝把 API 配置写入项目仓库内：{config_path}",
            hint="配置必须与项目隔离；默认位置为 ~/.whatbird/config.json"
            "（可用 WHATBIRD_CONFIG_FILE 指定其他仓库外的位置）。"
            "如果确实要把配置放进项目内兜底位置，请显式同意："
            f"设置 {CONSENT_ENV}=1，或在界面/CLI 上使用对应的同意开关。",
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
        hint = write_failure_hint(config_path, str(exc))
        status = config_file_status()
        if status["needs_consent"]:
            project_path = (status["project_candidate"] or {}).get("path", "")
            hint += (
                f"\n  5) 本机可用的项目内位置（已被 git 忽略，不会被提交）：{project_path}\n"
                f"     若同意把配置放在那里，请显式同意：加 --allow-project-config，"
                f"或设置 {CONSENT_ENV}=1；网页端在第 4 步勾选同意即可。"
            )
        raise ConfigError(f"无法写入配置文件：{config_path}", hint=hint) from exc

    _harden_permissions(config_path)
    return config_path


def _harden_permissions(path: Path) -> None:
    """尽力把配置文件权限收紧为「仅本人可读写」；失败不影响功能。"""
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - Windows 上通常无效，忽略
        pass


def config_exists(path: str | os.PathLike[str] | None = None) -> bool:
    """配置文件是否已存在（未指定路径时按读取顺序查找）。"""
    if path is not None:
        return Path(path).exists()
    return existing_config_file() is not None
