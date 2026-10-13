"""下载 YOLO 权重（国内镜像优先）。

`whatbird detect` 第一次运行会让 ultralytics 自己去 GitHub Releases 下载权重，
而 GitHub 在国内经常连不上（表现是长时间卡住，或报连接被重置）。这个脚本把
"手动下载再放进权重目录"这一步自动化：按顺序尝试若干国内可达的镜像。

用法::

    python tools/fetch_yolo_weight.py                      # yolo11n.pt -> ~/.whatbird/models
    python tools/fetch_yolo_weight.py --model yolo11s.pt   # 换更大的权重
    python tools/fetch_yolo_weight.py --dest .whatbird/models
    python tools/fetch_yolo_weight.py --list                # 只列出会尝试的地址

下好后离线可用::

    python -m whatbird.cli detect D:\\photos\\bird.jpg --model-dir <目录>

只依赖标准库；装了 ``requests`` 就用它（证书链更省心），没装则退回 ``urllib``。
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: ultralytics 官方资源的发布标签（与 ultralytics 自身下载逻辑保持一致）
RELEASE_TAG = "v8.4.0"

#: 官方地址（国内常连不上，但仍先试一次：能直连时它最可靠）
OFFICIAL = "https://github.com/ultralytics/assets/releases/download/{tag}/{name}"

#: 镜像模板：``{url}`` 是官方地址，``{name}`` 是文件名
MIRRORS = (
    "https://ghfast.top/{url}",
    "https://gh-proxy.com/{url}",
    "https://ghproxy.net/{url}",
    "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/{name}",
    "https://www.modelscope.cn/api/v1/models/Ultralytics/YOLO11/repo?Revision=master&FilePath={name}",
)

#: 权重是 torch 的 zip 容器，头两个字节必须是 ``PK``；小于这个大小一定是错误页面
MIN_BYTES = 1_000_000
ZIP_MAGIC = b"PK"

DEFAULT_MODEL = "yolo11n.pt"


def candidate_urls(name: str) -> list[str]:
    """按「官方 → 各镜像」的顺序给出候选地址。"""
    official = OFFICIAL.format(tag=RELEASE_TAG, name=name)
    urls = [official]
    urls.extend(template.format(url=official, name=name) for template in MIRRORS)
    return urls


def _ssl_context() -> ssl.SSLContext:
    """尽量用 certifi 的证书链。

    Windows 自带的证书存储在某些受限环境里读不到（``SEC_E_NO_CREDENTIALS``），
    而 certifi 是随 Python 包分发的 PEM 文件，不依赖系统存储。
    """
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path, *, timeout: float = 60.0) -> int:
    """把 ``url`` 下载到 ``dest``，返回字节数。失败抛异常。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests  # type: ignore[import-not-found]
    except ImportError:
        requests = None  # type: ignore[assignment]

    if requests is not None:
        response = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(1 << 16):
                handle.write(chunk)
        response.close()
    else:
        with urllib.request.urlopen(url, timeout=timeout, context=_ssl_context()) as response:
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)

    return dest.stat().st_size


def looks_like_weight(path: Path) -> bool:
    """下载是否真的是权重（而不是镜像返回的 HTML 错误页）。"""
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == ZIP_MAGIC


def resolve_dest(dest: str | None, name: str) -> Path:
    """默认落到与程序一致的权重目录（``~/.whatbird/models``）。"""
    if dest:
        return Path(dest).expanduser() / name
    return Path.home() / ".whatbird" / "models" / name


def fetch(name: str, dest: Path, *, force: bool = False) -> Path:
    """按候选顺序下载；只有校验通过的才保留。"""
    if dest.is_file() and not force and looks_like_weight(dest):
        print(f"已存在，跳过：{dest}（{dest.stat().st_size / 1024 / 1024:.1f} MB）")
        return dest

    errors: list[str] = []
    for index, url in enumerate(candidate_urls(name), start=1):
        print(f"[{index}/{len(MIRRORS) + 1}] {url}")
        try:
            size = download(url, dest)
        except Exception as exc:  # noqa: BLE001 - 逐个镜像容错，最后统一报告
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            print(f"    失败：{type(exc).__name__}")
            continue

        if looks_like_weight(dest):
            print(f"    成功：{size / 1024 / 1024:.1f} MB -> {dest}")
            return dest

        errors.append(f"{url} -> 内容校验不通过（{size} 字节）")
        print(f"    内容不像权重（{size} 字节），换下一个")
        dest.unlink(missing_ok=True)

    print("\n所有地址都失败了。可以自己下载后放进权重目录：", file=sys.stderr)
    print(f"  {OFFICIAL.format(tag=RELEASE_TAG, name=name)}", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载 YOLO 权重到 WhatBird 权重目录（国内镜像优先）",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"权重文件名（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--dest", default=None, help="权重目录（默认 ~/.whatbird/models）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--list", action="store_true", help="只列出会尝试的地址，不下载")
    args = parser.parse_args(argv)

    if args.list:
        for url in candidate_urls(args.model):
            print(url)
        return 0

    target = resolve_dest(args.dest, args.model)
    fetch(args.model, target, force=args.force)
    print("\n现在可以离线跑：")
    print(f"  python -m whatbird.cli detect <图片> --model-dir {target.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
