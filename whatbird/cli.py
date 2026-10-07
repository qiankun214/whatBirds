"""CLI 前端：``CLI -> core``。

用法示例::

    python -m whatbird.cli config set --api-key <你的Key> --model doubao-seed-2-1-lite
    python -m whatbird.cli config show
    python -m whatbird.cli identify D:\\photos\\bird.jpg
    python -m whatbird.cli identify D:\\photos\\bird.jpg --json
    python -m whatbird.cli serve --port 8000

本模块只做参数解析与结果展示，所有业务逻辑都在 ``whatbird.core`` 里。
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .core import (
    ArkConfig,
    BirdIdentification,
    ConfigError,
    WhatBirdError,
    config_file_status,
    default_config_file,
    load_config,
    save_config,
)
from .core.config import (
    CONFIG_DIR_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    write_failure_hint,
)
from .core.recognizer import BirdRecognizer


# --------------------------------------------------------------------- 输出
def _print_error(exc: WhatBirdError) -> None:
    print(f"[错误] {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"       提示：{exc.hint}", file=sys.stderr)


def _render_result(result: BirdIdentification) -> str:
    lines: list[str] = []
    rule = "─" * 60

    name = result.chinese_name or "(未能确定中文名)"
    latin = result.scientific_name or "(未能确定学名)"
    lines.append(rule)
    lines.append(f"  {name}  {latin}")
    if result.english_name:
        lines.append(f"  英文名：{result.english_name}")
    lines.append(f"  置信度：{result.confidence:.0%}")
    if not result.is_bird:
        lines.append("  注意：模型判断这张图片中可能不包含鸟类")
    lines.append(rule)

    rows = [(label, value) for _, label, value in result.taxonomy.as_rows() if value]
    if rows:
        lines.append("分类学信息")
        for label, value in rows:
            lines.append(f"  {label}  {value}")

    if result.identify_basis:
        lines.append("")
        lines.append("识别依据")
        for index, item in enumerate(result.identify_basis, start=1):
            lines.append(f"  {index}. {item}")

    if result.similar_species:
        lines.append("")
        lines.append("相似物种")
        for item in result.similar_species:
            if item.difference:
                lines.append(f"  · {item.name}：{item.difference}")
            else:
                lines.append(f"  · {item.name}")

    for label, value in (
        ("栖息环境", result.habitat),
        ("分布范围", result.distribution),
        ("保护级别", result.protection_status),
        ("备注", result.notes),
    ):
        if value:
            lines.append("")
            lines.append(f"{label}：{value}")

    lines.append("")
    lines.append(
        f"元数据：模型 {result.model or '-'} ｜ 图片 {result.source_image or '-'} "
        f"｜ 耗时 {result.elapsed_seconds:.2f}s"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------- 命令
def cmd_config_set(args: argparse.Namespace) -> int:
    config = load_config()

    api_key = args.api_key
    if api_key == "-":
        api_key = getpass.getpass("请输入火山方舟 API Key（输入不回显）：").strip()

    if all(
        value is None
        for value in (
            api_key,
            args.base_url,
            args.model,
            args.timeout,
            args.max_tokens,
            args.temperature,
            args.max_image_side,
            args.jpeg_quality,
        )
    ):
        print("没有提供任何要修改的配置项。用 --help 查看可用参数。", file=sys.stderr)
        return 2

    config = config.replace(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_image_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )
    path = save_config(config)
    status = config_file_status()
    print(f"配置已保存：{path}")
    print(f"  位置：{status['label']}｜{'可写' if status['writable'] else '不可写'}")
    if status["warning"]:
        print(f"  提示：{status['warning']}")
    _print_masked(config)
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    config = load_config()
    status = config_file_status()
    print(f"配置文件：{status['path']}（{'已存在' if status['exists'] else '尚未创建'}）")
    print(f"  位置：{status['label']}｜{'可写' if status['writable'] else '不可写'}")
    if status["warning"]:
        print(f"  提示：{status['warning']}")
    if args.show_key:
        if not config.configured:
            print("API Key：未配置")
        else:
            print(f"API Key：{config.api_key}")
        return 0
    _print_masked(config)
    return 0


def cmd_config_path(_: argparse.Namespace) -> int:
    print(default_config_file())
    return 0


def cmd_config_doctor(_: argparse.Namespace) -> int:
    """排查「配置存不下来」：逐个候选位置真实试写并报告。"""
    status = config_file_status()

    print("配置位置候选（按优先级，逐个真实试写）：")
    for index, candidate in enumerate(status["candidates"], start=1):
        mark = "√" if candidate["writable"] else "×"
        print(f"  {mark} {index}. {candidate['path']}")
        print(f"        {candidate['label']}")
        if candidate["writable"]:
            if candidate["git_ignored"] is not None:
                print("        git 已忽略：是")
        else:
            print(f"        不可写：{candidate['error']}")

    print()
    print(f"最终使用：{status['path']}")
    print(f"  位置：{status['label']}")
    print(f"  已存在：{'是' if status['exists'] else '否'}")
    if status["warning"]:
        print(f"  提示：{status['warning']}")

    if not status["writable"]:
        print()
        print(write_failure_hint(Path(status["path"]), status["error"] or "未知原因"))
    return 0


def cmd_config_verify(_: argparse.Namespace) -> int:
    recognizer = BirdRecognizer(load_config())
    reply = recognizer.verify_connection()
    print(f"连接正常，模型回复：{reply.strip()[:80]}")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    recognizer = BirdRecognizer(load_config())
    result = recognizer.recognize(args.image)

    if args.json:
        print(json.dumps(result.to_dict(include_raw=args.raw), ensure_ascii=False, indent=2))
    else:
        print(_render_result(result))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "[错误] 缺少依赖 uvicorn\n"
            "       提示：python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uvicorn",
            file=sys.stderr,
        )
        return 2

    print(f"WhatBird Web 已启动：http://{args.host}:{args.port}")
    uvicorn.run(
        "whatbird.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def _print_masked(config: ArkConfig) -> None:
    data = config.masked_dict()
    print(f"  API Key      : {data['api_key'] or '（未配置）'}")
    print(f"  base_url     : {data['base_url']}")
    print(f"  model        : {data['model']}")
    print(f"  timeout      : {data['timeout']}s")
    print(f"  max_tokens   : {data['max_tokens']}")
    print(f"  temperature  : {data['temperature']}")
    print(f"  max_image_side / jpeg_quality : {data['max_image_side']} / {data['jpeg_quality']}")
    if not config.configured:
        print("  ⚠ 尚未配置 API Key，识别功能不可用")


# --------------------------------------------------------------------- 解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whatbird",
        description="基于火山引擎方舟 Doubao-Seed-2.1 的鸟类识别工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  whatbird config set --api-key - \n"
            "  whatbird identify D:\\photos\\bird.jpg\n"
            "  whatbird serve --port 8000\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"whatbird {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # config
    config_parser = subparsers.add_parser("config", help="管理火山方舟 API 配置")
    config_sub = config_parser.add_subparsers(dest="config_command")

    set_parser = config_sub.add_parser("set", help="写入/更新配置（持久保存在用户目录）")
    set_parser.add_argument("--api-key", help="方舟 API Key；填 '-' 则安全地交互式输入")
    set_parser.add_argument("--base-url", help=f"默认 {DEFAULT_BASE_URL}")
    set_parser.add_argument("--model", help=f"Model ID 或推理接入点 ID，默认 {DEFAULT_MODEL}")
    set_parser.add_argument("--timeout", type=float, help="单次请求超时（秒）")
    set_parser.add_argument("--max-tokens", type=int, dest="max_tokens")
    set_parser.add_argument("--temperature", type=float)
    set_parser.add_argument("--max-image-side", type=int, dest="max_image_side", help="图片最长边上限")
    set_parser.add_argument("--jpeg-quality", type=int, dest="jpeg_quality", help="重编码 JPEG 质量 1~100")
    set_parser.set_defaults(handler=cmd_config_set)

    show_parser = config_sub.add_parser("show", help="查看当前配置（默认脱敏）")
    show_parser.add_argument("--show-key", action="store_true", help="显示明文 API Key（注意旁人可见）")
    show_parser.set_defaults(handler=cmd_config_show)

    path_parser = config_sub.add_parser("path", help="打印配置文件位置")
    path_parser.set_defaults(handler=cmd_config_path)

    doctor_parser = config_sub.add_parser(
        "doctor",
        help="排查配置写入失败：逐个候选位置试写并给出处理办法",
    )
    doctor_parser.set_defaults(handler=cmd_config_doctor)

    verify_parser = config_sub.add_parser("verify", help="验证配置能否成功调用方舟")
    verify_parser.set_defaults(handler=cmd_config_verify)

    config_parser.set_defaults(handler=lambda _: (config_parser.print_help(), 1)[1])

    # identify
    identify_parser = subparsers.add_parser("identify", help="识别一张 jpg 图片中的鸟")
    identify_parser.add_argument("image", help="jpg / jpeg 文件路径")
    identify_parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本处理）")
    identify_parser.add_argument("--raw", action="store_true", help="JSON 输出中包含模型原始返回")
    identify_parser.set_defaults(handler=cmd_identify)

    # serve
    serve_parser = subparsers.add_parser("serve", help="启动 Web 服务（后端 + 静态前端）")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true", help="代码变更自动重载（开发用）")
    serve_parser.add_argument("--log-level", default="info", dest="log_level")
    serve_parser.set_defaults(handler=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 1

    try:
        return int(handler(args))
    except ConfigError as exc:
        _print_error(exc)
        return 3
    except WhatBirdError as exc:
        _print_error(exc)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
