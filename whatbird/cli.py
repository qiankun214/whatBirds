"""CLI 前端：``CLI -> core``。

用法示例::

    python -m whatbird.cli config set --api-key <你的Key> --model doubao-seed-2-1-lite
    python -m whatbird.cli config show
    python -m whatbird.cli identify D:\\photos\\bird.jpg
    python -m whatbird.cli identify D:\\photos\\bird.jpg --json
    python -m whatbird.cli focus D:\\photos\\bird.jpg --overlay out.png
    python -m whatbird.cli serve --port 8000

本模块只做参数解析与结果展示，所有业务逻辑都在 ``whatbird.core`` 里。
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .core import (
    ArkConfig,
    BatchItem,
    BatchReport,
    BirdIdentification,
    ConfigError,
    DetectParams,
    FocusHit,
    FocusParams,
    FocusReport,
    WhatBirdError,
    analyze_focus,
    config_file_status,
    default_config_file,
    detect_birds,
    interpreter_status,
    load_config,
    migrate_config,
    render_bird_overlay,
    render_focus_overlay,
    save_config,
)
from .core.config import (
    CONFIG_DIR_ENV,
    CONSENT_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    write_failure_hint,
)
from .core.detect.models import BIRD_CLASS_ID, DEFAULT_MODEL as DEFAULT_DETECT_MODEL, DetectionReport
from .core.prompt import MODE_FAST, MODE_FULL
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
    meta_bits = [
        f"模型 {result.model or '-'}",
        f"图片 {result.source_image or '-'}",
        f"耗时 {result.elapsed_seconds:.2f}s",
    ]
    if result.mode == MODE_FAST:
        meta_bits.append("快速模式")
    if result.usage:
        meta_bits.append(
            f"token 输入 {result.usage.get('prompt_tokens', '-')}"
            f" / 输出 {result.usage.get('completion_tokens', '-')}"
        )
    lines.append("元数据：" + " ｜ ".join(meta_bits))
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
    path = save_config(config, allow_project=args.allow_project_config)
    status = config_file_status(allow_project=args.allow_project_config)
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
    loaded_from = status.get("loaded_from")
    if loaded_from and loaded_from != status["path"]:
        print(f"  当前读取自：{loaded_from}（下次保存会写入上面的新位置）")
    if status["warning"]:
        print(f"  提示：{status['warning']}")
    if status["preferred_error"]:
        print(f"  首选位置不可写原因：{status['preferred_error']}")
    if status["needs_consent"]:
        project = status.get("project_candidate") or {}
        print(f"  提示：用户级位置都不可写。若同意把配置放到 {project.get('path')}"
              "（已被 git 忽略、不会提交），请显式同意：")
        print("        whatbird config set --allow-project-config ...")
        print(f'        或设置环境变量 {CONSENT_ENV}=1')
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
    interp = interpreter_status()

    print(f"当前解释器：{interp['executable']}（Python {interp['version']}）")
    print(f"  位于项目内：{'是' if interp['inside_project'] else '否'}")
    print()

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
    loaded_from = status.get("loaded_from")
    if loaded_from and loaded_from != status["path"]:
        print(f"  当前读取自：{loaded_from}")
    if status["warning"]:
        print(f"  提示：{status['warning']}")
    if status["needs_consent"]:
        print("  注意：项目内兜底位置可写，但需要显式同意才会使用"
              f"（--allow-project-config 或 {CONSENT_ENV}=1）")

    if not status["writable"]:
        print()
        print(write_failure_hint(Path(status["path"]), status["error"] or "未知原因"))
        if interp["hint"]:
            print()
            print("解释器提示：")
            for line in interp["hint"].splitlines():
                print(f"  {line}")
    return 0


def cmd_config_migrate(args: argparse.Namespace) -> int:
    """把现有配置迁移到当前可写位置（典型场景：从项目内兜底位置搬到用户目录）。"""
    result = migrate_config(remove_old=args.remove_old)

    if not result["migrated"]:
        print(result["message"])
        print(f"  位置：{result['path']}")
        return 0

    print(f"已迁移：{result['from']}")
    print(f"    ->  {result['path']}")
    if result["removed_old"]:
        print("  旧配置已删除")
    else:
        print(f"  旧配置仍保留：{result['from']}（加 --remove-old 可自动删除）")

    status = config_file_status()
    print(f"  当前使用位置：{status['label']}")
    return 0


def cmd_config_verify(_: argparse.Namespace) -> int:
    recognizer = BirdRecognizer(load_config())
    reply = recognizer.verify_connection()
    print(f"连接正常，模型回复：{reply.strip()[:80]}")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    recognizer = BirdRecognizer(load_config())
    images = list(args.images)
    mode = args.mode

    # ---- 单张：保持原有输出格式，仅追加 token 信息 ----
    if len(images) == 1:
        result = recognizer.recognize(images[0], mode=mode)
        if args.json:
            print(json.dumps(result.to_dict(include_raw=args.raw), ensure_ascii=False, indent=2))
        else:
            print(_render_result(result))
        return 0

    # ---- 多张：依次识别（可并发），边跑边报进度 ----
    return _run_batch(
        recognizer,
        images,
        as_json=args.json,
        mode=mode,
        concurrency=args.concurrency,
    )


def _run_batch(
    recognizer: BirdRecognizer,
    images: list[str],
    *,
    as_json: bool,
    mode: str | None,
    concurrency: int | None,
) -> int:
    report = BatchReport(total=len(images), mode=recognizer.prompts(mode).mode)
    interrupted = False
    started = time.perf_counter()

    def on_start(index: int, total: int, name: str) -> None:
        print(f"[{index}/{total}] 正在识别：{name}", file=sys.stderr, flush=True)

    async def drive() -> None:
        async for item in recognizer.aiter_batch(
            images,
            on_start=on_start,
            mode=mode,
            concurrency=concurrency,
        ):
            report.items.append(item)
            if not as_json:
                print(_render_batch_line(item), flush=True)

    try:
        asyncio.run(drive())
    except KeyboardInterrupt:
        interrupted = True
        print("\n已中止（Ctrl+C）", file=sys.stderr)
    finally:
        report.elapsed_seconds = time.perf_counter() - started
        report.items.sort(key=lambda item: item.index)
        report.stopped = interrupted or report.completed < report.total

    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            f"\n批量结束：共 {report.total} 张，成功 {report.succeeded}，失败 {report.failed}"
            f"，用时 {report.elapsed_seconds:.1f}s"
            + (f"（{report.mode} 模式）" if report.mode == MODE_FAST else "")
            + ("（已中止）" if report.stopped else "")
        )
    return 130 if interrupted else 0


def _render_batch_line(item: BatchItem) -> str:
    """批量模式下一张图一行的进度输出。"""
    if item.ok and item.identification is not None:
        result = item.identification
        name = result.chinese_name or result.scientific_name or "未识别"
        latin = f"（{result.scientific_name}）" if result.scientific_name and result.chinese_name else ""
        family = result.taxonomy.family or "-"
        body = f"{name}{latin}  科：{family}  置信度 {result.confidence:.0%}"
    else:
        body = f"失败：{item.error_message}"

    usage = ""
    if item.ok and item.identification is not None and item.identification.usage:
        tokens = item.identification.usage
        usage = f"  token {tokens.get('prompt_tokens', '-')}/{tokens.get('completion_tokens', '-')}"
    return f"[{item.index}/{item.total}] {body}  | {item.source}  {item.elapsed_seconds:.1f}s{usage}"


# --------------------------------------------------------------------- 对焦点
def _focus_params(args: argparse.Namespace) -> FocusParams:
    return FocusParams(
        max_side=args.max_side,
        metric=args.metric,
        top_n=args.top,
        use_exif=not args.no_exif,
        use_content=not args.no_content,
    )


def _render_hit(title: str, hit: FocusHit, source: str) -> list[str]:
    lines = [title]
    if hit.found:
        lines.append(
            f"  √ 对焦点 ({hit.x}, {hit.y})"
            f"  相对位置 {hit.x_ratio:.1%} / {hit.y_ratio:.1%}"
            f"  置信度 {hit.confidence:.0%}"
        )
        lines.append(f"    来源：{source}")
        lines.append(f"    说明：{hit.detail}")
    else:
        lines.append(f"  × 无结论：{hit.detail}")
    return lines


def _render_focus(report: FocusReport) -> str:
    rule = "─" * 60
    lines = [rule]
    lines.append(f"  对焦点分析  {report.source_image or '-'}")
    lines.append(
        f"  图像 {report.width}×{report.height}"
        f"（分析尺度 {report.analyzed_width}×{report.analyzed_height}，"
        f"缩放系数 {report.scale:.3f}）"
    )
    lines.append(rule)

    exif_source = report.exif.source_tags.get("tag", "") or "EXIF"
    lines.extend(_render_hit("路线一 · EXIF 元数据（相机声称的对焦点）", report.exif, exif_source))
    lines.append("")
    lines.extend(
        _render_hit(
            "路线二 · 内容分析（画面实际最清晰处）",
            report.content,
            "SML 改进拉普拉斯清晰度峰值",
        )
    )

    if report.region is not None:
        region = report.region
        lines.append("")
        lines.append("合焦区域（清晰度超阈值的连通区域）")
        lines.append(
            f"  包围盒  x {region.x}  y {region.y}  {region.width}×{region.height}"
            f"   占画面 {region.area_ratio:.1%}"
        )

    other = [item for item in report.candidates[1:]]
    if other:
        lines.append("")
        lines.append("其他候选点（按相对锐度排序）")
        for index, item in enumerate(other, start=2):
            lines.append(f"  {index}. ({item.x}, {item.y})  相对锐度 {item.score:.0%}")

    lines.append("")
    if report.exif.found and report.content.found:
        lines.append(f"结论：两条路线{'一致（互相印证）' if report.agree else '不一致，值得看一眼'}")
    elif report.available_methods():
        lines.append("结论：只有一条路线给出了结论（见上面的原因）")
    else:
        lines.append("结论：两条路线都没能定位对焦点")

    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"提示：{note}")

    lines.append("")
    lines.append(f"元数据：耗时 {report.elapsed_seconds:.2f}s ｜ 参数 {report.params.get('metric', '-')}")
    return "\n".join(lines)


def _overlay_path_for(args: argparse.Namespace, image: str, index: int) -> Path | None:
    """决定这张图的标注图存哪。"""
    if args.overlay and len(args.images) == 1:
        return Path(args.overlay).expanduser()
    if args.overlay_dir:
        stem = Path(image).stem or f"image{index}"
        return Path(args.overlay_dir).expanduser() / f"{stem}.focus.png"
    return None


def cmd_focus(args: argparse.Namespace) -> int:
    """分析图片的对焦点（EXIF 元数据 + 传统图形学内容分析，两条路线都上报）。"""
    if args.overlay and args.overlay_dir:
        print("[错误] --overlay 与 --overlay-dir 只能给一个", file=sys.stderr)
        return 2
    if args.overlay and len(args.images) > 1:
        print(
            "[错误] 一次给多张图时请用 --overlay-dir（多张图不能写进同一个文件）",
            file=sys.stderr,
        )
        return 2

    params = _focus_params(args)
    overlay_dir = Path(args.overlay_dir).expanduser() if args.overlay_dir else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    reports: list[FocusReport] = []
    for index, image in enumerate(args.images, start=1):
        report = analyze_focus(image, params=params)
        reports.append(report)

        target = _overlay_path_for(args, image, index)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(render_focus_overlay(image, report))

        if not args.json:
            if len(args.images) > 1:
                print(f"[{index}/{len(args.images)}] {report.summary_line()}")
            else:
                print(_render_focus(report))
            if target is not None:
                print(f"标注图已保存：{target}")

    if args.json:
        payload: Any
        if len(reports) == 1:
            payload = reports[0].to_dict()
        else:
            payload = {"total": len(reports), "items": [item.to_dict() for item in reports]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------- 鸟类位置检测
def _detect_params(args: argparse.Namespace) -> DetectParams:
    return DetectParams(
        model_name=args.model,
        model_dir=Path(args.model_dir).expanduser() if args.model_dir else None,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=() if args.all_classes else (BIRD_CLASS_ID,),
    )


def _detect_overlay_path_for(args: argparse.Namespace, image: str, index: int) -> Path | None:
    """决定这张图的标注图存哪（与 ``focus`` 的规则保持一致）。"""
    if args.overlay and len(args.images) == 1:
        return Path(args.overlay).expanduser()
    if args.overlay_dir:
        stem = Path(image).stem or f"image{index}"
        return Path(args.overlay_dir).expanduser() / f"{stem}.detect.png"
    return None


def _render_detections(report: DetectionReport) -> str:
    rule = "─" * 60
    lines = [rule]
    lines.append(f"  鸟类位置检测  {report.source_image or '-'}")
    lines.append(f"  图像 {report.width}×{report.height}")
    lines.append(f"  设备 {report.device or '-'} ｜ 模型 {report.model_name or '-'}")
    lines.append(rule)

    if not report.birds:
        lines.append("  未检测到鸟")
    else:
        lines.append(f"  检测到 {report.count} 只鸟（按置信度降序）")
        for index, bird in enumerate(report.birds, start=1):
            lines.append(
                f"  #{index}  框 x={bird.x} y={bird.y} w={bird.width} h={bird.height}"
                f" ｜ 相对位置 {bird.x_ratio:.1%}/{bird.y_ratio:.1%}"
                f" ｜ 置信度 {bird.confidence:.0%}"
            )

    if report.other_classes:
        lines.append("")
        lines.append("画面里还有（已按参数过滤，不计入结果）：" + "、".join(report.other_classes))

    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"  说明：{note}")

    lines.append("")
    lines.append(f"  耗时 {report.elapsed_seconds:.2f}s")
    return "\n".join(lines)


def cmd_detect(args: argparse.Namespace) -> int:
    """检测图片里每一只鸟的位置（纯本地 YOLO 推理，不调用方舟、不消耗额度）。"""
    if args.overlay and args.overlay_dir:
        print("[错误] --overlay 与 --overlay-dir 只能给一个", file=sys.stderr)
        return 2
    if args.overlay and len(args.images) > 1:
        print(
            "[错误] 一次给多张图时请用 --overlay-dir（多张图不能写进同一个文件）",
            file=sys.stderr,
        )
        return 2

    params = _detect_params(args)
    overlay_dir = Path(args.overlay_dir).expanduser() if args.overlay_dir else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    reports: list[DetectionReport] = []
    for index, image in enumerate(args.images, start=1):
        report = detect_birds(image, params=params)
        reports.append(report)

        target = _detect_overlay_path_for(args, image, index)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(render_bird_overlay(image, report))

        if not args.json:
            if len(args.images) > 1:
                print(f"[{index}/{len(args.images)}] {report.summary_line()}  | {report.source_image}")
            else:
                print(_render_detections(report))
            if target is not None:
                print(f"标注图已保存：{target}")

    if args.json:
        payload: Any
        if len(reports) == 1:
            payload = reports[0].to_dict()
        else:
            payload = {"total": len(reports), "items": [item.to_dict() for item in reports]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
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
            "  whatbird focus D:\\photos\\bird.jpg --overlay out.png\n"
            "  whatbird detect D:\\photos\\bird.jpg --overlay boxes.png\n"
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
    set_parser.add_argument(
        "--allow-project-config",
        action="store_true",
        default=None,
        dest="allow_project_config",
        help="用户目录不可写时，显式同意把配置保存到项目内 .whatbird/（已被 git 忽略）",
    )
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

    migrate_parser = config_sub.add_parser(
        "migrate",
        help="把现有配置迁移到当前可写位置（例如从项目内搬到用户目录）",
    )
    migrate_parser.add_argument(
        "--remove-old",
        action="store_true",
        dest="remove_old",
        help="迁移成功后删除旧配置文件",
    )
    migrate_parser.set_defaults(handler=cmd_config_migrate)

    verify_parser = config_sub.add_parser("verify", help="验证配置能否成功调用方舟")
    verify_parser.set_defaults(handler=cmd_config_verify)

    config_parser.set_defaults(handler=lambda _: (config_parser.print_help(), 1)[1])

    # identify
    identify_parser = subparsers.add_parser(
        "identify",
        help="识别 jpg 图片中的鸟（可一次给多张，按顺序依次识别）",
    )
    identify_parser.add_argument(
        "images",
        nargs="+",
        help="jpg / jpeg 文件路径；给多个则依次识别并显示进度",
    )
    identify_parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本处理）")
    identify_parser.add_argument("--raw", action="store_true", help="JSON 输出中包含模型原始返回（仅单张）")
    identify_parser.add_argument(
        "--fast",
        dest="mode",
        action="store_const",
        const=MODE_FAST,
        help="快速模式：只返回名称与分类学信息（输出短得多，单张明显更快）",
    )
    identify_parser.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const=MODE_FULL,
        help="完整模式（默认；实际取决于配置里的 fast_mode）",
    )
    identify_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="批量并发数 1~8（默认取配置，通常是 3；1 = 完全串行）",
    )
    identify_parser.set_defaults(handler=cmd_identify, mode=None)

    # focus
    focus_parser = subparsers.add_parser(
        "focus",
        help="分析对焦点：EXIF 元数据 + 传统图形学内容分析（两条路线都上报）",
        description=(
            "找出这张照片对焦在哪里。\n"
            "同时跑两条独立路线并都上报：\n"
            "  1) EXIF 路线：读相机写进元数据的对焦点（相机声称的）；\n"
            "  2) 内容路线：用传统图形学算法分析像素清晰度分布（画面实际的）。\n"
            "两条路线都不需要联网、不消耗任何额度。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    focus_parser.add_argument("images", nargs="+", help="jpg / jpeg 文件路径")
    focus_parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本处理）")
    focus_parser.add_argument(
        "--overlay",
        help="把标注图保存到该路径（仅单张图；PNG，原分辨率）",
    )
    focus_parser.add_argument(
        "--overlay-dir",
        dest="overlay_dir",
        help="多张图时把标注图按 <原名>.focus.png 存进该目录",
    )
    focus_parser.add_argument(
        "--metric",
        choices=("sml", "tenengrad"),
        default="sml",
        help="清晰度评价函数：sml = 改进拉普拉斯（默认，对景深最稳），tenengrad = 梯度平方和",
    )
    focus_parser.add_argument(
        "--max-side",
        type=int,
        dest="max_side",
        default=1024,
        help="分析时把图片最长边缩到该值（默认 1024），越大越慢越细",
    )
    focus_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="上报的候选对焦点个数（默认 5）",
    )
    focus_parser.add_argument(
        "--no-exif",
        action="store_true",
        dest="no_exif",
        help="只跑内容分析路线",
    )
    focus_parser.add_argument(
        "--no-content",
        action="store_true",
        dest="no_content",
        help="只跑 EXIF 路线",
    )
    focus_parser.set_defaults(handler=cmd_focus)

    # detect
    detect_parser = subparsers.add_parser(
        "detect",
        help="找出图片里每一只鸟的位置（本地 YOLO 推理，可用 Intel 显卡加速）",
        description=(
            "用本地 YOLO 权重检测画面里每一只鸟的位置（框 + 置信度）。\n"
            "只报鸟（COCO 类别 id 14），其他类别只统计不返回。\n"
            "纯本地推理：不调用方舟、不消耗额度、不需要 API Key；\n"
            "首次运行会自动下载 YOLO 权重（也可用 --model 指向本地权重文件）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    detect_parser.add_argument("images", nargs="+", help="jpg / jpeg 文件路径")
    detect_parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本处理）")
    detect_parser.add_argument(
        "--overlay",
        help="把画框标注图保存到该路径（仅单张图；PNG，原分辨率）",
    )
    detect_parser.add_argument(
        "--overlay-dir",
        dest="overlay_dir",
        help="多张图时把标注图按 <原名>.detect.png 存进该目录",
    )
    detect_parser.add_argument(
        "--model",
        default=DEFAULT_DETECT_MODEL,
        help=f"YOLO 权重名或本地权重路径，默认 {DEFAULT_DETECT_MODEL}",
    )
    detect_parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=None,
        help="权重目录（默认 ~/.whatbird/models）；同名权重已存在就用本地的，不再联网",
    )
    detect_parser.add_argument(
        "--device",
        choices=("auto", "xpu", "cpu"),
        default="auto",
        help="推理设备：auto 优先用 Intel 显卡（xpu），不可用退回 cpu",
    )
    detect_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认 0.25）")
    detect_parser.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值（默认 0.7）")
    detect_parser.add_argument("--imgsz", type=int, default=640, help="推理分辨率（默认 640，越大越慢）")
    detect_parser.add_argument(
        "--all-classes",
        action="store_true",
        dest="all_classes",
        help="不限制类别（默认只检测鸟；加了它仍只把鸟算进结果，其他类别会列出来）",
    )
    detect_parser.set_defaults(handler=cmd_detect)

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
