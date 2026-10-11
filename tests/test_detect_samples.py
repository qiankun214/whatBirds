"""真实鸟片的端到端验证（可选）。

合成图能验证"业务规则对不对"，但**验证不了"模型在真实鸟片上框得准不准"** ——
后者必须用真实照片。这里把 ``samples/`` 里的图片自动发现出来逐张检测。

两种情况下整组 skip（不会让测试变红）：

1. 没装可选依赖（ultralytics / torch）或权重不在本地；
2. ``samples/`` 里没有图片。

测试不做"框一定落在鸟身上"的硬断言（那需要人工标注，属于主观判断），而是守住机械性质：
**每张真实照片都能跑通、框在画面内、置信度在 0~1**，并把结果打印出来供人工核对。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whatbird.core import DetectorDependencyError, DetectParams, detect_birds
from whatbird.core.detect.models import DEFAULT_MODEL

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
SUFFIXES = (".jpg", ".jpeg")

_DEFAULT_MODEL_CACHE: dict[str, Path | None] = {}


def sample_images() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )


def _local_default_model() -> Path | None:
    """默认权重是否已经在本地（有就不再联网，避免测试卡在下载）。"""
    if "path" not in _DEFAULT_MODEL_CACHE:
        from whatbird.core.detect.backend import resolve_model_path

        candidate = Path(resolve_model_path(DetectParams(model_name=DEFAULT_MODEL)))
        _DEFAULT_MODEL_CACHE["path"] = candidate if candidate.is_file() else None
    return _DEFAULT_MODEL_CACHE["path"]


@pytest.fixture(scope="module")
def detector_params() -> DetectParams:
    weights = _local_default_model()
    if weights is None:
        pytest.skip(
            f"未找到本地权重 {DEFAULT_MODEL}（默认目录 ~/.whatbird/models）。"
            "真实鸟片验证需要权重：手动下载该文件放进权重目录，或本组测试跳过即可。"
        )
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        pytest.skip(
            f"未安装检测可选依赖（{exc.name}）。安装方式见 requirements-detect.txt。"
        )
    return DetectParams(model_name=str(weights))


@pytest.fixture(scope="module")
def images() -> list[Path]:
    found = sample_images()
    if not found:
        pytest.skip(f"samples/ 里还没有 jpg 样张（见 {(SAMPLES_DIR / 'README.md')}）")
    return found


def test_samples_directory_is_discovered(images):
    """每个发现的样张都必须能被 Pillow 打开（先排掉损坏文件，免得后面误判为检测失败）。"""
    from PIL import Image

    for path in images:
        with Image.open(path) as image:
            image.verify()


@pytest.mark.parametrize("label", ["auto"])
def test_real_photo_produces_boxes_inside_the_image(images, detector_params, label):
    """真实照片必须能跑通，且框/置信度落在合法范围内（label 只为让报告里区分运行场景）。"""
    images_without_output = []

    for path in images:
        report = detect_birds(path, params=detector_params)

        # 机械性质：跑通 + 坐标合法 + 置信度范围正确
        assert report.width > 0 and report.height > 0
        assert report.device in {"xpu", "cpu"}, f"上报的设备名异常：{report.device}"
        for bird in report.birds:
            assert 0 <= bird.confidence <= 1
            assert bird.x >= 0 and bird.y >= 0
            assert bird.width >= 1 and bird.height >= 1
            assert bird.x + bird.width <= report.width
            assert bird.y + bird.height <= report.height
            assert 0.0 <= bird.x_ratio <= 1.0 and 0.0 <= bird.y_ratio <= 1.0

        print(
            f"\n[真实样张] {path.name}  设备={report.device}  耗时={report.elapsed_seconds:.2f}s  "
            f"检测到 {report.count} 只鸟"
            + "".join(
                f"\n    #{index} 框 x={bird.x} y={bird.y} w={bird.width} h={bird.height} "
                f"置信度 {bird.confidence:.0%}"
                for index, bird in enumerate(report.birds, start=1)
            )
        )
        if not report.birds:
            images_without_output.append(path.name)

    if images_without_output:
        # 不 fail：有些样张可能本来就没有鸟，或鸟太小导致漏检。
        # 打印出来是为了让人知道该调 --conf / --imgsz，而不是误以为代码坏了。
        print(f"\n[提示] 以下样张未检测到鸟，可试 --conf 0.15 / --imgsz 960：{images_without_output}")


def test_samples_work_with_cpu_device(images, detector_params):
    """真实照片在 CPU 上也要能跑（没有 Intel 显卡的机器不会因此不可用）。"""
    path = images[0]
    report = detect_birds(path, params=DetectParams(**{**detector_params.to_dict(), "device": "cpu"}))
    assert report.device == "cpu"
    assert report.width > 0


def test_bird_class_rule_holds_on_real_photos(images, detector_params):
    """真实照片上也要守住"birds 只装鸟"：结果里不允许出现非鸟类别。"""
    for path in images:
        report = detect_birds(path, params=detector_params)
        for bird in report.birds:
            assert bird.class_name.lower() == "bird"
            assert bird.class_id == 14


def test_dependency_error_is_reported_not_crashed(images):
    """没装依赖时，真实照片路径下也必须给中文提示而不是裸 ImportError。"""
    path = images[0]
    try:
        report = detect_birds(path)
    except DetectorDependencyError as exc:
        assert exc.hint, "缺少依赖时必须给出安装提示"
    else:
        assert report.width > 0  # 装了依赖就直接跑通
