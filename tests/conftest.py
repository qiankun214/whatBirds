"""pytest 公共夹具。

关键点：**每个测试都把配置路径重定向到临时目录**，
保证测试永远不会读写用户真实的 ``~/.whatbird/config.json``。

对焦分析相关的两个夹具（``sharp_patch_factory`` / ``exif_factory``）是**合成样本**：
不需要真实相机照片就能验证算法，且完全确定性 —— 不会因为样张换了就时好时坏。
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFilter


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


def _checkerboard(size: int, step: int = 4) -> Image.Image:
    """高频棋盘格：清晰锐利，且在 JPEG 压缩后依然存活。"""
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for y in range(0, size, step):
        for x in range(0, size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                draw.rectangle(
                    [x, y, min(x + step - 1, size - 1), min(y + step - 1, size - 1)],
                    fill=255,
                )
    return image


@pytest.fixture
def sharp_patch_factory(tmp_path):
    """生成「一小块锐利纹理 + 大片模糊背景」的 jpg。

    这是内容分析路线的**真值可控**样本：清晰的那块在哪，对焦点就该在哪。
    ``center`` 用 0~1 的相对位置给出，方便把纹理块挪到画面各个角落。
    """

    def _make(
        name: str = "sharp.jpg",
        size: tuple[int, int] = (480, 360),
        center: tuple[float, float] = (0.5, 0.5),
        patch_size: int = 96,
        blur: float = 6.0,
        quality: int = 95,
    ) -> tuple[object, tuple[int, int]]:
        width, height = size
        # 背景：先造噪声，再重模糊 —— 干净的低频，几乎没有高频细节
        background = (
            Image.effect_noise(size, 45)
            .convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur))
        )
        patch = _checkerboard(patch_size).convert("RGB")

        left = int(width * center[0] - patch_size / 2)
        top = int(height * center[1] - patch_size / 2)
        left = max(0, min(left, width - patch_size))
        top = max(0, min(top, height - patch_size))

        canvas = background.copy()
        canvas.paste(patch, (left, top))

        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, format="JPEG", quality=quality)
        return path, (left + patch_size // 2, top + patch_size // 2)

    return _make


@pytest.fixture
def exif_factory(tmp_path):
    """生成带标准 EXIF 标签的 jpg（SubjectArea / SubjectLocation / Orientation / GPS）。

    只使用 EXIF 标准标签，不涉及厂商 MakerNote —— 所以纯 Pillow 就能造，
    不需要任何真实相机照片。
    """

    def _make(
        name: str = "exif.jpg",
        size: tuple[int, int] = (400, 300),
        subject_area=None,
        subject_location=None,
        pixel_dimension=None,
        orientation=None,
        gps: bool = False,
    ):
        image = Image.new("RGB", size, (120, 140, 120))
        exif = Image.Exif()

        exif_ifd: dict[int, object] = {}
        if subject_area is not None:
            exif_ifd[0x9214] = tuple(subject_area)  # SubjectArea
        if subject_location is not None:
            exif_ifd[0xA214] = tuple(subject_location)  # SubjectLocation（旧标签）
        if pixel_dimension is not None:
            exif_ifd[0xA002] = int(pixel_dimension[0])  # PixelXDimension
            exif_ifd[0xA003] = int(pixel_dimension[1])  # PixelYDimension
        if exif_ifd:
            exif[0x8769] = exif_ifd  # Exif 子 IFD
        if orientation is not None:
            exif[0x0112] = int(orientation)
        if gps:
            exif[0x8825] = {0x0000: bytes([2, 3, 0, 0])}  # GPSVersionID

        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="JPEG", exif=exif)
        return path

    return _make
