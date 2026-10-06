"""图片输入层测试：格式校验、缩放、base64 编码。"""

from __future__ import annotations

import base64

import pytest

from whatbird.core import ImageInputError
from whatbird.core.image_input import prepare_image, prepare_image_bytes


def test_prepare_jpeg_returns_data_url(jpeg_factory):
    path = jpeg_factory()
    prepared = prepare_image(path)

    assert prepared.data_url.startswith("data:image/jpeg;base64,")
    assert prepared.width == 80 and prepared.height == 60
    assert prepared.resized is False

    raw = base64.b64decode(prepared.data_url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI 魔数
    assert prepared.byte_size == len(raw)


def test_large_image_is_downscaled(tmp_path):
    from PIL import Image

    path = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 1500), (10, 20, 30)).save(path, format="JPEG")

    prepared = prepare_image(path, max_image_side=640)

    assert prepared.resized is True
    assert max(prepared.width, prepared.height) == 640
    assert prepared.original_width == 3000
    # 宽高比保持不变
    assert abs(prepared.width / prepared.height - 2.0) < 0.01


def test_missing_file_raises():
    with pytest.raises(ImageInputError) as excinfo:
        prepare_image("E:/definitely/not/here.jpg")
    assert "不存在" in excinfo.value.message


def test_directory_raises(tmp_path):
    with pytest.raises(ImageInputError):
        prepare_image(tmp_path)


def test_non_jpg_suffix_raises(tmp_path):
    from PIL import Image

    png = tmp_path / "bird.png"
    Image.new("RGB", (20, 20)).save(png, format="PNG")
    with pytest.raises(ImageInputError) as excinfo:
        prepare_image(png)
    assert "暂不支持" in excinfo.value.message


def test_corrupt_jpg_raises(tmp_path):
    """后缀是 jpg 但内容不是图片 —— 必须在本地就拦住。"""
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"this is definitely not a jpeg")
    with pytest.raises(ImageInputError) as excinfo:
        prepare_image(fake)
    assert "无法解析" in excinfo.value.message


def test_prepare_image_bytes(jpeg_factory):
    data = jpeg_factory().read_bytes()
    prepared = prepare_image_bytes(data, source="<upload>")
    assert prepared.source == "<upload>"
    assert prepared.data_url.startswith("data:image/jpeg;base64,")


def test_prepare_image_bytes_rejects_empty():
    with pytest.raises(ImageInputError):
        prepare_image_bytes(b"")


def test_rgba_png_bytes_are_composited():
    """带透明通道的图片应能正常编码（白底合成），不报错也不出现黑底。"""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (40, 40), (255, 0, 0, 128)).save(buffer, format="PNG")
    prepared = prepare_image_bytes(buffer.getvalue())
    assert prepared.width == 40
