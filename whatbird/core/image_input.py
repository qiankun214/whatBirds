"""图片输入层：把用户给的 jpg 文件校验、必要时压缩，并编码成 base64 data URL。

方舟的图片理解接口既支持公网 URL 也支持 base64。本应用面向「本地 jpg 文件路径」，
所以统一走 base64，好处是无需把图片上传到任何其他地方。

使用 Pillow 的价值（也是选它的原因）：
- 提前拦下损坏文件 / 改了后缀的假 jpg，给出中文错误，而不是等 API 报错；
- 超大图按最长边等比缩小，显著降低 token 消耗与请求体大小。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import ImageInputError

#: 按 Agent.md 第 6 条，第一版只接受 jpg
SUPPORTED_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg")

DEFAULT_MAX_SIDE = 1568
DEFAULT_QUALITY = 88
MIME_TYPE = "image/jpeg"


@dataclass(slots=True)
class PreparedImage:
    """已就绪的图片：data URL + 尺寸/体积等元信息。"""

    source: str
    data_url: str
    width: int
    height: int
    original_width: int
    original_height: int
    byte_size: int
    resized: bool
    mime_type: str = MIME_TYPE

    @property
    def size_kb(self) -> float:
        return round(self.byte_size / 1024, 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "size_kb": self.size_kb,
            "resized": self.resized,
            "mime_type": self.mime_type,
        }


def prepare_image(
    path: str | Path,
    *,
    max_image_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
) -> PreparedImage:
    """读取本地 jpg 文件并生成可直接投喂模型的 data URL。"""
    image_path = Path(path).expanduser()

    if not image_path.exists():
        raise ImageInputError(
            f"图片文件不存在：{image_path}",
            hint="请检查路径是否正确；Windows 路径含空格时记得加引号",
        )
    if not image_path.is_file():
        raise ImageInputError(
            f"这是一个目录，不是图片文件：{image_path}",
            hint="请传入具体的 .jpg / .jpeg 文件",
        )
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageInputError(
            f"暂不支持该格式：{image_path.suffix or '（无扩展名）'}",
            hint="第一版仅支持 .jpg / .jpeg，可用 Pillow 或画图工具转为 jpg",
        )

    try:
        raw = image_path.read_bytes()
    except OSError as exc:
        raise ImageInputError(f"读取图片失败：{image_path}", hint=str(exc)) from exc

    return _encode(
        raw,
        source=str(image_path),
        max_image_side=max_image_side,
        quality=quality,
    )


def prepare_image_bytes(
    data: bytes,
    *,
    source: str = "<upload>",
    max_image_side: int = DEFAULT_MAX_SIDE,
    quality: int = DEFAULT_QUALITY,
) -> PreparedImage:
    """与 :func:`prepare_image` 相同，但输入是内存中的字节（Web 上传用）。"""
    if not data:
        raise ImageInputError("上传的图片内容为空", hint="请重新选择一张 jpg 图片")
    return _encode(data, source=source, max_image_side=max_image_side, quality=quality)


def _encode(raw: bytes, *, source: str, max_image_side: int, quality: int) -> PreparedImage:
    """校验 -> 转 RGB -> 按需缩放 -> 重新编码为 JPEG -> base64。"""
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageInputError(
            f"无法解析该图片，文件可能已损坏或并不是真正的 JPEG：{source}",
            hint="请换一张能正常打开的 jpg 图片",
        ) from exc

    try:
        with Image.open(io.BytesIO(raw)) as image:
            original_width, original_height = image.size
            prepared = _to_rgb(image)
            if max(original_width, original_height) > max_image_side:
                prepared.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
            width, height = prepared.size
            buffer = io.BytesIO()
            prepared.save(buffer, format="JPEG", quality=quality, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError(f"图片处理失败：{source}", hint=str(exc)) from exc

    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    return PreparedImage(
        source=source,
        data_url=f"data:{MIME_TYPE};base64,{encoded}",
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        byte_size=len(payload),
        resized=(width, height) != (original_width, original_height),
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    """统一转成 RGB；带透明通道的图片用白底合成，避免出现黑色背景。"""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return image.convert("RGB")


@dataclass(slots=True)
class ImageSource:
    """一张待识别图片：既可以是本地文件路径，也可以是内存中的字节（Web 上传）。

    批处理要同时支持「服务器上的多个路径」和「浏览器上传的多个文件」，
    用这个类型把两者统一起来，core 层就不必区分来源。
    """

    name: str
    path: str | None = None
    data: bytes | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageSource":
        text = str(path)
        return cls(name=Path(text).name or text, path=text)

    @classmethod
    def from_bytes(cls, name: str, data: bytes) -> "ImageSource":
        return cls(name=name, data=data)

    @classmethod
    def coerce(cls, value: "ImageSource | str | Path | bytes | tuple[str, bytes]") -> "ImageSource":
        """把各种写法统一成 :class:`ImageSource`（方便 CLI 直接传路径字符串）。"""
        if isinstance(value, ImageSource):
            return value
        if isinstance(value, (bytes, bytearray)):
            return cls.from_bytes("<bytes>", bytes(value))
        if isinstance(value, tuple) and len(value) == 2:
            name, payload = value
            return cls.from_bytes(str(name), bytes(payload))
        return cls.from_path(value)

    def prepare(
        self,
        *,
        max_image_side: int = DEFAULT_MAX_SIDE,
        quality: int = DEFAULT_QUALITY,
    ) -> PreparedImage:
        """校验并编码成可直接投喂模型的 data URL。"""
        if self.path is not None:
            return prepare_image(self.path, max_image_side=max_image_side, quality=quality)
        return prepare_image_bytes(
            self.data or b"",
            source=self.name,
            max_image_side=max_image_side,
            quality=quality,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "byte_size": len(self.data) if self.data is not None else None,
        }
