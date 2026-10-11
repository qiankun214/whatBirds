"""对焦分析的「元数据」路线：读相机自己写进 EXIF 的对焦点。

分三层，从最可靠的往下退：

**L1 标准 EXIF 标签（Pillow，零额外依赖）**
``SubjectArea`` (0x9214) / ``SubjectLocation`` (0xA214) 是 EXIF 标准里的
"主体位置"。相当多的相机在单点对焦时会把对焦框写在这里 —— 这是覆盖面最广、
也最容易验证的一层，所以放在最前面。

**L2 厂商 MakerNote（ExifRead + 自解结构）**
- 富士 ``FocusPixel``：直接就是图像像素坐标，最干净；
- 佳能 ``AFInfo2``：ExifRead 只解出标量、会把坐标数组截断，所以这里自己按
  uint16 数组解（布局见 :data:`_CANON_AFINFO2`）；
- 尼康 / 索尼：只给出 AF 点序号或私有坐标系，缺机型对照表无法可靠换算，
  宁可老实说"没有"也不猜一个坐标出来。

**L3 降级**
三层都没有 → ``found=False``，并给出中文原因。截图、社交软件转存、多数手机
照片本来就没有 MakerNote，这是正常情况，不是错误。

隐私
----
EXIF 里有 GPS 等敏感信息。本模块**只读白名单里的标签**，对外只暴露
``make / model / tag / raw_value``，绝不把整个 EXIF 字典传出去。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import ExifTags, Image

from ..errors import FocusDependencyError
from .models import METHOD_EXIF, FocusHit

# ---- 标准 EXIF 标签号 ----
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_ORIENTATION = 0x0112
TAG_EXIF_IFD = 0x8769
TAG_SUBJECT_AREA = 0x9214
TAG_SUBJECT_LOCATION = 0xA214
TAG_PIXEL_X = 0xA002
TAG_PIXEL_Y = 0xA003

# ---- 佳能 MakerNote 标签号 ----
CANON_TAG_AFINFO2 = 0x0026

#: 佳能 AFInfo2 的布局（**uint16 数组下标**，不是字节偏移）。
#:
#: 这份下标与 ExifRead 自己的 ``_AF_INFO_2`` 表一致（2=AFAreaMode、3=NumAFPoints、
#: 4=ValidAFPoints、5=CanonImageWidth），数组起点按同一份 Canon 规格顺推：
#: 9 起依次是 AFAreaWidths / AFAreaHeights / AFAreaXPositions / AFAreaYPositions，
#: 每个数组长度都是 NumAFPoints。
_CANON_AFINFO2: dict[str, int] = {
    "af_area_mode": 2,
    "num_points": 3,
    "valid_points": 4,
    "canon_width": 5,
    "canon_height": 6,
    "af_width": 7,
    "af_height": 8,
    "arrays_start": 9,
}

#: 佳能的 AF 坐标是以**图像中心为原点**给出的，换算成左上角原点：
#: ``x = 宽/2 + X * 宽 / AFImageWidth``
_CANON_CENTER_ORIGIN = True

#: 允许出现在对外输出里的标签键（白名单，防止 GPS 之类顺手泄漏）
_TAG_KEYS = ("make", "model", "tag", "raw_value")

_MAX_AF_POINTS = 256


@dataclass(slots=True)
class AfReading:
    """EXIF 路线的原始读数（坐标在**图像存储方向**的像素空间）。"""

    x: int
    y: int
    confidence: float
    detail: str
    source: str
    tags: dict[str, str] = field(default_factory=dict)

    def whitelisted(self) -> dict[str, str]:
        return {key: self.tags[key] for key in _TAG_KEYS if key in self.tags}


# --------------------------------------------------------------------- 工具
def _find(tags: dict[str, Any], name: str) -> Any | None:
    """按标签名找 ExifRead 的条目（键形如 ``"MakerNote FocusPixel"``）。"""
    for key, value in tags.items():
        if key == name or key.endswith(" " + name):
            return value
    return None


def _values(node: Any) -> list[Any]:
    raw = getattr(node, "values", node)
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("\x00")


def _uint16_list(value: Any) -> list[int]:
    """把 MakerNote 里的一段原始数据统一成 uint16 列表。"""
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray)):
        count = len(value) // 2
        return list(struct.unpack("<%dH" % count, bytes(value[: count * 2])))
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, int):
                out.append(item)
            else:
                out.extend(_uint16_list(item))
        return out
    return []


def apply_orientation(x: int, y: int, width: int, height: int, orientation: int) -> tuple[int, int]:
    """把存储方向的像素坐标，按 EXIF Orientation 映射到**转正后**的坐标系。

    与 ``PIL.ImageOps.exif_transpose`` 用的是同一组变换（见其源码的 method 映射表），
    所以 EXIF 读出的点和转正后的图像天然对齐。
    """
    if width <= 0 or height <= 0:
        return max(x, 0), max(y, 0)
    last_x, last_y = width - 1, height - 1
    if orientation == 2:  # 水平镜像
        return max(last_x - x, 0), max(y, 0)
    if orientation == 3:  # 旋转 180°
        return max(last_x - x, 0), max(last_y - y, 0)
    if orientation == 4:  # 垂直镜像
        return max(x, 0), max(last_y - y, 0)
    if orientation == 5:  # 转置
        return max(y, 0), max(x, 0)
    if orientation == 6:  # 顺时针 90°
        return max(last_y - y, 0), max(x, 0)
    if orientation == 7:  # 反转置
        return max(last_y - y, 0), max(last_x - x, 0)
    if orientation == 8:  # 逆时针 90°
        return max(y, 0), max(last_x - x, 0)
    return max(x, 0), max(y, 0)


def orientation_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    """转正后的图像尺寸（5/6/7/8 会交换宽高）。"""
    if orientation in (5, 6, 7, 8):
        return height, width
    return width, height


# --------------------------------------------------------------------- L1 标准标签
def _interpret_subject(
    values: list[Any],
    stored_width: int,
    stored_height: int,
    reference: tuple[int, int],
) -> tuple[int, int] | None:
    """按 EXIF 规范解释 SubjectArea / SubjectLocation 的取值。

    - 2 个值 ``(x, y)`` → 中心点；
    - 3 个值 ``(x, y, d)`` → 圆（取圆心）；
    - 4 个值 ``(x, y, w, h)`` → 矩形（取中心）。

    单位可能是像素，也可能是 0~1 的比例（不同厂商写法不一），用前两个值的量级判断。
    """
    try:
        numbers = [float(item) for item in values]
    except (TypeError, ValueError):
        return None

    if len(numbers) == 2:
        raw_x, raw_y = numbers[0], numbers[1]
    elif len(numbers) == 3:
        raw_x, raw_y = numbers[0], numbers[1]
    elif len(numbers) == 4:
        raw_x = numbers[0] + numbers[2] / 2.0
        raw_y = numbers[1] + numbers[3] / 2.0
    else:
        return None

    if max(abs(numbers[0]), abs(numbers[1])) <= 1.0:
        # 0~1 的比例写法
        x = raw_x * stored_width
        y = raw_y * stored_height
    else:
        ref_w, ref_h = reference
        scale_x = stored_width / ref_w if ref_w > 0 else 1.0
        scale_y = stored_height / ref_h if ref_h > 0 else 1.0
        x = raw_x * scale_x
        y = raw_y * scale_y

    point = (int(round(x)), int(round(y)))
    # 允许一点点越界（有些相机的矩形起点在画面外），但离谱的直接判为不可信
    if not (-stored_width <= point[0] <= stored_width * 2):
        return None
    if not (-stored_height <= point[1] <= stored_height * 2):
        return None
    return max(min(point[0], stored_width - 1), 0), max(min(point[1], stored_height - 1), 0)


def _from_standard(
    exif: Any,
    stored_width: int,
    stored_height: int,
    base_tags: dict[str, str],
) -> AfReading | None:
    try:
        exif_ifd = exif.get_ifd(TAG_EXIF_IFD) or {}
    except Exception:  # pragma: no cover - 畸形 EXIF
        exif_ifd = {}

    reference = (
        int(exif_ifd.get(TAG_PIXEL_X) or 0),
        int(exif_ifd.get(TAG_PIXEL_Y) or 0),
    )
    if reference[0] <= 0 or reference[1] <= 0:
        reference = (stored_width, stored_height)

    for tag, name, label in (
        (TAG_SUBJECT_AREA, "Exif.Photo.SubjectArea", "SubjectArea"),
        (TAG_SUBJECT_LOCATION, "Exif.Photo.SubjectLocation", "SubjectLocation"),
    ):
        values = exif_ifd.get(tag)
        if not values:
            continue
        values = list(values) if isinstance(values, (list, tuple)) else [values]
        point = _interpret_subject(values, stored_width, stored_height, reference)
        if point is None:
            continue
        shape = {2: "点", 3: "圆", 4: "矩形"}.get(len(values), "未知形状")
        tags = dict(base_tags)
        tags.update({"tag": name, "raw_value": ",".join(str(item) for item in values)})
        return AfReading(
            x=point[0],
            y=point[1],
            confidence=0.8,
            detail=f"来自 EXIF 标准标签 {label}（{len(values)} 值{shape}）",
            source=name,
            tags=tags,
        )
    return None


# --------------------------------------------------------------------- L2 MakerNote
def _from_fujifilm(
    tags: dict[str, Any], base_tags: dict[str, str]
) -> AfReading | None:
    """富士 ``FocusPixel`` 直接给出图像像素坐标，是最干净的一种。"""
    node = _find(tags, "FocusPixel")
    if node is None:
        return None
    values = _values(node)
    if len(values) < 2:
        return None
    try:
        x, y = int(values[0]), int(values[1])
    except (TypeError, ValueError):
        return None
    merged = dict(base_tags)
    merged.update({"tag": "MakerNote.FocusPixel", "raw_value": f"{x},{y}"})
    return AfReading(
        x=max(x, 0),
        y=max(y, 0),
        confidence=0.95,
        detail="来自富士 MakerNote FocusPixel（相机直接记录的对焦像素）",
        source="MakerNote.FocusPixel",
        tags=merged,
    )


def _canon_afinfo_values(exif: Any) -> list[int]:
    """从 Pillow 的 MakerNote 子 IFD 取出佳能 AFInfo2 的原始 uint16 数组。

    不用 ExifRead 的佳能解码结果：它把坐标数组当标量输出，只保留了每个数组的
    第一个元素，拿不到对焦点位置。
    """
    try:
        makernote = exif.get_ifd(ExifTags.IFD.MakerNote) or {}
    except Exception:  # pragma: no cover - 畸形 MakerNote
        return []
    return _uint16_list(makernote.get(CANON_TAG_AFINFO2))


def _from_canon(exif: Any, stored_width: int, stored_height: int,
                base_tags: dict[str, str]) -> AfReading | None:
    values = _canon_afinfo_values(exif)
    layout = _CANON_AFINFO2
    need = layout["arrays_start"]
    if len(values) < need + 1:
        return None

    num = int(values[layout["num_points"]])
    af_width = int(values[layout["af_width"]])
    af_height = int(values[layout["af_height"]])
    canon_width = int(values[layout["canon_width"]])
    canon_height = int(values[layout["canon_height"]])

    # ---- 严格校验：任何一条不满足就宁可不说，也不给一个可能错的坐标 ----
    if not (0 < num <= _MAX_AF_POINTS):
        return None
    if not (0 < af_width <= 20000 and 0 < af_height <= 20000):
        return None
    if not (0 < canon_width <= 20000 and 0 < canon_height <= 20000):
        return None
    if len(values) < need + 4 * num:
        return None

    start = need
    xs = values[start + 2 * num : start + 3 * num]
    ys = values[start + 3 * num : start + 4 * num]
    if len(xs) != num or len(ys) != num:
        return None

    focus_words = values[start + 4 * num : start + 5 * num] or [0]
    mask = int(focus_words[0]) if focus_words else 0
    index = None
    for bit in range(num):
        if mask & (1 << bit):
            index = bit
            break
    if index is None:
        # AFPointsInFocus 为空表示"没有单点合焦"（区域对焦/手动对焦），不硬猜
        return None

    # 佳能坐标以图像中心为原点，换算到左上角原点并缩放回实际存储尺寸
    x = canon_width / 2.0 + xs[index] * canon_width / float(af_width)
    y = canon_height / 2.0 + ys[index] * canon_height / float(af_height)
    x = x * stored_width / float(canon_width)
    y = y * stored_height / float(canon_height)

    if not (0 <= x < stored_width and 0 <= y < stored_height):
        return None

    merged = dict(base_tags)
    merged.update({
        "tag": "MakerNote.AFInfo2",
        "raw_value": f"point={index + 1}/{num} x={xs[index]} y={ys[index]}",
    })
    return AfReading(
        x=int(round(x)),
        y=int(round(y)),
        confidence=0.95,
        detail=f"来自佳能 MakerNote AFInfo2（第 {index + 1}/{num} 个对焦点合焦）",
        source="MakerNote.AFInfo2",
        tags=merged,
    )


def _nikon_unavailable(tags: dict[str, Any], base_tags: dict[str, str]) -> str:
    if _find(tags, "AFInfo2") is None and _find(tags, "AFInfo") is None:
        return ""
    return (
        "这台尼康相机确实写了 AFInfo，但里面只有 AF 点的网格序号；"
        "换算成像素坐标需要该机型的对焦点位置表，暂不支持（不猜坐标）"
    )


def _read_makernote(
    image: Image.Image,
    exif: Any,
    raw: bytes | None,
    stored_width: int,
    stored_height: int,
    base_tags: dict[str, str],
) -> AfReading | None:
    make = base_tags.get("make", "").lower()

    # 佳能优先走 Pillow 原始字节（ExifRead 会把坐标数组截断）
    if "canon" in make:
        reading = _from_canon(exif, stored_width, stored_height, base_tags)
        if reading is not None:
            return reading

    if raw is None:
        return None

    try:
        import exifread
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise FocusDependencyError(
            "EXIF 对焦点解析缺少依赖：ExifRead",
            hint=(
                "python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple exifread"
            ),
        ) from exc

    try:
        tags = exifread.process_file(
            BytesIO(raw), details=True, extract_thumbnail=False
        )
    except Exception:
        return None

    if "fuji" in make:
        reading = _from_fujifilm(tags, base_tags)
        if reading is not None:
            return reading

    return None


# --------------------------------------------------------------------- 对外入口
def read_af_point(
    image: Image.Image,
    raw: bytes | None = None,
) -> AfReading | None:
    """读出一张图里相机记录的对焦点；没有就返回 ``None``。

    :param image: **存储方向**的 PIL 图像（还没做 exif_transpose）。
    :param raw: 原始文件字节（读 MakerNote 用；内存上传的图片也需要）。
    """
    try:
        exif = image.getexif()
    except Exception:  # pragma: no cover - 畸形 EXIF
        exif = None

    try:
        stored_width, stored_height = image.size
    except Exception:  # pragma: no cover
        return None

    base_tags: dict[str, str] = {}
    if exif is not None:
        base_tags["make"] = _text(exif.get(TAG_MAKE))
        base_tags["model"] = _text(exif.get(TAG_MODEL))
        reading = _from_standard(exif, stored_width, stored_height, base_tags)
        if reading is not None:
            return reading
        return _read_makernote(
            image, exif, raw, stored_width, stored_height, base_tags
        )
    return None


def unavailable_reason(image: Image.Image, raw: bytes | None) -> str:
    """没有可用的 EXIF 对焦点时，给一句**具体**的中文原因。"""
    make = model = ""
    tags: dict[str, Any] = {}
    try:
        exif = image.getexif()
        make = _text(exif.get(TAG_MAKE))
        model = _text(exif.get(TAG_MODEL))
    except Exception:  # pragma: no cover
        exif = None

    if raw:
        try:
            import exifread

            tags = exifread.process_file(BytesIO(raw), details=True, extract_thumbnail=False)
        except Exception:
            tags = {}

    if "nikon" in make.lower():
        reason = _nikon_unavailable(tags, {})
        if reason:
            return reason

    if not make:
        return (
            "该图片没有可用的 EXIF 对焦信息"
            "（截图、社交软件转存或部分手机拍摄的图通常不带）"
        )
    camera = f"{make} {model}".strip()
    return (
        f"该图片来自 {camera}，但里面没有可用的对焦点信息"
        "（可能是区域/手动对焦，或该机型未写入对焦点）"
    )


def to_hit(reading: AfReading | None, reason: str, *, width: int, height: int,
           orientation: int = 1, stored_width: int = 0, stored_height: int = 0) -> FocusHit:
    """把存储方向的读数转成**转正后坐标系**的 :class:`FocusHit`。"""
    from .models import ratio_of

    if reading is None:
        return FocusHit(method=METHOD_EXIF, found=False, detail=reason)

    x, y = reading.x, reading.y
    if orientation and orientation != 1:
        x, y = apply_orientation(x, y, stored_width, stored_height, orientation)
    x = min(max(int(x), 0), max(width - 1, 0))
    y = min(max(int(y), 0), max(height - 1, 0))
    return FocusHit(
        method=METHOD_EXIF,
        found=True,
        x=x,
        y=y,
        x_ratio=ratio_of(x, width),
        y_ratio=ratio_of(y, height),
        confidence=reading.confidence,
        detail=reading.detail,
        source_tags=reading.whitelisted(),
    )
