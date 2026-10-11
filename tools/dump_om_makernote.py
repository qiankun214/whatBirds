"""把 OM System（OM-1 等）原始 MakerNote 里的所有字段导出成表格，用于和 exiftool 对照。

背景
----
OM System 的 MakerNote 用的是私有头 ``"OM SYSTEM\\0\\0\\0"``（老机型是 ``"OLYMPUS"``），
不是标准 TIFF IFD，所以 Pillow 与 ExifRead 都会直接跳过它。exifread 跳过它的原因更具体：
``exif_header.py`` 里写死了 ``if make.startswith("OLYMPUS")``，而新机身报的是
``"OM Digital Solutions"``。

但它的内部结构其实很规整，本脚本按下面这套结构解：

    "OM SYSTEM\\0" + \\0\\0 + "II" + version(2)      <- 字节序标记在偏移 12
    IFD @ MakerNote+16，所有偏移**相对 MakerNote 起点**
      ├─ 0x2010 Equipment
      ├─ 0x2020 CameraSettings
      ├─ 0x2030 RawDevelopment
      ├─ 0x2040 ImageProcessing
      └─ 0x2050 FocusInfo          <- AF 数据在这里

**本脚本只导出原始字段，不做任何语义判断。** 把厂商私有字段翻译成
"对焦点在哪"必须拿 exiftool 的输出当基准，猜是猜不出来的。

用法::

    python tools/dump_om_makernote.py PA052644.JPG
    python tools/dump_om_makernote.py *.JPG --ifd FocusInfo --watch
    python tools/dump_om_makernote.py A.JPG B.JPG --compare
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

#: MakerNote 的私有头（新机身 / 老机身）
OM_HEADERS: tuple[bytes, ...] = (b"OM SYSTEM", b"OLYMPUS")

#: 已知的子 IFD 标签 -> 名字
SUB_IFDS: dict[int, str] = {
    0x2010: "Equipment",
    0x2020: "CameraSettings",
    0x2030: "RawDevelopment",
    0x2040: "ImageProcessing",
    0x2050: "FocusInfo",
}

#: TIFF 字段类型 -> 每元素字节数
TYPE_SIZE: dict[int, int] = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8,
}

#: 这些是"看着像坐标"的量级范围；只用来给候选打标记，不做判断
_COORD_HINT = (0, 8192)


class MakernoteError(Exception):
    """找不到或解不开 MakerNote。"""


def locate_makernote(raw: bytes) -> tuple[bytes, int]:
    """从 JPEG 字节里取出 MakerNote 的原始字节，返回 ``(内容, 在文件里的偏移)``。"""
    app1 = raw.find(b"\xff\xe1")
    if app1 < 0:
        raise MakernoteError("没有 APP1 段（不是带 EXIF 的 JPEG？）")
    tiff = app1 + 10
    if raw[tiff : tiff + 2] == b"II":
        order = "<"
    elif raw[tiff : tiff + 2] == b"MM":
        order = ">"
    else:
        raise MakernoteError("TIFF 头不对")

    def entries(off: int) -> list[tuple[int, int, int, bytes]]:
        count = struct.unpack(order + "H", raw[tiff + off : tiff + off + 2])[0]
        out = []
        for index in range(count):
            entry = tiff + off + 2 + index * 12
            tag, kind, num = struct.unpack(order + "HHI", raw[entry : entry + 8])
            size = TYPE_SIZE.get(kind, 1) * num
            if size <= 4:
                data = raw[entry + 8 : entry + 8 + size]
            else:
                pointer = struct.unpack(order + "I", raw[entry + 8 : entry + 12])[0]
                data = raw[tiff + pointer : tiff + pointer + size]
            out.append((tag, kind, num, data))
        return out

    def offset_of(parsed: list[tuple[int, int, int, bytes]], want: int, base: int) -> int | None:
        for index, (tag, _kind, _num, _data) in enumerate(parsed):
            if tag == want:
                entry = base + 2 + index * 12
                return struct.unpack(order + "I", raw[entry + 8 : entry + 12])[0]
        return None

    ifd0 = struct.unpack(order + "I", raw[tiff + 4 : tiff + 8])[0]
    top = entries(ifd0)
    exif_off = offset_of(top, 0x8769, tiff + ifd0)
    if exif_off is None:
        raise MakernoteError("没有 Exif 子 IFD")
    sub = entries(exif_off)
    mn_off = offset_of(sub, 0x927C, tiff + exif_off)
    if mn_off is None:
        raise MakernoteError("Exif 子 IFD 里没有 MakerNote (0x927C)")
    blob = raw[tiff + mn_off :]
    return blob, tiff + mn_off


def _header_end(blob: bytes) -> tuple[int, str]:
    """返回 ``(字节序标记的位置, 字节序)``；顺带认出是哪一代的头。"""
    for header in OM_HEADERS:
        if blob.startswith(header):
            break
    else:
        raise MakernoteError(f"不是 OM System 的 MakerNote（头是 {blob[:12]!r}）")
    for index in range(0, 32):
        pair = blob[index : index + 2]
        if pair in (b"II", b"MM"):
            return index, "<" if pair == b"II" else ">"
    raise MakernoteError("MakerNote 里找不到字节序标记")


def parse_makernote(blob: bytes) -> dict[str, dict[int, tuple[int, int, bytes]]]:
    """把 MakerNote 解成 ``{IFD 名: {标签: (类型, 数量, 原始字节)}}``。"""
    marker, order = _header_end(blob)
    ifd_start = marker + 4

    def read_ifd(offset: int) -> dict[int, tuple[int, int, bytes]]:
        count = struct.unpack(order + "H", blob[offset : offset + 2])[0]
        out: dict[int, tuple[int, int, bytes]] = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            if entry + 12 > len(blob):
                break
            tag, kind, num = struct.unpack(order + "HHI", blob[entry : entry + 8])
            size = TYPE_SIZE.get(kind, 1) * num
            if size <= 4:
                data = blob[entry + 8 : entry + 8 + size]
            else:
                pointer = struct.unpack(order + "I", blob[entry + 8 : entry + 12])[0]
                data = blob[pointer : pointer + size]   # 偏移相对 MakerNote 起点
            out[tag] = (kind, num, data)
        return out

    main = read_ifd(ifd_start)
    result: dict[str, dict[int, tuple[int, int, bytes]]] = {"MakerNote": main}
    for index, (tag, kind, num, _data) in enumerate(
        _raw_entries(blob, order, ifd_start)
    ):
        del kind, num
        if tag in SUB_IFDS:
            entry = ifd_start + 2 + index * 12
            pointer = struct.unpack(order + "I", blob[entry + 8 : entry + 12])[0]
            if 0 < pointer < len(blob):
                result[SUB_IFDS[tag]] = read_ifd(pointer)
    return result


def _raw_entries(blob: bytes, order: str, offset: int):
    """主 IFD 的原始条目（保留顺序，用来算子 IFD 的指针位置）。"""
    count = struct.unpack(order + "H", blob[offset : offset + 2])[0]
    for index in range(count):
        entry = offset + 2 + index * 12
        tag, kind, num = struct.unpack(order + "HHI", blob[entry : entry + 8])
        yield tag, kind, num, entry


def decode(kind: int, num: int, data: bytes):
    """把原始字节解成 python 值（能解成数字就解，否则给十六进制）。"""
    try:
        if kind == 3:
            return list(struct.unpack("<%dH" % num, data[: num * 2]))
        if kind == 8:
            return list(struct.unpack("<%dh" % num, data[: num * 2]))
        if kind == 4:
            return list(struct.unpack("<%dI" % num, data[: num * 4]))
        if kind == 9:
            return list(struct.unpack("<%di" % num, data[: num * 4]))
        if kind == 5 and num <= 6:
            return [struct.unpack("<II", data[i * 8 : i * 8 + 8]) for i in range(num)]
    except struct.error:
        pass
    if kind == 2:
        return data.rstrip(b"\x00").decode("latin-1", "replace")
    return f"<{num}B> {data[:32].hex()}"


def looks_like_coordinates(values) -> bool:
    """粗筛：值都落在"图像坐标"该有的量级里（只用来提示，不下结论）。"""
    if not isinstance(values, list) or len(values) < 2:
        return False
    numbers = [v for v in values if isinstance(v, int)]
    if len(numbers) < 2:
        return False
    return all(_COORD_HINT[0] <= v <= _COORD_HINT[1] for v in numbers) and max(numbers) > 32


def dump(path: Path, *, ifd: str | None, watch: bool) -> dict:
    raw = path.read_bytes()
    blob, offset = locate_makernote(raw)
    tree = parse_makernote(blob)
    print(f"\n=== {path.name} ===  MakerNote @file+{offset}  共 {len(blob)} 字节")
    for name, tags in tree.items():
        if ifd and name != ifd:
            continue
        print(f"  ── {name}  {len(tags)} 个标签")
        for tag in sorted(tags):
            kind, num, data = tags[tag]
            value = decode(kind, num, data)
            hint = ""
            if watch and looks_like_coordinates(value):
                hint = "   <-- 看着像坐标"
            elif watch and any(
                word in str(value).upper() for word in ("AF", "FOCUS")
            ):
                hint = "   <-- 提到 AF/Focus"
            print(f"     0x{tag:04X} type={kind:<2d} n={num:<5d} {str(value)[:78]}{hint}")
    return tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="导出 OM System MakerNote 的原始字段，用于和 exiftool 对照"
    )
    parser.add_argument("images", nargs="+", help="jpg 文件路径")
    parser.add_argument("--ifd", help=f"只看某个 IFD：{' / '.join(SUB_IFDS.values())}")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="给量级像坐标的字段打标记（只是提示，不是结论）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="多张图时只打印同一个 IFD 的字段对照表",
    )
    args = parser.parse_args(argv)

    paths = [Path(item) for item in args.images]
    for path in paths:
        if not path.is_file():
            print(f"[错误] 文件不存在：{path}", file=sys.stderr)
            return 2

    if args.compare and len(paths) > 1:
        wanted = args.ifd or "CameraSettings"
        tables = []
        for path in paths:
            raw = path.read_bytes()
            blob, _ = locate_makernote(raw)
            tables.append((path.name, parse_makernote(blob).get(wanted, {})))
        tags = sorted({tag for _n, table in tables for tag in table})
        print(f"\n=== {wanted} 对照表 ===")
        header = "  0xTAG  " + "".join(f"{name[:22]:>26s}" for name, _t in tables)
        print(header)
        for tag in tags:
            cells = []
            for _name, table in tables:
                if tag in table:
                    kind, num, data = table[tag]
                    cells.append(f"{str(decode(kind, num, data))[:24]:>26s}")
                else:
                    cells.append(f"{'—':>26s}")
            print(f"  0x{tag:04X} " + "".join(cells))
        return 0

    for path in paths:
        dump(path, ifd=args.ifd, watch=args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
