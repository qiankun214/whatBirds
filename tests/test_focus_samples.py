"""真实相机样张的验证（``samples/`` 目录）。

这一组用例**只在 ``samples/`` 里放了真实带 EXIF 的相机原图时才会跑**，
否则整体 skip —— 没有样张不应该让测试变红。

为什么需要它：厂商 MakerNote 的坐标语义（佳能的"以图像中心为原点"、索尼的私有
坐标系）无法靠合成样本验证，只能拿真实原图对照。手机截图、微信转存会把 MakerNote
洗掉，所以样张必须是**相机导出的原片**。

放样张的约定见 ``samples/README.md``。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from whatbird.core import analyze_focus

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SUFFIXES = (".jpg", ".jpeg")
SAMPLES = sorted(
    (item for item in SAMPLES_DIR.iterdir() if item.suffix.lower() in SUFFIXES),
    key=lambda item: item.name,
) if SAMPLES_DIR.is_dir() else []

NO_SAMPLES = pytest.mark.skipif(
    not SAMPLES,
    reason="samples/ 里还没有真实相机原图（见 samples/README.md）",
)


@NO_SAMPLES
@pytest.mark.parametrize("path", SAMPLES, ids=[item.name for item in SAMPLES])
def test_sample_analyzes_without_crashing(path):
    report = analyze_focus(path)

    assert report.width > 0 and report.height > 0
    assert report.params
    # 两条路线都必须给出明确结论或明确原因，不能是空字符串
    assert report.exif.detail
    assert report.content.detail
    assert report.to_dict()["image"]["width"] == report.width


@NO_SAMPLES
@pytest.mark.parametrize("path", SAMPLES, ids=[item.name for item in SAMPLES])
def test_sample_exif_point_is_inside_the_image(path):
    report = analyze_focus(path)

    if not report.exif.found:
        pytest.skip(f"该样张没有可用的 EXIF 对焦点：{report.exif.detail}")

    assert 0 <= report.exif.x < report.width
    assert 0 <= report.exif.y < report.height
    assert 0.0 <= report.exif.x_ratio <= 1.0
    assert report.exif.source_tags.get("tag"), "必须记录对焦点是从哪个标签读出来的"
    # 白名单之外不许出现任何键（GPS 之类的隐私回归）
    assert set(report.exif.source_tags) <= {"make", "model", "tag", "raw_value"}


@NO_SAMPLES
@pytest.mark.parametrize("path", SAMPLES, ids=[item.name for item in SAMPLES])
def test_sample_content_point_is_inside_the_image(path):
    report = analyze_focus(path)

    if not report.content.found:
        pytest.skip(f"该样张未能定位内容对焦点：{report.content.detail}")

    assert 0 <= report.content.x < report.width
    assert 0 <= report.content.y < report.height


@NO_SAMPLES
@pytest.mark.parametrize("path", SAMPLES, ids=[item.name for item in SAMPLES])
def test_sample_agreement_is_reported(path):
    """两条路线是否一致只作**记录**，不作断言 —— 不一致本身就是有价值的信息。"""
    report = analyze_focus(path)

    if report.exif.found and report.content.found:
        assert isinstance(report.agree, bool)
        print(
            f"\n{path.name}: EXIF ({report.exif.x},{report.exif.y}) "
            f"vs 内容 ({report.content.x},{report.content.y}) → "
            f"{'一致' if report.agree else '不一致'}"
        )


def _exiftool_value(path: Path, wanted: str):
    """用 exiftool 取某个标签的值；没装 exiftool 就返回 None。"""
    executable = shutil.which("exiftool")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "-j", "-G1", "-s", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境相关
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:  # pragma: no cover
        return None
    for entry in payload:
        for key, value in entry.items():
            if wanted.lower() in key.lower():
                return value
    return None


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="本机没有 exiftool，跳过交叉验证")
@pytest.mark.parametrize("path", SAMPLES, ids=[item.name for item in SAMPLES])
def test_sample_standard_tag_matches_exiftool(path):
    """有 exiftool 时，用它的结果交叉验证我们自己解出来的 SubjectArea。

    这是目前唯一能独立验证坐标语义的办法：exiftool 是被业界当作事实标准的实现。
    """
    expected = _exiftool_value(path, "SubjectArea")
    if not isinstance(expected, list) or len(expected) < 2:
        pytest.skip("该样张的 exiftool 输出里没有 SubjectArea")

    report = analyze_focus(path)
    if not report.exif.found or "SubjectArea" not in report.exif.source_tags.get("tag", ""):
        pytest.skip("EXIF 路线不是从 SubjectArea 得到的结论，无法直接比对")

    if len(expected) == 4:
        want_x, want_y = expected[0] + expected[2] / 2, expected[1] + expected[3] / 2
    else:
        want_x, want_y = expected[0], expected[1]

    span = max(report.width, report.height) * 0.02
    assert abs(report.exif.x - want_x) <= span, (
        f"与 exiftool 不一致：我们 ({report.exif.x},{report.exif.y})，exiftool ({want_x},{want_y})"
    )
    assert abs(report.exif.y - want_y) <= span
