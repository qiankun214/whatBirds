"""``whatbird detect`` 子命令的行为测试（注入假后端，不加载模型）。"""

from __future__ import annotations

import json

import pytest

from whatbird import cli
from whatbird.core.detect.models import BIRD_CLASS_ID, RawBox

BIRD = RawBox(0.10, 0.20, 0.30, 0.60, 0.91, BIRD_CLASS_ID, "bird")
DOG = RawBox(0.50, 0.50, 0.90, 0.90, 0.99, 16, "dog")


class FakeDetector:
    def __init__(self, boxes, device="xpu"):
        self.boxes = list(boxes)
        self.device = device

    def __call__(self, image, params):
        return list(self.boxes), self.device


@pytest.fixture
def patched_detect(monkeypatch):
    """把 cli 用到的 detect_birds 换成假后端版本，并记录调用参数。"""

    def install(boxes, device="xpu"):
        recorder = {"params": [], "sources": []}

        def fake_detect(source, *, params=None, detector=None):
            from whatbird.core.detect import detect_birds as real

            recorder["sources"].append(str(source))
            recorder["params"].append(params)
            return real(source, params=params, detector=FakeDetector(boxes, device))

        monkeypatch.setattr(cli, "detect_birds", fake_detect)
        return recorder

    return install


def test_detect_prints_positions_and_boxes(patched_detect, jpeg_factory, capsys):
    path = jpeg_factory(name="bird.jpg", size=(200, 100))
    patched_detect([BIRD])

    code = cli.main(["detect", str(path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "鸟类位置检测" in out
    assert "#1" in out
    assert "x=20 y=20 w=40 h=40" in out
    assert "91%" in out


def test_detect_json_output_matches_contract(patched_detect, jpeg_factory, capsys):
    path = jpeg_factory(name="bird.jpg", size=(200, 100))
    patched_detect([BIRD])

    assert cli.main(["detect", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["birds"][0]["x"] == 20
    assert payload["birds"][0]["x_ratio"] == pytest.approx(0.1)
    assert payload["device"] == "xpu"
    assert payload["found"] is True


def test_detect_reports_other_classes(patched_detect, jpeg_factory, capsys):
    path = jpeg_factory(name="mixed.jpg")
    patched_detect([DOG])

    assert cli.main(["detect", str(path)]) == 0

    out = capsys.readouterr().out
    assert "未检测到鸟" in out
    assert "dog" in out


def test_detect_saves_overlay_for_single_image(patched_detect, jpeg_factory, tmp_path, capsys):
    path = jpeg_factory(name="bird.jpg", size=(240, 180))
    patched_detect([BIRD], device="cpu")
    target = tmp_path / "out" / "marked.png"

    assert cli.main(["detect", str(path), "--overlay", str(target)]) == 0

    assert target.is_file()
    from PIL import Image

    with Image.open(target) as image:
        assert image.size == (240, 180)
    assert "标注图已保存" in capsys.readouterr().out


def test_detect_overlay_dir_for_multiple_images(patched_detect, jpeg_factory, tmp_path):
    first = jpeg_factory(name="a.jpg")
    second = jpeg_factory(name="b.jpg")
    patched_detect([BIRD])
    out_dir = tmp_path / "overlays"

    assert cli.main(["detect", str(first), str(second), "--overlay-dir", str(out_dir)]) == 0

    assert (out_dir / "a.detect.png").is_file()
    assert (out_dir / "b.detect.png").is_file()


def test_detect_rejects_overlay_with_multiple_images(jpeg_factory, tmp_path, capsys):
    first = jpeg_factory(name="a.jpg")
    second = jpeg_factory(name="b.jpg")

    code = cli.main(["detect", str(first), str(second), "--overlay", str(tmp_path / "x.png")])

    assert code == 2
    assert "--overlay-dir" in capsys.readouterr().err


def test_detect_rejects_overlay_and_overlay_dir_together(jpeg_factory, tmp_path, capsys):
    path = jpeg_factory(name="a.jpg")

    code = cli.main(
        ["detect", str(path), "--overlay", str(tmp_path / "x.png"), "--overlay-dir", str(tmp_path)]
    )

    assert code == 2
    assert "只能给一个" in capsys.readouterr().err


def test_detect_passes_device_and_confidence_through(patched_detect, jpeg_factory):
    path = jpeg_factory(name="bird.jpg")
    recorder = patched_detect([BIRD])

    assert cli.main(["detect", str(path), "--device", "cpu", "--conf", "0.4", "--imgsz", "960"]) == 0

    params = recorder["params"][0]
    assert params.device == "cpu"
    assert params.conf == pytest.approx(0.4)
    assert params.imgsz == 960
    assert params.classes == (BIRD_CLASS_ID,)


def test_detect_all_classes_clears_the_filter(patched_detect, jpeg_factory):
    path = jpeg_factory(name="bird.jpg")
    recorder = patched_detect([BIRD])

    assert cli.main(["detect", str(path), "--all-classes"]) == 0

    assert recorder["params"][0].classes == ()


def test_detect_project_config_never_used(patched_detect, jpeg_factory, tmp_path, monkeypatch):
    """检测是纯本地的：即使完全没配置 API Key 也必须能跑。"""
    monkeypatch.setenv("WHATBIRD_CONFIG_FILE", str(tmp_path / "none" / "config.json"))
    path = jpeg_factory(name="bird.jpg")
    patched_detect([BIRD])

    assert cli.main(["detect", str(path)]) == 0


def test_detect_dependency_error_returns_code_2(jpeg_factory, monkeypatch, capsys):
    from whatbird.core import DetectorDependencyError

    path = jpeg_factory(name="bird.jpg")

    def broken(source, *, params=None, detector=None):
        raise DetectorDependencyError("缺少依赖 ultralytics", hint="用清华镜像装")

    monkeypatch.setattr(cli, "detect_birds", broken)

    code = cli.main(["detect", str(path)])

    assert code == 2
    err = capsys.readouterr().err
    assert "缺少依赖" in err
    assert "清华镜像" in err


def test_detect_all_classes_lists_non_bird_objects_as_results(patched_detect, jpeg_factory, capsys):
    """--all-classes 之后非鸟目标进结果表，且措辞不能谎报成「2 只鸟」。"""
    path = jpeg_factory(name="mixed.jpg")
    patched_detect([BIRD, DOG])

    assert cli.main(["detect", str(path), "--all-classes"]) == 0

    out = capsys.readouterr().out
    assert "其中 1 只鸟" in out
    assert "[dog]" in out
    assert "#2" in out
