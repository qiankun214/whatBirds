# samples/ —— 真实照片验证样张

这个目录同时给两组测试用：

| 测试 | 用途 | 目录为空时 |
| --- | --- | --- |
| [`tests/test_focus_samples.py`](../tests/test_focus_samples.py) | 对焦点分析：验证厂商 MakerNote 的坐标语义 | 整体 skip |
| [`tests/test_detect_samples.py`](../tests/test_detect_samples.py) | 鸟类位置检测：验证框是否落在鸟身上 | 整体 skip |

两组都会**自动发现**这里的 `.jpg` / `.jpeg`，空目录不会让测试变红。

> 这个目录里的图片**不会进 git**（`.gitignore` 里是 `samples/*` + `!samples/README.md`），
> 因为是你自己的拍摄原片。

## 为什么要放样张在这里

### 对焦点分析

有两条路线（见 [`../docs/focus-algorithm.md`](../docs/focus-algorithm.md)）：

| 路线 | 验证方式 |
| --- | --- |
| 内容分析（传统图形学） | **合成样本就能验证**，`tests/test_focus_content.py` 已经覆盖，不需要样张 |
| EXIF 元数据 | 只有**真实相机原片**能验证 |

EXIF 那条路读的是厂商私有数据（MakerNote），同一个厂商不同机型写的大小、坐标系原点、
缩放比例都可能不一样。这些**无法靠推理确定，只能拿真实原图对照**，所以必须用样张。

### 鸟类位置检测

检测的"业务规则"（只报鸟、排序、钳制、标注图）用假后端与合成图就能完整覆盖
（见 [`../docs/bird-detection.md`](../docs/bird-detection.md) 第 9 节）。**只有一件事合成图测不了：
模型在真实鸟片上到底框得准不准。** 所以样张用来回答"框住的是不是鸟"，而不是"代码对不对"。

放鸟片时请顺便说明每张图里**大概有几只鸟**，测试会把这个数字和检测结果对照：

- 检测到**更多**：说明有误检（树枝、石头被当成鸟）；
- 检测到**更少**：说明漏检（鸟太小/太远/被遮挡），可试 `--imgsz 960`、`--conf 0.15`。

## 放什么

**必须是相机导出的原片**，不要用这些（它们会把 MakerNote 洗掉，测试就没有意义了）：

- 手机截图
- 微信 / QQ / 微博等社交软件转发或保存的图片
- 经过修图软件导出并"去除元数据"的图片
- 从 HEIC / RAW 转换过来、但转换时没保留 EXIF 的图片

想要覆盖得全一些的话，**佳能 / 尼康 / 索尼 / 富士各来 1~2 张**最好 —— 这四家的
MakerNote 结构差别最大。

## 怎么放

直接把文件拷进来即可，命名随意（文件名只用于测试 ID）：

```
samples/
├── README.md
├── canon-r5-单点对焦.jpg
├── nikon-z9-区域对焦.jpg
└── sony-a1-鸟眼对焦.jpg
```

## 可选：让我知道"正确答案"在哪

如果想让验证更硬，可以在文件名或下面的表格里写一句话说明这张的对焦点大致在哪
（例如"鸟眼，画面中部偏左"）。我会把它和算法结果对照，而不是只看"坐标在图像范围内"。

| 文件名 | 机型 | 对焦点大概在哪 | 备注 |
| --- | --- | --- | --- |
| （示例）canon-r5-单点对焦.jpg | Canon EOS R5 | 鸟眼，中部偏左 | 单点 AF |
| | | | |
| | | | |

## 可选：鸟片里大概有几只鸟

位置检测的测试会读这张表里的"期望只数"，与检测结果对照（不做硬断言，只报告差异）：

| 文件名 | 大概几只鸟 | 备注（鸟的大小/遮挡/光照） |
| --- | --- | --- |
| （示例）sparrow-branch.jpg | 2 | 中等大小，树枝部分遮挡 |
| | | |

## 装样张后怎么跑

```bash
python -m pytest tests/test_focus_samples.py -v     # 对焦点分析
python -m pytest tests/test_detect_samples.py -v    # 鸟类位置检测
```

检测那组需要装好可选依赖（`requirements-detect.txt`）并且权重在位；否则整组 skip。
想先把权重备好（避免测试时卡在下载），要么手动放到 `~/.whatbird/models/yolo11n.pt`，
要么在 `_expected_birds` 表外的测试里指定 `--model`：

```bash
python -m whatbird.cli detect samples\你的鸟片.jpg --overlay out.png
```

如果本机装了 [exiftool](https://exiftool.org/)，对焦点测试会**自动**拿它的输出来交叉验证
我们解出来的 `SubjectArea`，这是目前唯一能独立检验坐标语义的办法。
