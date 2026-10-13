# 鸟类位置检测：模型、坐标与边界

给一张 jpg，找出画面里**每一只鸟的位置**（框 + 置信度），并可选生成画框标注图。

对应代码：`whatbird/core/detect/`，门面函数 `detect_birds()`，CLI `whatbird detect`。
与识别种类（`identify`）是两件事：**本功能只回答"鸟在哪"，不回答"是什么鸟"。**

---

## 1. 分层结构（为什么这么切）

```
whatbird/core/detect/
├── models.py    数据契约 + 参数 + 归一化框 → 像素框的换算（不依赖 torch/ultralytics）
├── backend.py   ultralytics 适配层：唯一 import torch/ultralytics 的地方
├── overlay.py   画框标注图（只依赖 Pillow）
└── __init__.py  门面 detect_birds()，可注入自定义 detector
```

三层隔离换来三个具体好处：

1. **可选依赖不污染主功能**：`import whatbird.core` 不会 import torch，也不要求装了检测依赖。
   没装时其余功能（识别 / 对焦 / 配置）完全不受影响；只有真正调用检测才抛
   `DetectorDependencyError`，并附带按清华镜像的安装命令。
2. **业务规则可以脱离显卡测试**："只报鸟、按置信度排序、越界框钳制、类别说明"
   全在 core 层，用注入的假后端就能跑 —— 53 条检测测试**不需要显卡、不需要权重、不联网**。
3. **换推理后端不用动业务代码**：`detector` 参数接受任何
   `(已转正的 RGB 图, DetectParams) -> (归一化框列表, 设备名)` 的可调用对象。
   将来换 ONNX Runtime / OpenVINO 只需新增一个后端。

后端只输出**归一化**的 `xyxy`（`RawBox`），像素钳制与比例换算统一在 `models.build_detection()`：
这样"越界框"只有一处修正逻辑，真后端与假后端行为一致。

---

## 2. 模型与类别

- 默认权重 **`yolo11n.pt`**（YOLO11 nano）。选择理由：找位置不需要细粒度分类，
  nano 在 CPU 上也只要几十毫秒，且权重最小（约 5 MB）。
- 权重来自 COCO 预训练，共 **80 类，其中 `bird` 是独立类别（id 14）** ——
  所以不需要自己训练就能检测"鸟"这个概念。
- **默认只返回鸟**（`DetectParams.classes = (14,)`）。画面里的狗、人、车会被列在
  结果的 `other_classes` 与中文说明里，用来解释"画面里明明有目标，为什么没报鸟"。
  `--all-classes` 把类别过滤关掉，非鸟目标也会进结果表并画框（结果里会标出类别名，
  摘要也会说"其中 N 只鸟"，不谎报成鸟）。
- **类别过滤是事后做的（core 层），不下推给模型** —— 这条容易写错，代价是一个真实 bug：
  曾经 `predict(classes=[14])` 直接传给 ultralytics，模型层的过滤让非鸟框根本回不到上层，
  于是**默认参数下 `other_classes` 永远是空的**，只有加 `--all-classes` 才有内容，
  文档承诺的"解释为什么没报鸟"实际上不成立。现在 `YoloDetector` 固定传 `classes=None`，
  过滤统一在 `build_detections()` 里按 `params.classes` 做。
  这样做**不会变慢**：ultralytics 本来就是先推理、再按类别筛 `classes` 只过滤 NMS 之后的列表。
  回归测试见 `tests/test_detect_dependency.py::test_backend_never_pushes_the_class_filter_down_to_the_model`。
- **不区分鸟种**：COCO 里鸟就是一个笼统类别。要区分种类请用 `identify`（Doubao）。

换更大的权重（精度换速度）：

```bash
whatbird detect D:\photos\bird.jpg --model yolo11s.pt    # 更准一些，慢 2~3 倍
whatbird detect D:\photos\bird.jpg --model yolo11m.pt    # 再准一些，再慢
```

---

## 3. 坐标系约定

对外的 `x / y / width / height` 都是**按 EXIF Orientation 转正后的原图**像素坐标：
原点在左上角，`x` 向右、`y` 向下。同时给出 0~1 的 `x_ratio / y_ratio / width_ratio / height_ratio`。

- **为什么要转正**：竖拍照片（`Orientation=6/8`）的像素数组是横着的，不先转正，
  框会和用户看到的画面错位。`detect_birds` 与 `render_bird_overlay` 都走
  `image_input.load_upright_rgb()`，与对焦点分析**同一套口径**，所以
  "检测框"和"对焦点"能叠在同一张图上。
- **越界处理**：`xyxy_to_xywh()` 把框钳制进画面，并保证宽高至少 1 像素
  （模型偶尔给出负坐标或超出 1 像素的框，画图时会变成不可见的线或越界矩形）。
- **归一化在边界上**：`raw_boxes_from_results()` 把 ultralytics 的像素框除以
  `orig_shape` 变成 0~1，`build_detection()` 再乘回目标图尺寸。
  好处是同一份原始结果可以按不同尺寸的图换算，也让测试不必构造真实尺寸的框。

---

## 4. 参数

| 参数 | 默认 | 作用与调法 |
| --- | --- | --- |
| `--conf` | 0.25 | 置信度阈值。漏检小鸟时调低（0.1~0.15），误检多时调高（0.4~0.5） |
| `--iou` | 0.7 | NMS 的 IoU 阈值，控制同一只鸟的重复框。重叠框多时调低 |
| `--imgsz` | 640 | 推理分辨率。**远距小鸟的主要杠杆**：960/1280 明显提召回，耗时随面积增长 |
| `--max-det` | 300 | 单张最多返回多少框（拥挤画面防爆内存） |
| `--model` / `--model-dir` | `yolo11n.pt` / `~/.whatbird/models` | 权重与缓存目录；服务端用环境变量 `WHATBIRD_MODEL_DIR` 覆盖（Web 没有 `--model-dir` 可传） |
| `--device` | `auto` | `auto` 优先 Intel 显卡（xpu），不可用退回 cpu |
| `--all-classes` | 关 | 关掉类别过滤：非鸟目标也进结果表并画框（默认只把鸟算结果，非鸟类别只进说明） |

Python API 里对应 `DetectParams`，可用 `DetectParams.from_dict({...})` 构造
（容忍未知字段，方便上层透传）。

---

## 5. 设备：Intel 显卡优先

```python
DEVICE_AUTO = "auto"   # 能用 XPU 就用，否则 cpu
DEVICE_XPU  = "xpu"
DEVICE_CPU  = "cpu"
```

- `auto` 走 `backend.xpu_available()`：先确认 `torch` 有 `xpu` 属性（CPU 版没有），
  再看 `torch.xpu.is_available()`。判定单独抽成一个函数，因此"有显卡/没显卡"两条分支都有测试覆盖。
- **显式指定 `--device xpu` 而不可用时直接报错**，不悄悄退回 CPU ——
  否则你会以为在用显卡，实际一直在 CPU 上跑。
- 实际使用的设备一定写进结果的 `device` 字段，并在 `notes` 里说明回退原因。
- 权重按 `(权重路径, 设备)` 缓存（`get_detector()`），CLI 一次给多张图不会反复加载模型。

前置条件（只在要显卡时）：torch 必须是 **XPU 版**（版本号带 `+xpu` 后缀），
默认源装的都是 CPU 版；以及 Intel 显卡驱动。自检（CPU 版会打印 False 而不是报错）：

```bash
python -c "import torch;print(torch.__version__, hasattr(torch,'xpu') and torch.xpu.is_available())"
```

本机环境：**无 Intel 显卡（CPU-only）**，torch 2.14.1+cpu、torchvision 0.29.1；`--device auto` 自动落到 `cpu`。
装法见 [`../requirements-detect.txt`](../requirements-detect.txt)：CPU 机器直接走清华镜像即可，
XPU 那条路只有真的有 Intel 显卡时才用。

---

## 6. 权重与离线使用

- 权重缓存目录默认 `~/.whatbird/models`（用户目录，与 API 配置同级，**不在仓库内**），
  可用 `--model-dir`、`DetectParams(model_dir=...)`、或环境变量 `WHATBIRD_MODEL_DIR` 覆盖
  （服务端没有 `--model-dir` 可传，只认这个环境变量）。
- 查找顺序：`--model` 指向的已存在文件 → 权重目录里的同名文件 → 交给 ultralytics 下载到权重目录。
  所以**只要文件在位就完全离线**，不会每次联网查一遍。
- 首次运行会从 GitHub Releases 取权重，国内基本连不上。仓库自带走镜像的下载脚本：

  ```bash
  python tools/fetch_yolo_weight.py                        # -> ~/.whatbird/models
  python tools/fetch_yolo_weight.py --dest .whatbird/models
  python tools/fetch_yolo_weight.py --list                 # 只列出会尝试的地址
  ```

  依次尝试：官方 GitHub → `ghfast.top` → `gh-proxy.com` → `ghproxy.net` → `hf-mirror.com` →
  ModelScope。每个地址下完都校验"文件头是 `PK` 且大于 1 MB"，避免把镜像的错误页当权重存下来。
- 权重目录不可写时（受限沙箱、只读用户目录）会抛 `DetectorDependencyError` 并提示换目录，
  而不是把裸 `PermissionError` 冒到 Web 层变成 500。

---

## 7. 已知边界

- **只给位置，不给种类**，也不做"是不是同一只鸟"的跟踪/去重；
- 小型远距鸟、强烈遮挡、罕见姿态（背面剪影、只露出头部）容易漏检 —— 先调 `--imgsz` 与 `--conf`；
- 误检：树枝、石块、玩具鸟可能被框成鸟，提高 `--conf` 可压掉一部分；
- 极暗/极低对比度的照片召回会下降；
- 不做鸟类分类模型微调（见第 8 节）；
- Web 界面已接：`/api/detect`、`/api/detect-path`，页面上是「4 · 鸟类位置检测」卡片。
  接口可以带 `conf` / `imgsz` / `all_classes`（上传走表单字段，路径走 JSON），页面按钮目前只用默认值。

---

## 8. 扩展方向

### 8.1 接上识别流程：检测 → 逐只裁剪 → 送 Doubao 识别种类

门面里的 `Detection` 已经带 `box = (x1, y1, x2, y2)` 与 `DetectionReport`，
裁剪只是加一个"按框裁出区域"的函数；`BirdRecognizer.recognize_source()` 接受
`ImageSource`，天然支持内存字节，所以"一张图多只鸟 → 每只一个物种结论"不需要改现有识别代码。

### 8.2 Web 界面（已实现）

`/api/focus` 那套是模板：同步返回 `{"result": ..., "overlay": overlay_data_url(png)}`。
检测版是 `web/app.py` 里的 `detect_response()`，两个路由：

| 路由 | 入参 | 说明 |
| --- | --- | --- |
| `POST /api/detect` | multipart `file` + 可选表单字段 `conf` / `imgsz` / `all_classes` | 上传一张 jpg |
| `POST /api/detect-path` | JSON `{path, conf?, imgsz?, all_classes?}`（`extra="forbid"`，拼错字段直接 422） | 服务器上的路径 |

两道防线与 focus 一致：都要请求头 `X-WhatBird-Client: 1`（防跨站），上传大小上限 20 MB。
前端契约由 `tests/test_frontend.py` 守着（JS 里出现的每个 `/api/...` 必须真实注册，
面板里的每个 `id` 必须存在于 `index.html`），接口行为由 `tests/test_detect_api.py` 覆盖
（注入假后端，不加载权重）。

### 8.3 换推理后端 / 导出

若 XPU 路径出现兼容问题，可用 ultralytics 自带 `export` 转 OpenVINO / ONNX，
再实现一个满足 `Detector` 协议的后端传给 `detect_birds(detector=...)` 即可，业务规则无需改动。

### 8.4 鸟类专用检测（提高稀有场景召回）

若 COCO 的笼统 `bird` 类在目标场景不够用，可以在鸟类检测数据集上做微调
（如公开的鸟类检测数据集），产出的 `best.pt` 用 `--model` 直接替换即可，
`DetectParams.classes` 改成新权重的鸟类 id 列表。

---

## 9. 验证方式

### 9.1 自动化测试

检测相关的自动化测试共 **76 条**（全量套件 292 条，全绿）。分布：

- **单元/集成测试**（`tests/test_detect_*.py`）：全部用注入的假后端与合成图，
  不联网、不需要显卡与权重，因此 CI / 新克隆的仓库都能跑。
- **接口层**（`tests/test_detect_api.py`）：注入假后端跑 REST 契约 —— 参数透传（含
  `conf` / `imgsz` / `all_classes`）、缺权重时翻成 400 + 中文提示、标注图能解码回原尺寸。
- **前端契约**（`tests/test_frontend.py`）：面板里每个 `id` 必须存在、JS 里每个 `/api/...`
  必须真实注册、`.detect-*` 样式必须存在。
- 其中一条特别重要（回归）：`test_backend_never_pushes_the_class_filter_down_to_the_model`
  —— 钉死"类别过滤只能在 core 层做"，否则 `other_classes` 又会悄悄变成空的。

### 9.2 真实照片（已实测）

```bash
python -m whatbird.cli detect <鸟片> --model-dir <权重目录> --overlay out.png
```

实测样张（公开鸟图，`yolo11n.pt`，`--imgsz 640`，`device=cpu`）：

| 样张 | 检出 | 置信度 | 框 | 人工核对 |
| --- | --- | --- | --- | --- |
| 翠鸟 640×427 | 1 只 | 92% | x=229 y=21 w=335 h=386 | 框贴住鸟身，无多余框 |
| 翠鸟 1200×801 | 1 只 | 94% | x=571 y=156 w=437 h=513 | 同上 |

耗时（同一台 CPU-only 机器，810×1080 的图）：冷启动 6.17 s（含 import torch + 加载权重 + 一次推理）；
模型加载完成后每张 —— `imgsz 640` 中位 **45 ms**、960 **72 ms**、1280 **124 ms**。
所以"单张命令要 6 秒"几乎全是进程启动成本，批量时后端会复用模型（按「权重 + 设备」缓存）。

`--overlay` 生成的标注图要人工看两件事：框是否落在鸟身上、图例里的设备与数量是否正确。

### 9.3 用你自己的原片

把鸟片放进 `samples/` 可让 `tests/test_detect_samples.py` 自动参与（见 `samples/README.md`，
表里写"大概几只鸟"即可，测试会对照检出数量）。
真实相机原片比公开网图更有价值：可以同时验证 EXIF 转正后的坐标系（竖拍照片）。
