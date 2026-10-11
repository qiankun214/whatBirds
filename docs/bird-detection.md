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
  `--all-classes` 可以看全部类别，但 `birds` 里仍然只装鸟。
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
| `--model` / `--model-dir` | `yolo11n.pt` / `~/.whatbird/models` | 权重与缓存目录 |
| `--device` | `auto` | `auto` 优先 Intel 显卡（xpu），不可用退回 cpu |
| `--all-classes` | 关 | 不限制类别；其他类别仍只列入说明、不算结果 |

Python API 里对应 `DetectParams`，可用 `DetectParams.from_dict({...})` 构造
（容忍未知字段，方便上层透传）。

---

## 5. 设备：Intel 显卡优先

```python
DEVICE_AUTO = "auto"   # 能用 XPU 就用，否则 cpu
DEVICE_XPU  = "xpu"
DEVICE_CPU  = "cpu"
```

- `auto` 走 `backend.xpu_available()`：`torch.xpu.is_available()` 为真才用显卡。
  判定单独抽成一个函数，因此"有显卡/没显卡"两条分支都有测试覆盖。
- **显式指定 `--device xpu` 而不可用时直接报错**，不悄悄退回 CPU ——
  否则你会以为在用显卡，实际一直在 CPU 上跑。
- 实际使用的设备一定写进结果的 `device` 字段，并在 `notes` 里说明回退原因。
- 权重按 `(权重路径, 设备)` 缓存（`get_detector()`），CLI 一次给多张图不会反复加载模型。

前置条件：torch 必须是 **XPU 版**（`2.8.0+xpu`），默认源装的是 CPU 版；
以及 Intel 显卡驱动。自检：

```bash
python -c "import torch;print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
```

本机环境：Intel Arc A770（16 GB），torch 2.8.0+xpu、torchvision 0.23.0+xpu。

---

## 6. 权重与离线使用

- 权重缓存目录默认 `~/.whatbird/models`（用户目录，与 API 配置同级，**不在仓库内**），
  可用 `--model-dir` 或 `DetectParams(model_dir=...)` 覆盖。
- 查找顺序：`--model` 指向的已存在文件 → 权重目录里的同名文件 → 交给 ultralytics 下载到权重目录。
  所以**只要文件在位就完全离线**，不会每次联网查一遍。
- 首次运行会从 GitHub 下载权重，国内可能失败。手动办法：拿到 `yolo11n.pt` 后放进
  `~/.whatbird/models`，或 `--model D:\models\yolo11n.pt` 直接指定路径。

---

## 7. 已知边界

- **只给位置，不给种类**，也不做"是不是同一只鸟"的跟踪/去重；
- 小型远距鸟、强烈遮挡、罕见姿态（背面剪影、只露出头部）容易漏检 —— 先调 `--imgsz` 与 `--conf`；
- 误检：树枝、石块、玩具鸟可能被框成鸟，提高 `--conf` 可压掉一部分；
- 极暗/极低对比度的照片召回会下降；
- 不做鸟类分类模型微调（见第 8 节）；
- Web 界面暂未接本功能（只有 CLI 与 Python API）。

---

## 8. 扩展方向

### 8.1 接上识别流程：检测 → 逐只裁剪 → 送 Doubao 识别种类

门面里的 `Detection` 已经带 `box = (x1, y1, x2, y2)` 与 `DetectionReport`，
裁剪只是加一个"按框裁出区域"的函数；`BirdRecognizer.recognize_source()` 接受
`ImageSource`，天然支持内存字节，所以"一张图多只鸟 → 每只一个物种结论"不需要改现有识别代码。

### 8.2 Web 界面

`/api/focus` 那套是现成模板：同步返回 `{"result": ..., "overlay": overlay_data_url(png)}`，
`overlay_data_url()` 与 `render_bird_overlay()` 检测模块已经提供。
注意 `tests/test_frontend.py` 的契约：前端 JS 里出现的每个 `/api/...` 必须真实注册。

### 8.3 换推理后端 / 导出

若 XPU 路径出现兼容问题，可用 ultralytics 自带 `export` 转 OpenVINO / ONNX，
再实现一个满足 `Detector` 协议的后端传给 `detect_birds(detector=...)` 即可，业务规则无需改动。

### 8.4 鸟类专用检测（提高稀有场景召回）

若 COCO 的笼统 `bird` 类在目标场景不够用，可以在鸟类检测数据集上做微调
（如公开的鸟类检测数据集），产出的 `best.pt` 用 `--model` 直接替换即可，
`DetectParams.classes` 改成新权重的鸟类 id 列表。

---

## 9. 验证方式

- **单元/集成测试**（`tests/test_detect_*.py`，53 条）：全部用注入的假后端与合成图，
  不联网、不需要显卡与权重，因此 CI / 新克隆的仓库都能跑。
- **真实照片**：`python -m whatbird.cli detect <你的鸟片> --overlay out.png`，
  人工看框是否落在鸟身上、`device` 是否为 `xpu`、标注图是否可打开。
  把样张放进 `samples/` 可让 `tests/test_detect_samples.py` 自动参与（见 `samples/README.md`）。
