# HANDOFF（检测）—— WhatBird 鸟类位置检测（YOLO + Intel XPU）

> 交接时间：2026-10-11 20:50 左右（Asia/Shanghai）｜ 工作区 `E:\WhatBird`
> 这是**检测功能**的交接文档。`focus`（对焦点分析）另有一份：[`HANDOFF.md`](HANDOFF.md)。
> 两份文档描述的是**同一工作区里并行推进的两件事**，共写文件见 §9。
> 接手前先读 [`Agent.md`](Agent.md)（固定载入约定）与 [`README.md`](README.md) §4.2。
> 📌 **本文件已提交入库**（2026-10-11 由用户确认）：交接完成后请**更新它**而不是删掉 ——
> 把已完成的条目勾掉、把新发现补进去。

---

## 0. 三十秒速览

**做了什么**：给 WhatBird 加「用 YOLO 找出图片里每一只鸟的位置」的能力 ——
给一张 jpg，返回每只鸟的**框 + 置信度**，可画出**原分辨率标注图**，优先用 **Intel 显卡（XPU）** 推理。

**交付面**：`core`（`detect_birds()`）+ CLI `whatbird detect` + 53 条测试 + 两份文档。
**纯本地**：不调用方舟、不消耗额度、不需要 API Key；只有首次需要下载 5.4 MB 的权重。

**当前状态**：代码写完、测试全绿；**环境已装好（XPU 可用）；真机推理验收还没跑**，
差的只是**权重文件**（GitHub / hf-mirror 在本机连不上，见 §6）。

**最要紧的事**：拿到 `yolo11n.pt` → 跑一次 `whatbird detect PA052099.JPG`（那是一只啄木鸟，
枝叶遮挡严重的"高漏检风险"样张）→ 人工确认框落在鸟身上。

---

## 1. 交付内容

### 1.1 新增文件

| 文件 | 职责 |
|---|---|
| `whatbird/core/detect/__init__.py` | 门面：`detect_birds(source, *, params, detector) -> DetectionReport` |
| `whatbird/core/detect/models.py` | 数据契约：`DetectParams` / `RawBox` / `Detection` / `DetectionReport`；归一化框→像素框换算与钳制（`xyxy_to_xywh`、`build_detection`） |
| `whatbird/core/detect/backend.py` | ultralytics 适配层：**唯一** import torch/ultralytics 的地方；设备解析、权重路径、结果转换、类别过滤、后端缓存 |
| `whatbird/core/detect/overlay.py` | 画框标注图（Pillow）：`render_detection_overlay` / `render_bird_overlay` / `overlay_data_url` |
| `docs/bird-detection.md` | 设计说明：分层、模型与类别、坐标约定、参数、设备、权重、边界、扩展方向 |
| `requirements-detect.txt` | 可选依赖与**安装顺序**（ASCII-only，含为什么必须 `--no-deps`） |
| `tests/test_detect_models.py` | 11 条：契约、参数容错、坐标换算与钳制 |
| `tests/test_detect_core.py` | 12 条：只报鸟、排序、比例换算、钳制、错误翻译、CPU 回退说明 |
| `tests/test_detect_dependency.py` | 11 条：缺 torch/ultralytics 时 `import whatbird.core` 仍可用、设备解析、权重路径 |
| `tests/test_detect_overlay.py` | 8 条：画框、不改原图、小框/越界框、EXIF 转正后的尺寸 |
| `tests/test_detect_cli.py` | 11 条：人类输出、JSON 契约、标注图落盘、参数透传、错误码 |
| `tests/test_detect_samples.py` | 4 条：真实样张（无权重/无样张时整体 skip） |

### 1.2 修改文件

| 文件 | 改动 |
|---|---|
| `whatbird/core/errors.py` | 新增 `DetectorDependencyError`（`FocusDependencyError` 旁边，同一写法） |
| `whatbird/core/__init__.py` | 导出 `DetectParams / Detection / DetectionReport / DetectionStyle / DetectorDependencyError / YoloDetector / detect_birds / render_bird_overlay` |
| `whatbird/core/image_input.py` | 新增 `load_upright_rgb()`（EXIF 转正 + RGB）、`encode_png()` |
| `whatbird/core/focus/overlay.py` | `render_focus_overlay()` 改为复用上面两个公共函数（**行为不变**；它同时改了 `max_side` 相关逻辑，见 §9） |
| `whatbird/cli.py` | 新增 `detect` 子命令 + `_detect_params()` / `_render_detections()` / `_detect_overlay_path_for()`；导入 `BIRD_CLASS_ID`、`DEFAULT_MODEL as DEFAULT_DETECT_MODEL`、`DetectionReport`；epilog 加示例 |
| `README.md` | 顶部能力说明；架构图与目录树加 `detect/`；新增 §4.2 检测用法；新增「可选：安装鸟类位置检测依赖」；FAQ 加 6 条；已知边界加检测小节 |

### 1.3 对外接口

```python
from whatbird.core import detect_birds, render_bird_overlay, DetectParams

report = detect_birds(r"D:\photos\bird.jpg")           # 或 bytes / ImageSource / PIL Image
report.count            # 检测到几只
report.device           # 实际推理设备：'xpu' 或 'cpu'
report.birds[0].box     # (x1, y1, x2, y2) 原图像素
report.birds[0].confidence
report.to_dict()        # 对外 JSON 契约（CLI --json 与将来 REST 共用）

png = render_bird_overlay(r"D:\photos\bird.jpg", report)   # 原分辨率画框 PNG
```

```powershell
whatbird detect <图片...> [--json] [--overlay 文件 | --overlay-dir 目录]
                [--model yolo11n.pt] [--model-dir 目录]
                [--device auto|xpu|cpu] [--conf 0.25] [--iou 0.7] [--imgsz 640] [--all-classes]
```

---

## 2. 关键设计决策（改动前先读理由）

### 2.1 三层隔离：`models` → `backend` → 门面

- `models.py` **不 import torch**：数据契约与坐标换算纯 Python，因此可用假后端测业务规则。
- `backend.py` 是**唯一**重依赖点，且 `import torch` / `import ultralytics` 全部**写在函数内部**。
  效果：`import whatbird.core`（CLI 每个子命令都会触发）不付出加载 PyTorch 的代价；
  **没装检测依赖时其余功能（识别 / 对焦 / 配置）完全不受影响**，只有调用检测才抛
  `DetectorDependencyError` 并给中文安装提示。这条有专门测试
  （`test_detect_dependency.py::test_import_whatbird_core_does_not_need_torch_or_ultralytics`）。
- 门面接受 `detector=` 参数（任何 `(RGB 图, DetectParams) -> (归一化框, 设备)` 的可调用对象），
  **53 条测试里绝大多数不需要显卡、不需要权重、不联网**，所以在新克隆的仓库里也能全绿。

### 2.2 后端只输出**归一化** `xyxy`

像素钳制与比例换算只在 `models.xyxy_to_xywh()` / `build_detection()` 一处：
假后端不必构造 640×480 的真实框，真后端也不必重复写钳制逻辑，两边行为一致。

### 2.3 只报鸟 + 解释"为什么没报鸟"

预训练权重是 COCO 80 类，`bird` 是独立类别（id 14）。默认 `classes=(14,)`：
`birds` 里**只装鸟**，其他类别进 `report.other_classes`，并附一句中文说明
（"画面里还有非鸟目标（按参数过滤，不计入结果）：dog(id 16)"）。这条规则在 core 层
（不在后端），所以没有显卡也能测。

### 2.4 坐标口径与 `focus` 一致

`detect_birds()` 与 `render_bird_overlay()` 都走 `image_input.load_upright_rgb()`
（`exif_transpose` → RGB），所以检测框与对焦点标注落在**同一套坐标系**（转正后的原图像素），
并且都提供 0~1 的比例字段供前端叠加。**竖拍照片不转正就会错位**，这条有测试
（`test_detect_overlay.py::test_render_bird_overlay_respects_exif_orientation`）。

### 2.5 设备策略：`auto` 回退，显式 `xpu` 不静默回退

- `auto` → 有 XPU 用 XPU，否则 CPU；
- 显式 `--device xpu` 而不可用 → **直接报错**（否则你会以为在用显卡，实际一直在 CPU 上跑）；
- **实际**设备一定写进 `report.device`，并在 `notes` 里说明回退原因。
- 判定抽成 `backend.xpu_available()`，两条分支都有测试覆盖。

### 2.6 权重按 `(路径, 设备)` 缓存

`backend.get_detector()` 带锁缓存，CLI 一次给多张图不会反复加载模型；
只缓存成功的后端，失败下次会重试（用户可能刚把权重放好）。

### 2.7 标注图文字只用 ASCII

不能假设用户机器装了中文字体。中文说明放在 CLI 输出与网页文本里，不烧进图片。
（与 `focus/overlay.py` 同一考虑。）

---

## 3. 环境现状（实测，2026-10-11 20:40）

```powershell
E:\WhatBird\.venv\Scripts\python.exe -c "import torch,torchvision,ultralytics,cv2,numpy;print(torch.__version__,torchvision.__version__,ultralytics.__version__,cv2.__version__,numpy.__version__,torch.xpu.is_available())"
# 实测：2.14.1+xpu 0.29.1+xpu 8.4.175 4.10.0 1.26.4 True   （设备 Intel(R) Arc(TM) A770 Graphics）
```

| 项 | 状态 |
|---|---|
| 解释器 | `E:\WhatBird\.venv\Scripts\python.exe`（Python 3.10.11） |
| torch / torchvision | **2.14.1+xpu / 0.29.1+xpu**（比预期新；ultralytics 要求 `torch>=1.8.0`，兼容） |
| ultralytics | 8.4.175 |
| 其它已装 | pyyaml、scipy、psutil、py-cpuinfo、ultralytics-thop、opencv-python-headless 4.10.0.84、numpy 1.26.4、exifread 3.5.1、pillow 11.3.0 |
| 权重 | ❌ 不在本地（`~/.whatbird/models` 与项目内都没有 `yolo11n.pt`） |
| 系统 Python | 里面**也**有一份 ultralytics + torch 2.8.0+xpu（之前误装），缺 cv2；**不要用它跑本项目** |

---

## 4. 接手后的第一步

```powershell
cd E:\WhatBird

# 1) 拿权重（§6 有下载办法），放到项目内目录（.whatbird/ 已 gitignore，沙箱也可写）
#    目标：E:\WhatBird\.whatbird\models\yolo11n.pt

# 2) 真机检测（PA052099.JPG 是一只枝叶遮挡下的啄木鸟，最能考验漏检）
E:\WhatBird\.venv\Scripts\python.exe -m whatbird.cli detect PA052099.JPG `
  --model-dir E:\WhatBird\.whatbird\models `
  --overlay E:\WhatBird\_out\PA052099.detect.png

# 3) 看：device 是否 xpu；框是否落在鸟身上；标注图能否打开
```

**可用的真实素材**（仓库根目录，用户自己的 OM-1 原片，未提交）：

| 文件 | 内容（另一会话已核实） | 备注 |
|---|---|---|
| `PA052099.JPG` | **大斑啄木鸟**，树冠里，枝叶遮挡严重 | 本次会话我人工看过缩略图，确认有鸟 |
| `PA052644.JPG` | **两只伯劳**（蓝天背景） | 对焦点在下鸟 |
| `PA052737.JPG` | **山斑鸠**，树叶背景 | |

三张都是 `OM Digital Solutions / OM-1`、**5184×3888**、Orientation=1。
验证通过后把鸟片拷进 `samples/`（`samples/*` 已 gitignore），`test_detect_samples.py` 会自动发现。

---

## 5. 验证方式与当前结果

| 验证 | 结果 |
|---|---|
| `pytest tests/test_detect_*.py -q` | ✅ 53 条全过（不需要显卡/权重/联网） |
| `pytest -q`（全量回归） | ✅ 全绿；`10 skipped` = `focus` 的 5 条 + 检测的 5 条（都因缺样张/权重） |
| CLI 参数解析 / `--help` | ✅ 实测正常（`detect` 已出现在子命令列表） |
| 真机推理（XPU） | ❌ **未跑**（缺权重） |
| 检测框落在鸟身上 | ❌ **未人工确认**（缺权重） |

---

## 6. ⛔ 当前阻塞点：权重下载

`yolo11n.pt` 约 5.4 MB，官方地址：

```
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
```

本机实测失败：

- `Invoke-WebRequest` 连 GitHub 与 `hf-mirror.com` 都报 `基础连接已经关闭: 接收时发生错误`；
- 另一会话在 [`HANDOFF.md`](HANDOFF.md) §7.1 记录得更清楚：**这个沙箱的 HTTPS 出网有问题
  （`schannel: SEC_E_NO_CREDENTIALS`），明文 HTTP 才通**。所以 pip 要用阿里云的 HTTP 源：
  `--trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/`。

**可行办法（按推荐顺序）**：

1. **浏览器 / 下载管理器**手动下上面那个 URL，存成
   `E:\WhatBird\.whatbird\models\yolo11n.pt`（约 5.4 MB，肉眼可判完整性）；
2. 让 `ultralytics` 自己下：`.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"`
   —— 如果它的下载器走的是能通的通道就会成功；
3. 走 HTTP 的镜像（如 ghproxy 类），用 `curl.exe -L` 试；
4. 换一台能连网的机器下好再拷过来。

**权重查找顺序**（`backend.resolve_model_path`）：
`--model` 指向的**已存在文件** → `--model-dir`（默认 `~/.whatbird/models`）里的同名文件 →
交给 ultralytics 下载到该目录。**建议显式用 `--model-dir E:\WhatBird\.whatbird\models`**：
本机沙箱写不进 `C:\Users\...\.whatbird\`（实测 `UnauthorizedAccessException`），
而项目内 `.whatbird/` 已 gitignore、可写。

---

## 7. 待办清单（按优先级）

1. **[阻塞]** 拿到权重（§6）。
2. **[阻塞]** 真机跑一次并人工确认（§4）。鸟小/遮挡严重时试 `--imgsz 960 --conf 0.15`；
   若确实漏检，记录实际 `--conf` / `--imgsz` 组合，写进文档。
3. **[重要]** 把 `requirements-detect.txt` 里的版本号**改成实测值**：
   `torch==2.14.1+xpu`、`torchvision==0.29.1+xpu`（当前写的是预估的 2.8.0 / 0.23.0），
   其余包用 `pip list` 查实际版本后一并钉住。
4. **[重要]** 更新 `docs/bird-detection.md` §5 与 README 里"本机环境：torch 2.8.0+xpu"这行
   （实际 2.14.1+xpu）。
5. **[重要]** 补**实测性能数字**（XPU 下单张耗时、`imgsz 640` vs `960`），
   现在文档里只有输出格式示例、没有真实数字。
6. **[建议]** git 提交，**分两个 commit**（见 §8）。
7. **[可选]** Web 界面接检测：`/api/detect` + `/api/detect-path`，照 `web/app.py` 里
   `focus_response()` 那段抄（同步返回 `{"result": ..., "overlay": overlay_data_url(png)}`，
   `overlay_data_url` 检测模块已提供）。注意 `tests/test_frontend.py` 的契约：
   前端 JS 里出现的每个 `/api/...` 必须真实注册。
8. **[可选]** 检测 → 逐只裁剪 → 送 Doubao 识别种类：`Detection.box` 已给出 `(x1,y1,x2,y2)`，
   `BirdRecognizer.recognize_source()` 接受内存字节，不需要改现有识别代码。
9. **[可选]** 加回归测试钉死"`cv2` 必须是 headless 版"，防止有人再装 GUI 版 opencv 顶掉它。
10. **[可选]** 若 XPU 路径出现兼容问题，用 ultralytics 的 `export` 转 OpenVINO/ONNX，
    再实现一个满足 `Detector` 协议的后端传给 `detect_birds(detector=...)`。

---

## 8. git 状态与提交建议

当前分支 `main`，**全部未提交**；工作区里混着 `focus`（另一会话）与 `detect`（本次）两轮工作。

**建议分两个 commit**（沿用仓库中文 conventional-commit 风格）：

```
feat(focus): 对焦点分析（EXIF + 传统图形学双路线，附标注图）        # 另一会话
feat(detect): YOLO 鸟类位置检测（XPU 优先 + 画框标注图 + CLI）      # 本次
```

⚠️ **不要 `git add -A`**：

- 仓库根目录有用户的**三张 OM-1 原片**（`PA052099.JPG` 等，12~15 MB 各），**不该入库**；
- `samples/` 里除 `README.md` 外的图片也由 `.gitignore` 的 `samples/*` 挡住，别强行加；
- 临时文件：`_torch_install.log`（已被 `*.log` 忽略）、`_preview_1.jpg`（我生成的缩略图）、
  `_out/`、`results/`、`.whatbird/`（配置与权重目录，已忽略）。

---

## 9. ⚠️ 两个会话并行改同一工作区（冲突风险）

另一个会话在做 **`focus` 对焦点分析**（见 [`HANDOFF.md`](HANDOFF.md)）。**共写文件**：

| 文件 | 谁改了什么 |
|---|---|
| `whatbird/core/image_input.py` | 我新增 `load_upright_rgb()` / `encode_png()` |
| `whatbird/core/focus/overlay.py` | 我把它 `render_focus_overlay()` 改为复用上面两个函数（**行为不变**，focus 测试即回归网）；它自己又改了 `max_side` 相关逻辑（我保留了它的改动） |
| `whatbird/core/__init__.py` | 我加 detect 的导出；它加 focus 的导出 |
| `whatbird/cli.py` | 我加 `detect` 子命令；它加 `focus` 子命令 |
| `whatbird/core/errors.py` | 我加 `DetectorDependencyError`；它加 `FocusDependencyError` |
| `README.md` | 我加 §4.2 与安装小节；它加了 focus 相关章节 |

**动手前先 `git status` + 看文件 mtime**。目前 `whatbird focus` 与 `whatbird detect` 两个子命令
并存且都能跑（已实测）。两边若合并成"检测框 + 该框锐不锐"，注意我的
`load_upright_rgb()` 文档里已写明"保证检测框与对焦点标注落在同一套坐标系" ——
**当前实现正好满足**，可以直接叠加。

---

## 10. 踩过的坑（别再踩）

### 10.1 PowerShell `Set-Content -Encoding utf8` 会给源码加 BOM

首行变成 `\ufeff"""` → **Python 直接 SyntaxError**。本次因此丢了三个文件才定位到。
写文件要无 BOM：

```powershell
[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
```

### 10.2 本机全路径调用解释器

在 `E:\WhatBird` 下直接敲 `python` 命中的是**系统 Python**（PATH 第一项），
包会装错地方 —— 本机已经踩过一次（ultralytics 装进系统 Python）。
一律写 `E:\WhatBird\.venv\Scripts\python.exe`，或先 `.\.venv\Scripts\Activate.ps1`。

### 10.3 本机文件系统偶发抽风（`fsync` / ENOENT / 写入丢失）

会话中遇到过：写工具报 `UNKNOWN: unknown error, fsync`；pwsh 所有命令报
`ENOENT: realpath 'E:\WhatBird'`（工作目录一度完全不可访问）；**期间写的文件没落盘**。
**教训：每次写文件后立刻用 `Test-Path` + `ast.parse` 校验**，别攒着。

### 10.4 `pip install ultralytics` 会顶掉 headless opencv

ultralytics 声明 `opencv-python>=4.7.0`（**GUI 版**），会覆盖本项目刻意选的
`opencv-python-headless`（对焦分析在用；且 headless 才能在没有桌面的环境安装）。
**必须 `--no-deps` 安装**，再手动补 `pyyaml / scipy / psutil / py-cpuinfo / ultralytics-thop`。
pip 随后会警告缺 `opencv-python / polars / nvidia-ml-py` —— **正常现象，别为了消警告去装**。

### 10.5 `pip install --progress-bar raw` 在 pip 23 不合法

只接受 `on` / `off`。写自动化脚本时注意。

### 10.6 pip 拉 XPU 版 torch 会卡死

1.8 GB 的 wheel 只有 `download-r2.pytorch.org` 一个来源；清华/阿里**没有** XPU 版
（实测 404，只有 CPU/CUDA）。单连接拉大文件在本机会中途卡死（实测 3 分钟零字节）。
可断点续传的 `Start-BITSTransfer` 或浏览器下载器更稳。

### 10.7 pip 23 的依赖回溯会卡死

给**未固定版本**的包会让 pip 23 拉一堆元数据卡十几分钟。**永远给定版本号**，
并加 `--only-binary :all:`。

---

## 11. 交接检查单

- [ ] 拿到 `yolo11n.pt` 并放进 `E:\WhatBird\.whatbird\models\`（§6）
- [ ] `whatbird detect PA052099.JPG --model-dir ... --overlay out.png` 跑通，`device == xpu`（§4）
- [ ] **人工看图**确认框落在啄木鸟身上；漏检就调 `--imgsz 960 --conf 0.15` 并记录
- [ ] `requirements-detect.txt` 版本号改成实测值（§7-3）
- [ ] `docs/bird-detection.md` / README 的 torch 版本与实测性能数字更新（§7-4、§7-5）
- [ ] `pytest -q` 全绿（允许样张/权重相关的 skip）
- [ ] 两个 commit 分开提交，**不**把用户的 JPG / 日志 / 预览图提交进去（§8）
- [ ] 交接完成后**更新**这份 HANDOFF（勾掉已完成项、补新发现），不要删
