# HANDOFF（检测）—— WhatBird 鸟类位置检测（YOLO）

> 交接时间：本文件在 CPU-only 机器（工作区 **`D:\whatBirds`**）上更新过一轮。
> 这是**检测功能**的交接文档；`focus`（对焦点分析）另有一份：[`HANDOFF.md`](HANDOFF.md)。
> 接手前先读 [`Agent.md`](Agent.md)（固定载入约定）与 [`README.md`](README.md) §4.2。
> 📌 本文件已提交入库：交接完成后请**更新它**而不是删掉 —— 勾掉已完成项、把新发现补进去。

---

## 0. 三十秒速览

**做了什么**：给 WhatBird 加「用 YOLO 找出图片里每一只鸟的位置」的能力 ——
给一张 jpg，返回每只鸟的**框 + 置信度**，可画出**原分辨率标注图**。

**交付面**：`core`（`detect_birds()`）+ CLI `whatbird detect` + **Web 界面（`/api/detect`、`/api/detect-path`、页面第 4 个卡片）**
+ 测试 + 两份文档 + 一个走国内镜像的**权重下载脚本**。

**纯本地**：不调用方舟、不消耗额度、不需要 API Key；只有首次需要一份 5.4 MB 的权重。

**当前状态（已验收）**：

| 项 | 状态 |
|---|---|
| 代码 / 测试 | ✅ 全绿：全量 `292 passed, 10 skipped`；检测相关 `76 passed, 5 skipped` |
| 权重 | ✅ 已能自动获取（`tools/fetch_yolo_weight.py` 走镜像） |
| 真机推理（CPU） | ✅ 跑通，`device=cpu` |
| 框落在鸟身上 | ✅ **人工看过**：两张翠鸟样张，92% / 94%，框贴住鸟身 |
| 用户自己的相机原片 | ❌ 本机没有鸟片，还没测（见 §7） |

**最要紧的事**：把你自己拍的鸟片（尤其**竖拍**，`Orientation=6/8`）放进 `samples/`，
用 `tests/test_detect_samples.py` 验证「EXIF 转正后的坐标系」在真实原片上成立。

---

## 1. 交付内容

### 1.1 核心与前端（文件职责）

| 文件 | 职责 |
|---|---|
| `whatbird/core/detect/__init__.py` | 门面：`detect_birds(source, *, params, detector) -> DetectionReport` |
| `whatbird/core/detect/models.py` | 数据契约：`DetectParams` / `RawBox` / `Detection` / `DetectionReport`；归一化框→像素框换算与钳制；权重目录 `default_model_dir()`（认 `WHATBIRD_MODEL_DIR`） |
| `whatbird/core/detect/backend.py` | ultralytics 适配层：**唯一** import torch/ultralytics 的地方；设备解析、权重路径与目录可写性、结果转换、类别过滤、后端缓存 |
| `whatbird/core/detect/overlay.py` | 画框标注图（Pillow）：`render_detection_overlay` / `render_bird_overlay` / `overlay_data_url` |
| `whatbird/web/app.py` | REST：`/api/detect`（multipart + 可选表单字段）、`/api/detect-path`（JSON，`extra="forbid"`） |
| `whatbird/web/static/{index.html,app.js,style.css}` | 页面第 4 个卡片「鸟类位置检测」：标注图 + 结果表 + 其他目标 + 说明 |
| `tools/fetch_yolo_weight.py` | 走镜像下权重：官方 → ghfast → gh-proxy → ghproxy → hf-mirror → ModelScope，逐条校验文件头 |
| `docs/bird-detection.md` | 设计说明：分层、模型与类别、坐标约定、参数、设备、权重、边界、扩展、验证 |
| `requirements-detect.txt` | 可选依赖与安装顺序（ASCII-only，含 CPU / XPU 两条路与 `--no-deps` 的原因） |

### 1.2 测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_detect_models.py` | 契约、参数容错、坐标换算与钳制、摘要措辞 |
| `tests/test_detect_core.py` | 只报鸟、排序、比例换算、钳制、错误翻译、CPU 回退说明、`--all-classes` 语义 |
| `tests/test_detect_dependency.py` | 缺 torch/ultralytics 时 `import whatbird.core` 仍可用、设备解析、权重路径、**类别过滤不下推给模型**（回归）、权重目录不可写的中文提示 |
| `tests/test_detect_overlay.py` | 画框、不改原图、小框/越界框、EXIF 转正后的尺寸 |
| `tests/test_detect_cli.py` | 人类输出、JSON 契约、标注图落盘、参数透传、错误码 |
| `tests/test_detect_api.py` | REST 契约：参数透传（`conf`/`imgsz`/`all_classes`）、缺权重的 400 + 提示、标注图可解码、CSRF 头、大小上限、严格字段 |
| `tests/test_detect_samples.py` | 真实样张（无权重/无样张时整体 skip） |
| `tests/test_frontend.py`（改） | 新增检测面板的 id / 类名 / `/api/detect*` 契约 |

### 1.3 对外接口

```python
from whatbird.core import detect_birds, render_bird_overlay, DetectParams

report = detect_birds(r"D:\photos\bird.jpg")           # 或 bytes / ImageSource / PIL Image
report.count            # 结果条数（默认只算鸟）
report.device           # 实际推理设备：'xpu' 或 'cpu'
report.birds[0].box     # (x1, y1, x2, y2) 原图像素
report.other_classes    # 画面里出现但不算结果的类别名（用来解释"为什么没报鸟"）
report.to_dict()        # 对外 JSON 契约（CLI --json 与 REST 共用）

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
- `backend.py` 是**唯一**重依赖点，`import torch` / `import ultralytics` 全部写在函数内部。
  效果：`import whatbird.core` 不会付出加载 PyTorch 的代价；**没装检测依赖时其余功能完全不受影响**。
- 门面接受 `detector=` 参数，所以绝大多数测试**不需要显卡、不需要权重、不联网**。

### 2.2 后端只输出**归一化** `xyxy`

像素钳制与比例换算只在 `models.xyxy_to_xywh()` / `build_detection()` 一处，
假后端不必构造 640×480 的真实框，真后端也不必重复写钳制逻辑。

### 2.3 ⚠ 类别过滤**只能在 core 层做**（本次修掉的真 bug）

预训练权重是 COCO 80 类，`bird` 是 id 14。默认 `classes=(14,)` → 只有鸟算结果，
其他类别进 `report.other_classes` 并附中文说明（"画面里还有非鸟目标…"）。

**曾经写成**把 `classes=[14]` 直接传给 `predict()`，结果模型层的过滤让非鸟框根本回不到上层，
于是**默认参数下 `other_classes` 永远是空的** —— 文档承诺的"解释为什么没报鸟"实际不成立，
只有加 `--all-classes` 才有内容。真机表现：一张有公交车和人的照片，默认参数下 `other_classes=[]`。

现在 `YoloDetector.__call__` 固定传 `classes=None`，过滤统一在 `build_detections()` 里按
`params.classes` 做。**不会变慢**：ultralytics 本来就是先推理、再按类别筛，`classes` 只过滤
NMS 之后的列表。回归测试：`test_backend_never_pushes_the_class_filter_down_to_the_model`。

`--all-classes` 的语义因此变得有意义：关掉结果过滤，非鸟目标也进结果表并画框
（行内标出类别名，摘要说"其中 N 只鸟"，不谎报成鸟）。

### 2.4 坐标口径与 `focus` 一致

`detect_birds()` 与 `render_bird_overlay()` 都走 `image_input.load_upright_rgb()`
（`exif_transpose` → RGB），所以检测框与对焦点标注落在**同一套坐标系**（转正后的原图像素），
并且都提供 0~1 的比例字段供前端叠加。**竖拍照片不转正就会错位**，这条有测试。

### 2.5 设备策略：`auto` 回退，显式 `xpu` 不静默回退

- `auto` → 有 XPU 用 XPU，否则 CPU；显式 `--device xpu` 而不可用 → **直接报错**；
- 实际设备写进 `report.device`，并在 `notes` 里说明回退原因；
- 判定抽成 `backend.xpu_available()`，两条分支都有测试覆盖。
- 当前机器**没有 Intel 显卡**：`torch 2.14.1+cpu` 里 `torch.xpu` 是 `None`，`auto` 必然落到 `cpu`。

### 2.6 权重按 `(路径, 设备)` 缓存

`backend.get_detector()` 带锁缓存，CLI 一次给多张图不会反复加载模型。
**这是性能关键**：单张推理只要几十毫秒，而加载权重约 6 秒（见 §5）。

### 2.7 权重目录可用环境变量覆盖

Web 端没有 `--model-dir` 可传，所以 `DetectParams` 的默认目录改成
`models.default_model_dir()`：先看环境变量 `WHATBIRD_MODEL_DIR`，再看 `~/.whatbird/models`。
受限环境（沙箱、只读用户目录、权重放项目内）靠这个把服务端指到已有权重。

### 2.8 标注图文字只用 ASCII

不能假设用户机器装了中文字体。中文说明放在 CLI 输出与网页文本里，不烧进图片。
图例在结果含非鸟目标时写 `N OBJECT(S) DETECTED | M BIRD(S)`，不谎报成鸟。

---

## 3. 环境现状（本机实测）

| 项 | 实测值 |
|---|---|
| 解释器 | `D:\whatBirds\.venv\Scripts\python.exe`（**Python 3.10.2**） |
| torch / torchvision | **2.14.1+cpu / 0.29.1**（CPU 版；`hasattr(torch,'xpu')` 为 True 但 `torch.xpu is None`） |
| ultralytics | **8.4.177**（注意：不是旧文档里的 8.4.175） |
| 其它 | PyYAML 6.0.3、psutil 7.2.2、ultralytics-thop 2.2.2、pillow 11.3.0、exifread 3.5.1 |
| numpy | ⚠ **2.2.6**（`requirements.txt` 钉的是 1.26.4） |
| opencv | ⚠ **`opencv-python 5.0.0.93`（GUI 版）与 `opencv-python-headless 4.10.0.84` 同时装着**，`import cv2` 实际加载的是 GUI 版 5.0.0 |
| scipy / pandas | ❌ 未安装（ultralytics 声明了，但推理路径不 import，实测跑得通） |
| 权重 | `D:\whatBirds\.whatbird\models\yolo11n.pt`（5.4 MB，项目内目录，已 gitignore） |
| ultralytics 设置目录 | 用 `YOLO_CONFIG_DIR` 指到 `.whatbird`（见 §10.2） |

> ⚠ 上面两条"⚠"是**环境与文档不一致**的地方，本轮没动它们（改 venv 需要放宽权限）。
> 影响评估：对焦分析在 numpy 2.2.6 下测试全过；GUI 版 opencv 顶掉 headless 是
> [`requirements-detect.txt`](requirements-detect.txt) 明确要避免的（装 ultralytics 时漏了 `--no-deps`）。

---

## 4. 现在怎么跑（照抄即可）

```powershell
cd D:\whatBirds
$py = ".\.venv\Scripts\python.exe"

# 1) 权重（已下好；换机器/换目录时重跑）
$py tools\fetch_yolo_weight.py --dest D:\whatBirds\.whatbird\models

# 2) 让 ultralytics 的设置目录可写（受限环境/沙箱下必须；否则 import 时就 PermissionError）
$env:YOLO_CONFIG_DIR = "D:\whatBirds\.whatbird"

# 3) CLI 检测 + 标注图
$py -m whatbird.cli detect D:\photos\bird.jpg --model-dir D:\whatBirds\.whatbird\models --overlay D:\out\bird.png

# 4) Web 端（服务端没有 --model-dir，用环境变量指权重目录）
$env:WHATBIRD_MODEL_DIR = "D:\whatBirds\.whatbird\models"
$py -m whatbird.cli serve --port 8000
#   浏览器打开 http://127.0.0.1:8000 -> 页面第 4 个卡片
```

---

## 5. 验证结果与实测数字

| 验证 | 结果 |
|---|---|
| `pytest`（全量，不受限运行） | ✅ **292 passed, 10 skipped** |
| `pytest tests/test_detect_*.py` | ✅ **76 passed, 5 skipped**（5 条是缺样张/权重的真实样张测试） |
| CLI `detect` 真机（CPU） | ✅ `device=cpu`，框正确 |
| `POST /api/detect` 真机 | ✅ 200，返回 `result` + `overlay`（原分辨率 PNG，1.3 MB base64） |
| 框精度（人工看标注图） | ✅ 翠鸟 640×427：1 只 92%（x=229 y=21 w=335 h=386）；翠鸟 1200×801：1 只 94%（x=571 y=156 w=437 h=513）——"框贴住鸟身，无多余框" |
| 反例（画面里没有鸟） | ✅ `bus.jpg`：`count=0`，`other_classes=['bus','person']`，说明文字正确 —— 这正是 §2.3 修好之后才成立的行为 |
| `fetch_yolo_weight.py` | ✅ 官方源超时后自动换 `ghfast.top`，5.4 MB 下载成功并通过文件头校验 |

**耗时（同一台 CPU-only 机器，810×1080 的图）**：

| 场景 | 耗时 |
|---|---|
| 冷启动（import torch + 加载权重 + 一次推理） | **6.17 s** |
| 之后每张，`--imgsz 640` | 中位 **45 ms**（43–98） |
| 之后每张，`--imgsz 960` | 中位 **72 ms**（70–105） |
| 之后每张，`--imgsz 1280` | 中位 **124 ms**（117–153） |

结论：**"CLI 单张要 6 秒"几乎全是进程启动与加载权重的成本，推理本身只要几十毫秒**。
所以调 `--imgsz 960` 抓远处小鸟的代价比想象中小得多；反复起进程跑单张才是浪费
（批量走一次加载、多次推理）。

---

## 6. 权重下载（已解决）

`yolo11n.pt` 约 5.4 MB，官方地址在 GitHub Releases，**国内基本连不上**（实测：
`github.com` 连接被重置、`upload.wikimedia.org` 也超时）。本次逐个探测后确认可用镜像：

| 地址 | 实测 |
|---|---|
| `https://ghfast.top/<官方URL>` | ✅ 206 + `PK` 头 |
| `https://gh-proxy.com/<官方URL>` | ✅ |
| `https://ghproxy.net/<官方URL>` | ✅ |
| `https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11n.pt` | ✅ |
| `https://www.modelscope.cn/api/v1/models/Ultralytics/YOLO11/repo?Revision=master&FilePath=yolo11n.pt` | ✅ |

已固化成 `tools/fetch_yolo_weight.py`（按上表顺序重试 + 校验文件头 + `--list` 预览），
`requirements-detect.txt` 与文档都指向它。**不需要再手工下载**。

---

## 7. 待办清单（按优先级）

1. **[重要]** 放一张**你自己的鸟片**（最好有竖拍）到 `samples/`，在
   [`samples/README.md`](samples/README.md) 的表里写"大概几只鸟"，然后跑
   `pytest tests/test_detect_samples.py -v`。这能验证公开网图验证不了的东西：
   EXIF 转正坐标系 + 高像素原片 + 枝叶遮挡下的漏检情况。
2. **[建议]** 若遮挡场景漏检，记录实际 `--imgsz` / `--conf` 组合（`--imgsz 960 --conf 0.15` 起步）并写进文档。
3. **[可选]** 网页端目前只用默认参数（`conf 0.25` / `imgsz 640`）；接口已经支持
   `conf` / `imgsz` / `all_classes`，可以给卡片加两个输入框。
4. **[建议]** 环境对齐（需要往 `.venv` 里写，受限终端下要放宽权限或由你手动执行）：
   ```bat
   .venv\Scripts\python.exe -m pip uninstall -y opencv-python
   .venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.10.0.84
   ```
   然后决定 numpy 走哪条路：装回 `1.26.4`（与 `requirements.txt` 一致）或把
   `requirements.txt` 的钉版改成 2.2.6（对焦测试在 2.2.6 下全过，但文件里写着"升 2.x 前先重跑对焦测试"）。
5. **[可选]** 加回归测试钉死「`cv2` 必须是 headless 版」，防止再被 GUI 版顶掉（这次就是被顶掉了）。
6. **[可选]** 检测 → 逐只裁剪 → 送 Doubao 识别种类：`Detection.box` 已给出 `(x1,y1,x2,y2)`，
   `BirdRecognizer.recognize_source()` 接受内存字节，不需要改现有识别代码。
7. **[可选]** 换推理后端：`export` 转 OpenVINO/ONNX，再实现一个满足 `Detector` 协议的后端。
8. **[建议]** git 提交本轮改动（见 §8）。

---

## 8. git 状态与提交建议

分支 `main`。本轮改动**尚未提交**（`git status`：17 个文件修改 + 2 个新增）。

**新增**：`tests/test_detect_api.py`、`tools/fetch_yolo_weight.py`。

**修改**：`.gitignore`、`README.md`、`docs/bird-detection.md`、`requirements-detect.txt`、
`whatbird/cli.py`、`whatbird/core/detect/{backend,models,overlay}.py`、`whatbird/web/app.py`、
`whatbird/web/static/{app.js,index.html,style.css}`、`tests/test_detect_{cli,core,dependency,models}.py`、
`tests/test_frontend.py`。

建议拆成两个 commit（沿用仓库中文 conventional-commit 风格）：

```
fix(detect): 类别过滤改回 core 层，修复 other_classes 永远为空
feat(web): 位置检测接入网页端 + 走镜像的权重下载脚本
```

⚠️ **不要 `git add -A`**：`.whatbird/`（配置、权重、验证样张、临时脚本）与 `samples/*`
都由 `.gitignore` 挡住，别强行加；本机没有用户的相机原片，也不要把公开网图（Pexels / Pixabay /
Unsplash / COCO）提交进仓库 —— 那是第三方图片，只用于本地验收。

---

## 9. 与 focus 会话共写文件

上一轮是两个会话并行改同一工作区，共写文件（`image_input.py`、`focus/overlay.py`、
`core/__init__.py`、`cli.py`、`errors.py`、`README.md`）。**本轮没有并行会话**，
但本轮又动了 `cli.py`、`README.md` 与 `core/detect/*`、`web/*`；
如果别处同时在改 focus，合并时优先看这几处。

---

## 10. 踩过的坑（别再踩）

### 10.1 PowerShell `Set-Content -Encoding utf8` 会给源码加 BOM

首行变成 `\ufeff"""` → **Python 直接 SyntaxError**。写文件要无 BOM：

```powershell
[System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
```

### 10.2 ultralytics 在 `import` 时就要写设置目录

`import ultralytics` → `get_user_config_dir()` → `%APPDATA%\Ultralytics` 建目录 + 写 settings.json。
受限环境（沙箱、只读用户目录）下这一步就 `PermissionError`，**整个检测功能在 import 阶段就崩**，
看起来像"装坏了"。两个后果：

- 跑任何涉及检测的命令前先 `set YOLO_CONFIG_DIR=<可写目录>`（本机用 `.whatbird`）；
- 没有这个变量时它会退回到**当前工作目录**下的 `Ultralytics/`，于是仓库根目录会多出一个
  未跟踪目录 —— `.gitignore` 已加 `Ultralytics/` 兜底。

### 10.3 本机全路径调用解释器

在项目目录里直接敲 `python` 命中的可能是**系统 Python**，包装错地方。
一律写 `.venv\Scripts\python.exe`，或先 `.\.venv\Scripts\Activate.ps1`。

### 10.4 `pip install ultralytics` 会顶掉 headless opencv

ultralytics 声明 `opencv-python>=4.7.0`（**GUI 版**），会覆盖本项目刻意选的
`opencv-python-headless`。**必须 `--no-deps` 安装**，再手动补
`pyyaml / psutil / py-cpuinfo / ultralytics-thop`。本机当前同时装着 GUI 版与 headless 版
（§3 的 ⚠），`import cv2` 实际加载 GUI 版 5.0.0，属于该坑的实例。

### 10.5 升级 numpy 前先重跑对焦测试

`requirements.txt` 钉 `numpy==1.26.4` 并注明"升 2.x 前重跑对焦测试"。本机现在装的是 2.2.6
（`opencv-python 5.0.0.93` 带上去的），对焦与检测测试都全过 —— 但两条信息要一起看：
`requirements.txt` 的钉版和实际环境已经不一致。

### 10.6 本机受限终端以 **Low 完整性级别**运行子进程

现象极具误导性：**能新建文件，却改不了已存在的文件**；`whatbird/` 里连建文件都失败，
但仓库根目录能建。原因不是 ACL（`Get-Acl` 显示 `Authenticated Users: Modify`），
而是沙箱把工作区根标成 Low、子进程以 Low 跑：新建对象继承 Low（可写），
**已存在的对象是 Medium（不可写，no-write-up）**。

对策：需要写目录时用**新建**目录（本机 `.whatbird/models`、`.whatbird/val`、`.whatbird/out`
都是这么来的）；文件编辑走编辑工具（不受这个限制）；需要在仓库已有目录里写文件
（例如 `pytest` 的 config 落点探针用例）时，必须放宽权限运行，否则那条用例必然失败。

### 10.7 `pytest -q` × `pyproject.toml` 的 `addopts = "-q"` 会吞掉统计行

命令行再加一个 `-q` 就变成 `-qq`，最后那行 `292 passed, 10 skipped` **不会打印**，
看着像"跑完了但不知道结果"。要统计就别再加 `-q`（或 `-rA`）。

### 10.8 COCO val2017 的图片 ID 很稀疏；HF 上的鸟类数据集没有直链图片

想找"确定有鸟"的公开样张时踩的：`images.cocodataset.org` 通（可以用），
但按步长扫 ID 命中率极低（扫 218 个 ID 只落到 2 张图）；COCO 的 captions 标注没有独立小文件
（`captions_val2017.zip` 404，要下 241 MB 的 `annotations_trainval2017.zip`）。
HF 上搜到的鸟类数据集（`merve/bird-species` 等）只有 parquet，没有可直接下载的 jpg。
**实测可用的公开图源**：`images.pexels.com`、`cdn.pixabay.com`、`images.unsplash.com`
（URL 加 `?w=1200` 控制大小）——本次验收的两张翠鸟就是这么来的。

---

## 11. 交接检查单

- [x] 拿到 `yolo11n.pt` 并放进 `D:\whatBirds\.whatbird\models\`（§6，已固化成脚本）
- [x] `whatbird detect` 真机跑通（`device == cpu`，本机无 Intel 显卡）
- [x] 人工看标注图确认框落在鸟身上（两张公开翠鸟样张，92% / 94%）
- [x] `requirements-detect.txt` 版本号改成实测值 + CPU/XPU 两条路（§3、§6）
- [x] `docs/bird-detection.md` / README 的设备、权重、性能、类别语义全部更新（§5）
- [x] 全量 `pytest` 全绿（292 passed / 10 skipped）
- [x] 修掉"类别过滤下推给模型导致 `other_classes` 恒空"的真 bug，并补回归测试（§2.3）
- [x] Web 端接入检测（`/api/detect`、`/api/detect-path`、页面第 4 个卡片 + 契约测试）
- [ ] **把用户自己的鸟片放进 `samples/` 再跑一次真机验收**（§7-1，只有你能做）
- [ ] 本轮改动 git 提交（§8）
- [x] 交接完成后更新本文件（这次就是）
