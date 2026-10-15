# HANDOFF · WhatBird「对焦点分析」交接

> 交接时间：2026-10-11 21:00 左右 ｜ 工作区 `E:\WhatBird`
> 这份文件写给接手的人/agent。**读完之后你应该能不问任何问题就继续干活。**
>
> 📌 **本文件已提交入库**（2026-10-11 由用户确认），所以"用完删除"这条已作废 ——
> 交接完成后请**更新它**而不是删掉：把已完成的条目划掉、把新发现写进去。
>
> 📌 **并行会话**：同一工作区里另一个会话在做「YOLO 鸟类位置检测」，
> 它的交接文档是 **[`HANDOFF-detect.md`](HANDOFF-detect.md)**。两份要一起读 ——
> 共写文件（`image_input.py` / `core/__init__.py` / `cli.py` / `errors.py` / `README.md`）见本文 §8。
> 检测那边的状态（2026-10-11 20:50）：代码与 53 条测试已完成、环境已就绪（torch 2.14.1+xpu，
> XPU 可用），**卡在权重 `yolo11n.pt` 下载**（本机 HTTPS 不通，见其 §6）。

---

## 0. 三十秒速览

**做了什么**：给 WhatBird 加了一个「对焦点分析」功能 —— 给一张 jpg，同时跑两条独立路线并**都上报**：

| 路线 | 数据来源 | 回答 |
|---|---|---|
| `exif` | 相机写进元数据的对焦点 | 相机*声称*对在哪 |
| `content` | 传统图形学（SML 清晰度图 + Otsu 合焦区域） | 画面*实际*最锐处在哪 |

交付面：`core` + CLI `whatbird focus` + Web `/api/focus`、`/api/focus-path` + 网页面板。**纯本地，不联网、不消耗额度。**

**当前状态**：`317 passed, 10 skipped`。区域清晰度评分（§6.1）**已实现**：
`whatbird sharpness` / `detect --sharpness` / Web 的 `sharpness` 开关，见
[`docs/focus-algorithm.md`](docs/focus-algorithm.md) §三。

> 本机换了工作区：**`D:\whatBirds`**（原 `E:\WhatBird`），**没有 Intel 显卡**（CPU-only，
> torch 2.14.1+cpu），环境与下面这张表略有出入，见 §1。

**最要紧的事**：**OM-1 的 EXIF 对焦点已经被解出来了，但还差最后一步确认**——见 §5。用户正在装 exiftool，等他发 `exif-dump.txt`。

---

## 1. 环境与运行

```powershell
cd D:\whatBirds
.\.venv\Scripts\python.exe -m pytest                     # 全量
.\.venv\Scripts\python.exe -m pytest tests/test_focus_*.py -q
.\.venv\Scripts\python.exe -m whatbird.cli focus PA052644.JPG --overlay out.png
.\.venv\Scripts\python.exe -m whatbird.cli sharpness PA052644.JPG --box 120,80,300,240
.\.venv\Scripts\python.exe -m whatbird.cli serve --port 8123
```

实测版本（**本机 D:\whatBirds 实测，不要凭猜改**）：

```
Python 3.10.2 ｜ numpy 2.2.6 ｜ cv2 5.0.0（GUI 版，见下）｜ exifread 3.5.1 ｜ Pillow 11.3.0
```

> ⚠ 与 `requirements.txt` 不一致的两处（都是装 ultralytics 时漏了 `--no-deps` 造成）：
> numpy 钉的是 1.26.4 但实际 2.2.6；`opencv-python`（GUI）与 headless 同时装着、
> `import cv2` 实际加载 GUI 版。对焦与清晰度相关的测试在 numpy 2.2.6 下**全过**，
> 但对齐办法与取舍写在 [`HANDOFF-detect.md`](HANDOFF-detect.md) §3 / §7-4。

---

## 2. 当前状态

**测试**：`266 passed, 10 skipped`。10 条 skip 的构成（别误以为是本功能的问题）：

| 数量 | 来源 | 原因 |
|---|---|---|
| 4 | `test_focus_samples.py` | `samples/` 里还没有真实相机原片（见 §6.3） |
| 1 | `test_focus_samples.py:120` | 本机没装 exiftool，跳过交叉验证 |
| 5 | `test_detect_samples.py` | **另一个会话的**，同样在等 `samples/` |

**git**：**全部未提交**。`M` 的文件 = 我改过的；`??` = 新增。

```
M  README.md  pyproject.toml  requirements.txt  .gitignore
M  tests/conftest.py  tests/test_frontend.py
M  whatbird/cli.py  whatbird/core/__init__.py  whatbird/core/errors.py
M  whatbird/core/image_input.py  whatbird/web/app.py
M  whatbird/web/static/{app.js,index.html,style.css}
?? whatbird/core/focus/    tests/test_focus_*.py    docs/focus-algorithm.md
?? tools/                  samples/
```

---

## 3. 交付内容

### 3.1 core（`whatbird/core/focus/`）

| 文件 | 职责 |
|---|---|
| `models.py` | 数据契约：`FocusParams` / `FocusHit` / `FocusRegion` / `FocusPoint` / `FocusReport` |
| `content.py` | 内容分析路线（opencv + numpy） |
| `exif_focus.py` | EXIF 路线（Pillow 标准标签 + ExifRead MakerNote） |
| `overlay.py` | 标注图渲染（Pillow） |
| `__init__.py` | 门面：`analyze_focus()` / `render_focus_overlay()` |

对外入口（也已从 `whatbird/core/__init__.py` 导出）：

```python
analyze_focus(source, *, params: FocusParams | None = None) -> FocusReport
render_focus_overlay(source, report, *, style=None, max_side=2048) -> bytes   # PNG
overlay_data_url(png_bytes) -> str
```

`source` 接受本地路径 / `bytes` / `ImageSource`（`ImageSource` 同时兼容两者，Web 上传与 CLI 共用）。
`FocusReport.to_dict()` 就是对外 JSON 契约（CLI `--json` 与 REST 共用）。

**坐标约定**：所有 `x`/`y` 都是**按 EXIF Orientation 转正后的原图**像素，原点左上、x 右 y 下；同时给 `x_ratio`/`y_ratio`。

### 3.2 CLI

```powershell
whatbird focus <图片...> [--json] [--overlay 文件] [--overlay-dir 目录]
                [--metric sml|tenengrad] [--max-side 1024] [--top 5]
                [--no-exif] [--no-content]
```

### 3.3 Web

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/focus` | multipart 上传，返回 `{"result": {...}, "overlay": "data:image/png;base64,..."}` |
| POST | `/api/focus-path` | JSON `{"path": "..."}` |

两者都要 `X-WhatBird-Client: 1`（与现有接口一致的 CSRF 防护）。同步返回，**不起后台任务**（亚秒级）。

### 3.4 文档与工具

- `docs/focus-algorithm.md` —— 算法说明、参数、已知边界、实测数据（**改算法必须同步改它**）
- `tools/dump_om_makernote.py` —— OM System MakerNote 原始字段导出器（诊断用，不在产品路径上）
- `samples/README.md` —— 真实样张放置约定

---

## 4. 关键设计决策（附实测依据，别凭感觉改回去）

这一节里每一条都是**实测把原方案推翻之后**定下来的。改动前先读理由。

### 4.1 内容分析：SML → Otsu 区域 → **区域锐度加权质心**

```
灰度 → INTER_AREA 缩放(max_side=1024) → 高斯去噪(σ=1.0)
     → SML: boxFilter(|Laplacian|, win)     win = clip(短边*0.08, 3, 短边/2) 取奇
     → 阈值 thr = max(Otsu(S), 峰值*0.5) → 取含峰值的连通域
     → 对焦点 = 区域内的锐度加权质心（有前提，见 4.2）
```

### 4.2 ❌ 砍掉了「局部对比度归一化」（原方案里有）

原方案打算把清晰度除以周围的清晰度。实测它会把峰值推到清晰区域的**边界**上
（边界处"周围很模糊"导致比值虚高），单像素定位偏差 **40px+**。

也试过"除以局部亮度标准差"当替代：**更糟**——平滑区标准差趋零，峰值全被拽到画面角落，
鸟的百分位反而下降。**现在的做法就是不归一化**，因为二阶导数（拉普拉斯）天然拒绝
失焦的高对比边缘（缓坡的二阶导接近零）。已有回归用例钉死：
`tests/test_focus_content.py::test_blurred_high_contrast_edge_does_not_win`。

### 4.3 ✅ 用质心而不是 argmax —— **但有两条前提**

合成样本上 argmax 偏差 30~50px（清晰块内部本来就有起伏），加权质心是 0~1px。
但真实照片里质心会翻车两次，所以加了两个闸门：

| 闸门 | 参数 | 默认 | 触发场景 |
|---|---|---|---|
| 区域不能铺满画面 | `max_region_for_centroid` | 0.25 | PA052644 的区域占 **43%**，质心 = 画面中心，偏了 850px |
| 质心不能离峰值太远 | `max_centroid_offset` | 2.0（个聚合窗口） | PA052737 区域只占 21%，但它是**稀疏骨架**，质心落在**失焦树枝**上，偏了 1000px |

触发任一闸门就**退回峰值**（也顺便不再上报那个假的"合焦区域"）。

> 试过但**不管用**的两个替代判据，别重走：
> - 「区域在包围盒里的填充率」：合成样本 0.55 vs 真实照片 0.50，界限太窄
> - 「质心处的清晰度」：清晰度是窗口**求和**，旁边有锐东西就抬起来——落点在模糊树枝上的那张仍有 0.63

### 4.4 置信度与「找不到」

```
contrast   = 峰值 / median(S)
confidence = clip(log10(contrast)/log10(contrast_ref), 0, 1)      # contrast_ref = 10
min_confidence = 0.45      # 对应峰值只比中位数高 2.8 倍
```

0.45 是拿真实 OM-1 鸟片校准的：三张照片的 `contrast` 只有 **2.16~2.26**，全都属于
"整片树叶都在焦内"。旧默认值 0.2 会让这种照片"自信地"报出一个贴边的坐标 → 误导。

**关键**：`found=False` 时**仍然算出并列出候选点**，`detail` 里写明最锐处在哪、有几处接近。
图上也会用空心圆 + `SHARPEST` 标出最锐点。"找不到"不等于"没有信息"。

### 4.5 标注图必须限尺寸

一张 5184×3888 原图画无损 PNG 是 **29 MB**（base64 约 39 MB），网页接口绝不能返回。
`render_focus_overlay(max_side=2048)` 默认缩到 4.6MB 级，通过 `coordinate_scale`
把坐标与标记尺寸一起等比换算（有测试）。**报告里的坐标始终是原图像素。**

### 4.6 EXIF 路线分层

| 层 | 覆盖 | 依赖 |
|---|---|---|
| L1 标准标签 `SubjectArea`(0x9214) / `SubjectLocation`(0xA214) | 佳能、尼康、部分索尼 | 仅 Pillow |
| L2 MakerNote | 富士 `FocusPixel`、佳能 `AFInfo2`（自解 uint16 数组） | ExifRead |
| **L3 OM System** | **见 §5，待确认后落地** | **无（自己解）** |

**隐私硬约束**：只白名单透出 `make / model / tag / raw_value`，GPS 一律不出现在任何输出。
`tests/test_focus_exif.py::test_gps_is_never_leaked` 钉死。

**朝向**：先读 EXIF（此时是存储方向），再 `exif_transpose`，坐标按同一变换映射。
`apply_orientation()` 有一条测试**直接拿 `exif_transpose` 的实际输出逐像素比对**。

---

## 5. ⭐ 最重要的开放线索：OM-1 的 EXIF 对焦点

### 5.1 为什么现在读不到

OM-1 的 MakerNote **存在**（20156 字节），但用的是私有头 `"OM SYSTEM\0\0\0"`，
Pillow 与 ExifRead 都会跳过。exifread 跳过的原因精确到一行：

```python
# exifread/core/exif_header.py:549
if make.startswith("OLYMPUS"):     # 实际机身字符串是 "OM Digital Solutions"，匹配不上
```

### 5.2 结构（已实测确认可解）

```
"OM SYSTEM\0" + \0\0 + "II" + version(2)     ← 字节序标记在 MakerNote 内偏移 12
IFD @ MakerNote+16，所有偏移**相对 MakerNote 起点**
  ├─ 0x2010 Equipment        26 个标签   (镜头型号 "OM 75-300mm F4.8-6.7 II" 对得上 ✓)
  ├─ 0x2020 CameraSettings   80 个标签   ← AF 数据在这里
  ├─ 0x2030 RawDevelopment   14 个标签
  ├─ 0x2040 ImageProcessing  80 个标签   (0x0614/0x0615 = 5184/3888 图像尺寸 ✓)
  └─ 0x2050 FocusInfo        80 个标签
```

解析器已经写好：`tools/dump_om_makernote.py`
（`locate_makernote()` + `parse_makernote()`，`--compare --ifd CameraSettings` 出对照表）

### 5.3 已找到的对焦点字段：`CameraSettings 0x030A`

格式：`[grid_w=640, grid_h=480, (cx, cy, w, h), (cx, cy, w, h)]`（**两组，可能一组为空**）

| 照片 | 0x030A | 换算到画面 | 框住了什么 |
|---|---|---|---|
| PA052099 | `[640,480, 270,266,64,104, 270,266,64,104]` | 42.2%, 55.4% | 啄木鸟（头与上身） |
| PA052644 | `[640,480, 306,225,34,48, 306,225,34,48]` | 47.8%, 46.9% | **下面**那只伯劳 |
| PA052737 | `[640,480, 0,0,0,0, 296,213,48,54]` | 46.3%, 44.4% | 山斑鸠的**头** |

可视化已验证：`results/om-af-boxes.png`

### 5.4 ⭐ 有一条独立交叉验证（比字段名更有说服力）

PA052644 里是**两只**伯劳。AF 框落在**下面那只**。

而用完全不相干的**像素清晰度分析**测同一张照片：

| | 清晰度百分位 |
|---|---|
| 下伯劳 | **97.9%**（合焦） |
| 上伯劳 | 66.9%（失焦） |

**两条独立证据链指向同一只鸟。** 这条正好当回归断言用。

### 5.5 ❌ 还差一步：`0x030A` 的语义没定死

**替代解释没排除**：OM-1 有鸟眼/主体识别，`0x030A` 可能是「**主体识别区域**」而不是
「**实际对焦的 AF 点**」。旁边 `0x030B` 三张照片值都不同：

```
PA052099   030A=(270,266,64,104)    030B=(299,244,64,104)
PA052644   030A=(306,225,34,48)     030B=(296,225,34,48)
PA052737   030A=[空, (296,213,48,54)]  030B=[空, 空, ..., 772]
```

**只有 exiftool 的标签名能定死。** 用户正在装 exiftool，等他发：

```powershell
.\exiftool.exe -a -G1 -s PA052099.JPG PA052644.JPG PA052737.JPG > exif-dump.txt
```

⚠️ dump 里会含 **GPS**，要提醒用户删掉那几行再发。

### 5.6 拿到之后怎么做

1. 用字段名定死 `0x030A` / `0x030B` / `FocusInfo 0x0305` 的语义
2. 在 `exif_focus.py` 里加 **OM System 层**（复用 `tools/dump_om_makernote.py` 的解析逻辑，
   把它提升成 `whatbird/core/focus/om_system.py`）
3. 加回归用例：
   - 三张照片的对焦框都必须落在鸟身上
   - **PA052644 必须落在下鸟而不是上鸟**（§5.4 那条交叉验证）
4. 改掉 `unavailable_reason()` 里"该品牌未写入对焦点"的提示

### 5.7 另一条待确认线索

`FocusInfo 0x0305` 是有理数：**1491.5 / 2900.5 / 1487.0**（三张）。变化有规律，像对焦距离（mm），
但按焦距几何反推得到的距离约 28m，对不上 —— **大概率不是距离，字段名确认前别用**。

### 5.8 ⛔ 纪律：没有对照就不写坐标

本功能一路踩的坑（825px、1000px 偏差）全部来自"按直觉解读厂商私有数据"。
**宁可只说"相机记了 AF 数据但本工具还不会换算"，也不要给一个可能错的坐标。**

---

## 6. 其他未完成线程

### 6.1 ✅ 「区域清晰程度」评分 API —— **已实现**（本会话）

用户问"能不能判断一个区域的清晰程度"。当时只有实测结论，现在落地成了：

```python
score_region_sharpness(source, box, *, params) -> RegionSharpness   # 单块
score_regions(source, boxes, *, params) -> RegionScoreReport        # 多块，只建一次图
```

CLI：`whatbird sharpness <图> --box x,y,w,h [--box-ratio …] [--overlay out.png]`；
与检测打通：`whatbird detect <图> --sharpness`（逐只鸟评）、Web `/api/detect` 的 `sharpness` 开关
（页面第 4 个卡片多了「清晰度」列）。
算法、参数、实测数据在 [`docs/focus-algorithm.md`](docs/focus-algorithm.md) §三。

**现状的实测依据（都进了回归测试）**：

- ✅ 同一内容比清晰：SML 单调，逐级模糊 41662 → 1539（27 倍动态范围）；
- ⚠️ 同图跨内容比绝对锐度：**不行**（棋盘格 13847 vs 噪声 4518，3.06 倍）→ 所以只报相对量；
- ❌ 跨照片绝对判定：需要基准标定 → 明确不做；
- ✅ 两个"测不出来"的情况被分开：框内无细节（纹理 σ，纯色 ≈1.1 vs 有内容 ≈5.6）→"无法判断"；
  全图峰值/中位数太小（判据与 `min_confidence` 共用）→"没有唯一锐处"；
- ✅ 真实鸟片：两张翠鸟的检测框都判"合焦"（画面最锐处就在框内，top_share 96~100%）。

**过程中推翻的两个设计（别改回去）**：

1. 一开始把"框内 p75 ÷ 全图单点峰值"当主指标 → 合焦的翠鸟只得到 42%、被判"可能轻微失焦"，
   而且逐级模糊时该比值反而**上升**。根因是两个量不可比（区域稳健分位数 vs 单窗口极大值）。
   现在一律同量纲比："框内最锐处 ÷ 全图最锐处" + "全图最锐 1% 像素落在框内的占比"。
2. 聚合窗口原本固定用短边的 8%（那是给全图找焦点调的）→ 150×120 的小框只有 1.9 个窗口宽，
   被误判成"不可靠"。现在**按框大小自适应选窗口**，并且全图参照用同一个窗口重算。

试过并**否决**的无参考模糊度量（别重走）：
- Crete 比率：同模糊不同内容 0.837 vs 0.497（方向都反了）；渐进模糊下只从 0.761 挤到 0.989
- 边缘宽度：7.25 → 9.00，基本不动

### 6.2 内容分析在满纹理画面里判别力弱（已知边界，非 bug）

三张真实照片全部判为"没有唯一对焦点"，因为都是树叶纹理场景。这是**诚实的结论**，不是失灵。
PA052644 的蓝天场景里它正确区分了两只伯劳（97.9% vs 66.9%），说明有景深时它工作正常。

### 6.3 待真实样张的用例

`tests/test_focus_samples.py` 从 `samples/` 自动发现 jpg，目录为空就整体 skip（当前 4 条）；
第 5 条是本机没装 exiftool 时的交叉验证，装了就会跑。另一个会话的
`tests/test_detect_samples.py` 同样在这等（那 5 条不属于本功能）。

**建议**：把 `PA052099.JPG / PA052644.JPG / PA052737.JPG` 复制进 `samples/`
（`samples/*` 已被 gitignore，不会入库）。放进去后 `test_focus_samples.py` 会立刻跑起来，
而且本机有 exiftool 时会**自动交叉验证** `SubjectArea`。

> 注意：`samples/` 是**两个会话共用**的，放进去会同时点亮 detect 的用例。

### 6.4 没做（用户明确说先不做）

- 不并入 `identify` 流程（对焦点是独立功能）
- 尼康 DSLR 网格 AF 点换算（不同机型点位表，先只上报"相机记了 AF 数据"）
- 索尼 `FocusLocation` 私有坐标系标定

---

## 7. 环境坑（会再踩，记下来）

### 7.1 pip 装不上？→ 这个沙箱的 TLS 是坏的

所有 HTTPS 出网都失败（`schannel: SEC_E_NO_CREDENTIALS`），但**明文 HTTP 通**。
清华镜像只有 HTTPS，所以用不了。可用的是：

```powershell
# 阿里云镜像支持明文 HTTP，且文件链接是同源相对路径
& .\.venv\Scripts\python.exe -m pip install --only-binary :all: `
  --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/ `
  numpy==1.26.4 opencv-python-headless==4.10.0.84 exifread==3.5.1
```

`requirements.txt` 里写的仍然是清华镜像命令（正常环境用那个）。

### 7.2 pip 23 的依赖回溯会卡死

给**未固定版本**的包会让 pip 23 拉一堆元数据卡住十几分钟（`requirements.txt` 顶部注释也写了）。
**永远给定版本号**，并用 `--only-binary :all:`。

### 7.3 用 `opencv-python-headless` 不是 `opencv-python`

不需要 `cv2.imshow`，headless 无 GUI 依赖、体积更小、服务器也能装。

### 7.4 Windows 中文路径 → 别用 `cv2.imread`

`cv2.imread` 遇到中文路径会**静默返回 None**。本功能一律走
`np.frombuffer` + `cv2.imdecode`（`image_input.read_image_bytes` → `open_stored_image`）。

### 7.5 控制台中文乱码是 PowerShell 代码页，不是 bug

CLI 输出的中文在 pwsh 里显示成乱码；Web 那边项目已经有 `UTF8JSONResponse` 处理。
用 Python 直接捕获 stdout 就能看到正确中文。

---

## 8. ⚠️ 这个工作区有**第二个会话**在并行改

同一目录下另一个会话在做 **YOLO 鸟类位置检测**：

```
whatbird/core/detect/{__init__,backend,models,overlay}.py
tests/test_detect_{cli,core,dependency,models,overlay,samples}.py
docs/bird-detection.md
requirements-detect.txt
```

它的交接文档：**[`HANDOFF-detect.md`](HANDOFF-detect.md)**（状态、待办、阻塞点都在那里）。

**共写文件（冲突风险最高）**：
`whatbird/core/image_input.py`、`whatbird/core/__init__.py`、`whatbird/cli.py`、
`whatbird/web/app.py`、`whatbird/web/static/*`

**已经发生过的事**：它重构了 `image_input.py`（新增 `load_upright_rgb` / `encode_png`），
并改写了 `focus/overlay.py` 的 `render_focus_overlay`。我**没有覆盖它的改动**，
而是把尺寸上限加到了它重构后的版本上。

**动手前先 `git status` + 看文件 mtime**。目前 `whatbird focus` 与 `whatbird detect`
两个子命令并存且都能跑（已实测）。

如果两边要合并成"检测框 + 该框锐不锐"，注意它的 `load_upright_rgb()` 文档里已经写了
"保证检测框与对焦点标注落在同一套坐标系"——**现在的实现正好满足**，可以直接叠加。

---

## 9. 附录：三张样张的实测数据

OM-1 / 300mm f/6.7 / 5184×3888

| | PA052099 | PA052644 | PA052737 |
|---|---|---|---|
| 快门 / ISO | 1/320, — | 1/5000, — | 1/400, ISO 2000 |
| 主体 | 大斑啄木鸟（树冠） | 两只伯劳（蓝天） | 山斑鸠（树叶） |
| 峰值/中位数 | 2.26 | 2.24 | 2.16 |
| 判定 | 没有唯一对焦点 | 没有唯一对焦点 | 没有唯一对焦点 |
| 内容路线坐标 | (188, 1729) | (2830, 1883) | (4855, 3701) |
| 该坐标是什么 | 左边缘树枝 | 紧邻下鸟 | 最锐的浆果 |
| EXIF AF 框（§5.3） | 啄木鸟 | **下**伯劳 | 山斑鸠头部 |

**分辨率不敏感**：1024/1536/2048/2592/全分辨率下，`峰值/中位数` 始终 2.3~2.6，
主体百分位始终 ~98%。提高分辨率只是让峰值在几个旗鼓相当的候选间跳 —— 这本身就是
"没有主导峰"的证据。所以默认保持 `max_side=1024`。

**已生成的可视化**（都在 `results/`，已 gitignore）：
`sharpness-heatmap.png`、`compare-1to1.png`、`PA052737-compare.png`、`om-af-boxes.png`、
`PA0520{99,644,737}-focus.png`

---

## 10. 交接检查单

- [ ] 收到 `exif-dump.txt` → 定死 `0x030A` / `0x030B` / `FocusInfo 0x0305` 语义（§5.5）
- [ ] 实现 OM System 层 + 回归用例（含 PA052644 必须命中的断言）（§5.6）
- [ ] 把三张样张放进 `samples/`（§6.3）
- [x] 决定要不要做 `score_region_sharpness()`（§6.1）→ **已做**（含 CLI / detect 打通 / Web 列 / 文档）
- [ ] 与另一个会话确认 `image_input.py` / `core/__init__.py` / `overlay.py` 的合并（§8）
- [ ] `docs/focus-algorithm.md` 与实现保持同步（**改算法必须改文档**）
- [ ] 决定是否提交 git（目前全部未提交）
- [ ] 交接完成后**更新**这份 HANDOFF（勾掉已完成项、补新发现），不要删
