# Agent.md — WhatBird 项目固定载入约定

> 本文件是本仓库的**固定载入内容**：每次会话开始、每收到一个新需求，都必须先读取并遵守以下约定。

## 1. 工作语言

使用 **Python** 作为工作语言。

## 2. 网络环境与包管理

当前处于**中国大陆**。若需要使用 pip 安装依赖，**必须使用清华镜像**（TUNA）：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

永久配置方式（可选，一次配置长期生效）：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 第三方库的使用原则

合理使用第三方库。收到需求后，必须**首先查找是否已有可用的第三方库**，并将「是否使用该第三方库」的结论**询问用户**，得到用户答复后再动手实现。

## 4. 架构

必须使用**前后端分离**的架构。核心能力集中在 `core` 层，UI / CLI 只通过 `core` 暴露的接口调用，保证后续可以拓展 `UI -> core` 或 `CLI -> core` 的能力。

## 5. 版本管理

使用 **git** 进行版本管理。

## 6. 第一版功能

用户提供一个 **jpg 文件路径**，将其发送到火山引擎的 **Doubao-Seed-2.1 系列**模型，参考网页：
<https://console.volcengine.com/ark/region:cn-beijing/model/detail?name=doubao-seed-2-1-lite>，
返回**鸟的种类**，包括**科、属、种**等一系列分类学信息。

## 7. 配置隔离

API 由**用户配置**，一次配置后续均有效。该配置必须和其他信息**隔离**，且**不能被提交到 git 上**。
